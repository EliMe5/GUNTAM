"""GUNTAM v2 training entrypoint: SeedEdgeTransformer + per-edge BCE.

Mirrors Train.py's file/event/bin-batch loop but drives the v2 stack:

* model  : SeedEdgeTransformer (in-graph rotation + candidate masking); the
           per-bin phi center is derived from the bin index and passed as a
           model input.
* truth  : layer-aware directed edge tensor (PrepareTensor --edge_truth).
* loss   : SeedLoss.edge_bce_loss over the candidate set (class-balanced,
           impossible truth edges pruned and counted).
* eval   : batched beam search over the top-5 triplets + a quick seed-level
           truth-match efficiency (>= 3 same-particle hits).

The legacy Train.py / SeedTransformer path is untouched. Usage:

    python -m GUNTAM.Seed.TrainEdge \
        --input_path /shared/elmenard/guntam_v2_training_data/v2_YYYYMMDD \
        --input_format parquet --input_tensor_path <tensor dir> \
        --sort_by_m --edge_truth \
        --hit_features x y z r phi eta m layer_key \
        --particle_features eta phi pT \
        --epoch_nb 20 --batch_size 2 --device cuda \
        [--edge_config edge_config.json] [--export_onnx model.onnx]
"""

import argparse
import contextlib
import math
import os
import random
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import GUNTAM.Seed.Reconstruction as Reconstruction
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.IO.PrepareTensor import compute_barcode, prepare_tensor
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.SeedEdgeTransformer import EdgeTransformerConfig, SeedEdgeTransformer
from GUNTAM.Seed.SeedLoss import edge_bce_loss
from GUNTAM.Transformer.Utils import ts_print
import GUNTAM.Transformer.Utils as Utils

REQUIRED_HIT_FEATURES = ["x", "y", "z", "r", "phi", "eta", "m", "layer_key"]


def bin_phi_centers(bin_indices: torch.Tensor, nb_bins: int) -> torch.Tensor:
    """Phi center of each bin. Binning spans (-pi, pi) with nb_bins bins, so the
    EFFECTIVE width is 2*pi/nb_bins (not the nominal cfg.bin_width)."""
    width_eff = 2.0 * math.pi / nb_bins
    return (-math.pi + (bin_indices.float() + 0.5) * width_eff)


def train_edge_model(
    model: SeedEdgeTransformer,
    train_file_indices: list,
    dataset: DataLoader,
    cfg: SeedConfig,
    writer: SummaryWriter,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    start_epoch: int = 0,
    use_amp: bool = False,
) -> SeedEdgeTransformer:
    """Train the edge model. Same file/event/bin-batch structure as Train.py.

    use_amp: bfloat16 autocast on CUDA (~1.5-2x throughput on H100; bf16 needs no
    GradScaler, and BCE-with-logits is autocast-upcast to fp32 by torch).
    """
    ts_print(f"Training SeedEdgeTransformer from epoch {start_epoch} to {start_epoch + cfg.epoch_nb}")
    if use_amp:
        ts_print("AMP enabled: bfloat16 autocast")

    def amp_ctx():
        if use_amp and cfg.device_acc.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    if optimiser and scheduler:
        scheduler.step()
        ts_print(f"Initial learning rate: {scheduler.get_last_lr()}")

    for epoch in range(start_epoch, start_epoch + cfg.epoch_nb):
        ts_print("Epoch: ", epoch)
        entry = 0
        epoch_train_losses: List[float] = []
        epoch_val_losses: List[float] = []
        epoch_pruned = 0
        epoch_pos = 0

        files = list(train_file_indices)
        n_val_files = int(cfg.val_fraction * len(files)) if hasattr(cfg, "val_fraction") else 0
        val_files_set = set(files[-n_val_files:]) if n_val_files > 0 else set()

        for file_idx in files:
            status = "Validation" if file_idx in val_files_set else "Training"
            model.eval() if status == "Validation" else model.train()

            batch_data = dataset.get_file(file_idx)
            hits_tensor = batch_data["hits_tensor"].to(cfg.device_acc, dtype=torch.float32)
            padding_mask = batch_data["padding_mask"].to(cfg.device_acc)
            good_pairs = batch_data["good_pairs"].to(cfg.device_acc)

            nb_bins = hits_tensor.shape[1]
            event_indices = list(range(hits_tensor.shape[0]))
            random.shuffle(event_indices)

            for event_idx in event_indices:
                ev_hits = hits_tensor[event_idx]      # [nb_bins, N, F]
                ev_pairs = good_pairs[event_idx]      # [nb_bins, P, 3]
                ev_mask = padding_mask[event_idx]     # [nb_bins, N]

                valid_bins = torch.where(ev_pairs[..., 2].sum(dim=1) > 0)[0].tolist()
                event_loss = 0.0
                num_valid_bins = 0
                grad_enabled = status == "Training"

                with torch.set_grad_enabled(grad_enabled):
                    for batch_start in range(0, len(valid_bins), cfg.batch_size):
                        bin_idx = valid_bins[batch_start : batch_start + cfg.batch_size]
                        if not bin_idx:
                            continue
                        bins_t = torch.tensor(bin_idx, device=cfg.device_acc)
                        centers = bin_phi_centers(bins_t, nb_bins)

                        with amp_ctx():
                            _, logits, cand, _ = model(ev_hits[bin_idx], ev_mask[bin_idx], centers)

                            batch_loss = torch.tensor(0.0, device=cfg.device_acc)
                            for i in range(len(bin_idx)):
                                p1, p2, w = ev_pairs[bin_idx[i]].unbind(dim=1)
                                loss_i, stats = edge_bce_loss(logits[i], cand[i], p1, p2, w)
                                batch_loss = batch_loss + loss_i
                                epoch_pruned += stats["n_pos_pruned"]
                                epoch_pos += stats["n_pos"]
                                num_valid_bins += 1

                        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                            raise ValueError(
                                f"Loss became NaN/Inf at epoch {epoch}, event {entry}, "
                                f"file {file_idx}, bins {bin_idx}"
                            )

                        if status == "Training":
                            optimiser.zero_grad()
                            batch_loss.backward()
                            optimiser.step()
                        event_loss += batch_loss.detach().item()

                if num_valid_bins > 0:
                    event_loss /= num_valid_bins
                if writer:
                    writer.add_scalar(f"edge_bce/{status}", event_loss, epoch * 10000 + entry)
                    if optimiser and scheduler:
                        writer.add_scalar(f"learning_rate/{status}", optimiser.param_groups[0]["lr"], epoch * 10000 + entry)
                (epoch_train_losses if status == "Training" else epoch_val_losses).append(event_loss)
                entry += 1

        if epoch_train_losses:
            avg = sum(epoch_train_losses) / len(epoch_train_losses)
            ts_print(f"Epoch {epoch} - Avg Training edge BCE: {avg:.6f} ({len(epoch_train_losses)} events)")
            if writer:
                writer.add_scalar("loss_epoch/Training", avg, epoch)
        if epoch_val_losses:
            avg = sum(epoch_val_losses) / len(epoch_val_losses)
            ts_print(f"Epoch {epoch} - Avg Validation edge BCE: {avg:.6f} ({len(epoch_val_losses)} events)")
            if writer:
                writer.add_scalar("loss_epoch/Validation", avg, epoch)
        if epoch_pos + epoch_pruned > 0:
            frac = epoch_pruned / (epoch_pos + epoch_pruned)
            ts_print(f"Epoch {epoch} - truth edges pruned by candidate cuts: {epoch_pruned} ({100 * frac:.3f}%)")
            if writer:
                writer.add_scalar("edges/pruned_fraction", frac, epoch)

        # Full-dataset epochs run for many hours: checkpoint after EVERY epoch
        backup = cfg.model_path.replace(".pt", f"_backup_epoch_{epoch + 1}.pt")
        model.save(epoch=epoch, path=backup, optimizer=optimiser, scheduler=scheduler)
        ts_print(f"Saved backup checkpoint to {backup}")
        if optimiser and scheduler:
            scheduler.step()

    return model


@torch.inference_mode()
def evaluate_edge_model(
    model: SeedEdgeTransformer,
    dataset: DataLoader,
    test_file_indices: list,
    cfg: SeedConfig,
    att_threshold: float = 0.4,
    max_eval_events: int = 50,
) -> Dict[str, float]:
    """Quick seed-level evaluation: beam search over triplets, truth-match rate.

    A seed (3-hit chain) is 'good' if >= 3 of its hits share a particle id;
    a particle is 'found' if at least one good seed matches it. This is a
    training-time sanity metric, not the full ACTS efficiency chain.
    """
    model.eval()
    n_seeds = n_good = 0
    found_particles = set()
    truth_particles = set()
    events_done = 0

    for file_idx in test_file_indices:
        if events_done >= max_eval_events:
            break
        batch_data = dataset.get_file(file_idx)
        hits_tensor = batch_data["hits_tensor"].to(cfg.device_acc, dtype=torch.float32)
        padding_mask = batch_data["padding_mask"].to(cfg.device_acc)
        h2p = batch_data["hit_to_particle_tensor"]
        nb_bins = hits_tensor.shape[1]

        for event_idx in range(hits_tensor.shape[0]):
            if events_done >= max_eval_events:
                break
            ev_hits = hits_tensor[event_idx]
            ev_mask = padding_mask[event_idx]

            # Forward pass batched by cfg.batch_size, same as training -- the dense
            # [B,N,N,H] edge tensor OOMs if B=nb_bins (~126) is forwarded in one shot.
            triplets_chunks = []
            for batch_start in range(0, nb_bins, cfg.batch_size):
                bin_idx = list(range(batch_start, min(batch_start + cfg.batch_size, nb_bins)))
                bins_t = torch.tensor(bin_idx, device=cfg.device_acc)
                centers = bin_phi_centers(bins_t, nb_bins)
                _, _, _, triplets_chunk = model(ev_hits[bin_idx], ev_mask[bin_idx], centers)
                triplets_chunks.append(triplets_chunk)
            triplets = torch.cat(triplets_chunks, dim=0)
            valid_mask = ~ev_mask.bool().reshape(nb_bins, -1)
            chains, _, scores = Reconstruction.batched_beam_search_seed_reconstruction(
                triplets.float(), valid_mask, att_threshold=att_threshold,
                max_chain_length=3, beam_width=5,
            )
            chains = chains.cpu().numpy()
            scores = scores.float().cpu().numpy()
            htp_ev = h2p[event_idx].cpu().numpy().reshape(nb_bins, -1)

            for b in range(nb_bins):
                pids_bin = htp_ev[b]
                truth_particles.update((events_done, int(p)) for p in np.unique(pids_bin) if p >= 0)
                ok = np.isfinite(scores[b]) & (scores[b] > att_threshold)
                for i in np.where(ok)[0]:
                    chain = chains[b, i]
                    chain = chain[chain >= 0]
                    if len(chain) < 3:
                        continue
                    n_seeds += 1
                    pids, counts = np.unique(pids_bin[chain], return_counts=True)
                    hit3 = pids[(pids >= 0) & (counts >= 3)]
                    if len(hit3) > 0:
                        n_good += 1
                        found_particles.add((events_done, int(hit3[0])))
            events_done += 1

    metrics = {
        "n_seeds": float(n_seeds),
        "purity": n_good / n_seeds if n_seeds else 0.0,
        "efficiency": len(found_particles) / len(truth_particles) if truth_particles else 0.0,
        "n_truth_particles": float(len(truth_particles)),
    }
    ts_print(
        f"Eval ({events_done} events): {n_seeds} seeds, purity {100 * metrics['purity']:.1f}%, "
        f"particle efficiency {100 * metrics['efficiency']:.1f}% "
        f"({len(found_particles)}/{len(truth_particles)})"
    )
    return metrics


def main():
    # Edge-specific args are stripped before SeedConfig sees the command line
    edge_parser = argparse.ArgumentParser(add_help=False)
    edge_parser.add_argument("--edge_config", type=str, default=None,
                             help="JSON file with EdgeTransformerConfig overrides")
    edge_parser.add_argument("--export_onnx", type=str, default=None,
                             help="Export the trained model to this ONNX path after training")
    edge_parser.add_argument("--amp", action="store_true", default=False,
                             help="bfloat16 autocast on CUDA for the training forward/backward")
    edge_args, remaining = edge_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    cfg = SeedConfig()
    cfg.parse_args()
    ts_print("Starting GUNTAM v2 edge-model training")
    ts_print(f"Using device: {cfg.device_acc}")

    pp = cfg.preprocessing_config
    missing = [f for f in REQUIRED_HIT_FEATURES if f not in pp.hit_features]
    if missing:
        raise ValueError(
            f"v2 training requires hit_features {REQUIRED_HIT_FEATURES}; missing {missing}. "
            "Pass --hit_features x y z r phi eta m layer_key"
        )
    if not pp.edge_truth or not pp.sort_by_m:
        raise ValueError("v2 training requires --edge_truth and --sort_by_m")

    if cfg.device_acc.type == "cuda":
        torch.set_float32_matmul_precision("high")

    mcfg = EdgeTransformerConfig()
    mcfg.hit_feature_names = list(pp.hit_features)
    if edge_args.edge_config:
        mcfg.load_config(edge_args.edge_config)
    model = SeedEdgeTransformer(mcfg, device_acc=cfg.device_acc)
    model.to(cfg.device_acc)
    model.print_model_info()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = Utils.create_cosine_schedule_with_min_lr(
        opt,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=cfg.num_training_steps,
        min_lr_ratio=cfg.min_lr_ratio,
    )

    barcode = compute_barcode(pp)
    metadata_path = f"{cfg.input_tensor_path}/metadata_{cfg.dataset_name}_{barcode}.pt"
    if not os.path.exists(metadata_path) or cfg.recompute_tensor:
        ts_print("Computing tensor dataset, barcode: ", barcode)
        prepare_tensor(pp)

    dataset = DataLoader(
        dataset_dir=cfg.input_tensor_path,
        dataset_name=f"{cfg.dataset_name}_{barcode}",
        tensor_names=["hits_tensor", "particles_tensor", "hit_to_particle_tensor", "padding_mask", "good_pairs"],
        device=cfg.device_acc,
    )

    num_files = dataset.get_file_number()
    train_files = math.ceil((1 - cfg.test_fraction) * num_files)
    train_file_indices = list(range(train_files))
    test_file_indices = list(range(train_files, num_files))
    ts_print(f"{num_files} tensor files: {len(train_file_indices)} train, {len(test_file_indices)} test")

    start_epoch = 0
    if cfg.resume_training:
        ts_print(f"Resuming from {cfg.model_path}")
        start_epoch = model.load(cfg.model_path, cfg.device_acc)

    writer = SummaryWriter("training_edge_seeding")

    if cfg.epoch_nb > 0:
        model = train_edge_model(
            model, train_file_indices, dataset, cfg, writer, opt, scheduler,
            start_epoch=start_epoch, use_amp=edge_args.amp,
        )
        model.save(epoch=start_epoch + cfg.epoch_nb - 1, path=cfg.model_path, optimizer=opt, scheduler=scheduler)
        ts_print(f"Saved model to {cfg.model_path}")
    writer.close()

    if not cfg.no_test and test_file_indices:
        evaluate_edge_model(model, dataset, test_file_indices, cfg)

    if edge_args.export_onnx:
        model.export_onnx(edge_args.export_onnx)


if __name__ == "__main__":
    sys.exit(main())
