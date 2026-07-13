"""GUNTAM v2 edge-classification seeding model.

Locked v2 architecture (July 2026):

* Inputs are spacepoints of one phi bin, SORTED BY ``m = (R + rho)/2``; the
  order defines "forward" for the candidate mask and beam search.
* Per-bin canonical-frame renormalisation happens IN-GRAPH: the bin's phi
  center is an explicit model input, ``phi_rel = wrap(phi - phi_center)``,
  ``x_rel = r cos(phi_rel)``, ``y_rel = r sin(phi_rel)``. x' and y' use
  distinct normalisation scales (x' up to ~500 mm radially, y' only up to
  ~half-bin transverse offset, ~13 mm at bin_width 0.05).
* Fourier embedding, k = 0..14 (15 bands, computed in float64 then cast):

      [ {sin/cos(2pi 2^k x_rel/Sx)}, {.. y_rel/Sy}, {.. z/Sz}, {.. r/Sr},
        {.. phi_rel/Sphi}, phi_rel, r, m, z, eta ]

* Transformer encoder (unchanged blocks), then a DENSE MASKED N x N edge
  head: pair features [h_i, h_j, deltas] -> MLP -> logit[i, j]. The first MLP
  layer is factorised (W_i h_i + W_j h_j + W_d deltas) so no [N, N, 2d] tensor
  is ever materialised.
* Candidate-edge rules computed IN-GRAPH so they export to ONNX:
      m_j - m_i > edge_min_dm   (forward + minimum edge length, locked 5 mm)
      layer_key_i != layer_key_j (VIP distinct-layer)
      both hits not padded
  Non-candidates are masked to -inf; scores = sigmoid(logits) so masked
  edges score exactly 0.
* Output triplets: strict top-5 per source hit (fixed ONNX shapes), no
  additional confidence threshold.
"""

import json
import math
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from GUNTAM.Transformer.Transformer import TransformerEncoder


class EdgeTransformerConfig:
    """Configuration for SeedEdgeTransformer.

    The hit feature layout is a locked contract with the preprocessing and the
    inference-side seeder: ``hit_feature_names`` gives the column order of the
    hits tensor.
    """

    def __init__(self):
        # Locked hit tensor layout (must match cfg.hit_features at preprocessing)
        self.hit_feature_names = ["x", "y", "z", "r", "phi", "eta", "m", "layer_key"]

        # Transformer encoder
        self.nb_layers_t = 4
        self.nb_heads = 2
        self.dim_embedding = 128
        self.feed_forward_ratio = 2
        self.dropout = 0.1

        # Fourier embedding: k = 0..num_frequencies-1 on [x_rel, y_rel, z, r, phi_rel]
        self.num_frequencies = 15
        # Normalisation scales, locked: x'/500, y'/13, z/1000, r/500, phi_rel/1.5
        self.fourier_scales = [500.0, 13.0, 1000.0, 500.0, 1.5]
        # Compute the Fourier projection in float64 (high k bands need it), cast after
        self.fourier_float64 = True
        # Raw passthrough [phi_rel, r, m, z, eta]; divided by these scales for
        # conditioning only (a linear layer absorbs any fixed rescale)
        self.raw_scales = [1.5, 500.0, 800.0, 1000.0, 4.0]

        # Edge head
        self.edge_hidden_dim = 64
        self.edge_num_hidden = 2  # hidden Linear+ReLU blocks after the factorised layer
        self.edge_min_dm = 5.0  # locked: |m_j - m_i| > 5 mm, forward only
        # Delta features [dm, dz, dr, dphi_rel] scales for conditioning
        self.delta_scales = [100.0, 100.0, 100.0, 0.1]

        # Output
        self.topk = 5  # locked: strict top-5, fixed ONNX shape

    # -- feature index helpers -------------------------------------------------
    def idx(self, name: str) -> int:
        return self.hit_feature_names.index(name)

    @property
    def num_hit_features(self) -> int:
        return len(self.hit_feature_names)

    # -- (de)serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def from_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)

    def save_config(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_config(self, filepath: str) -> None:
        with open(filepath) as f:
            self.from_dict(json.load(f))

    def validate(self) -> None:
        if len(self.fourier_scales) != 5:
            raise ValueError("fourier_scales must have 5 entries [x_rel, y_rel, z, r, phi_rel]")
        if len(self.raw_scales) != 5:
            raise ValueError("raw_scales must have 5 entries [phi_rel, r, m, z, eta]")
        if len(self.delta_scales) != 4:
            raise ValueError("delta_scales must have 4 entries [dm, dz, dr, dphi_rel]")
        if self.num_frequencies < 1 or self.topk < 1:
            raise ValueError("num_frequencies and topk must be >= 1")


class SeedEdgeTransformer(nn.Module):
    """Transformer + dense masked N x N MLP edge head (GUNTAM v2).

    Forward inputs:
        hits:           [B, N, F] float32, columns per cfg.hit_feature_names,
                        rows sorted by m (padding rows are all-zero).
        padding_mask:   [B, N] or [B, N, 1] bool, True = padding.
        bin_phi_center: [B] or [B, 1] float32, phi center of each bin's
                        canonical frame.

    Forward outputs:
        transformer_output: [B, N, d]
        edge_logits:        [B, N, N] (-inf outside the candidate mask)
        candidate_mask:     [B, N, N] bool
        triplets:           [B, N, topk, 3] (source, target, sigmoid score);
                            masked edges score exactly 0.
    """

    def __init__(
        self,
        config: EdgeTransformerConfig | None = None,
        device_acc: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.cfg = config if config is not None else EdgeTransformerConfig()
        self.cfg.validate()
        self.device_acc = device_acc
        self.dtype = dtype
        self._setup_modules()
        self.to(dtype)

    # ------------------------------------------------------------------ setup
    def _setup_modules(self) -> None:
        cfg = self.cfg
        # 5 Fourier-embedded dims x 2 (sin, cos) x bands + 5 raw passthrough
        self.embed_dim_in = 5 * 2 * cfg.num_frequencies + 5

        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(cfg.num_frequencies, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "fourier_scales_t",
            torch.tensor(cfg.fourier_scales, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "raw_scales_t", torch.tensor(cfg.raw_scales, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "delta_scales_t", torch.tensor(cfg.delta_scales, dtype=torch.float32), persistent=False
        )

        self.embedding_projection = nn.Linear(self.embed_dim_in, cfg.dim_embedding, device=self.device_acc)

        self.transformer = TransformerEncoder(
            n_layers=cfg.nb_layers_t,
            input_dim=cfg.dim_embedding,
            model_dim=cfg.feed_forward_ratio * cfg.dim_embedding,
            num_heads=cfg.nb_heads,
            dropout=cfg.dropout,
            device=self.device_acc,
        )

        # Factorised first edge layer: W_src h_i + W_tgt h_j + W_delta deltas + b
        h = cfg.edge_hidden_dim
        self.edge_src = nn.Linear(cfg.dim_embedding, h, bias=True, device=self.device_acc)
        self.edge_tgt = nn.Linear(cfg.dim_embedding, h, bias=False, device=self.device_acc)
        self.edge_delta = nn.Linear(4, h, bias=False, device=self.device_acc)
        hidden = []
        for _ in range(cfg.edge_num_hidden):
            hidden.append(nn.Linear(h, h, device=self.device_acc))
            hidden.append(nn.ReLU())
        self.edge_hidden = nn.Sequential(*hidden)
        self.edge_out = nn.Linear(h, 1, device=self.device_acc)

    # -------------------------------------------------------------- embedding
    def canonical_frame(self, hits: Tensor, bin_phi_center: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Rotate to the bin canonical frame.

        Returns:
            (phi_rel, x_rel, y_rel), each [B, N].
        """
        cfg = self.cfg
        phi = hits[..., cfg.idx("phi")]
        r = hits[..., cfg.idx("r")]
        center = bin_phi_center.reshape(-1, 1).to(phi.dtype)  # [B, 1]
        dphi = phi - center
        # Periodic wrap without branching (ONNX-safe)
        phi_rel = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        x_rel = r * torch.cos(phi_rel)
        y_rel = r * torch.sin(phi_rel)
        return phi_rel, x_rel, y_rel

    def embedding(self, hits: Tensor, bin_phi_center: Tensor) -> Tensor:
        """Locked v2 embedding block -> projected [B, N, dim_embedding]."""
        cfg = self.cfg
        phi_rel, x_rel, y_rel = self.canonical_frame(hits, bin_phi_center)

        z = hits[..., cfg.idx("z")]
        r = hits[..., cfg.idx("r")]
        m = hits[..., cfg.idx("m")]
        eta = hits[..., cfg.idx("eta")]

        coords = torch.stack([x_rel, y_rel, z, r, phi_rel], dim=-1)  # [B, N, 5]
        work_dtype = torch.float64 if cfg.fourier_float64 else coords.dtype
        coords_n = coords.to(work_dtype) / self.fourier_scales_t.to(work_dtype)

        # cycles[b, n, d, k] = 2^k coords_n[b, n, d]; the high-k precision lives in
        # the float64 argument reduction (frac part), while sin/cos themselves run
        # in float32 -- onnxruntime's CPU provider has no float64 Sin/Cos kernel.
        cycles = coords_n.unsqueeze(-1) * self.frequencies.to(work_dtype)
        frac = cycles - torch.floor(cycles)  # exact mod-1 in float64
        angle = (2.0 * math.pi * frac).to(hits.dtype)
        fourier = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)  # [B, N, 5, 2K]
        fourier = fourier.flatten(start_dim=-2)  # [B, N, 5*2K]

        raw = torch.stack([phi_rel, r, m, z, eta], dim=-1) / self.raw_scales_t

        return self.embedding_projection(torch.cat([fourier, raw], dim=-1))

    # ------------------------------------------------------------ edge logits
    def candidate_mask(self, hits: Tensor, padding_mask: Tensor) -> Tensor:
        """In-graph candidate-edge rules -> bool [B, N, N] (True = candidate).

        kept(i -> j)  iff  m_j - m_i > edge_min_dm  (forward + min length)
                      and  layer_key_i != layer_key_j
                      and  neither i nor j is padding.
        """
        cfg = self.cfg
        m = hits[..., cfg.idx("m")]
        layer = hits[..., cfg.idx("layer_key")]

        dm = m.unsqueeze(1) - m.unsqueeze(2)  # dm[b, i, j] = m_j - m_i
        forward_ok = dm > cfg.edge_min_dm
        layer_ok = layer.unsqueeze(1) != layer.unsqueeze(2)

        valid = ~padding_mask  # [B, N] True = real hit
        valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)

        return forward_ok & layer_ok & valid_pair

    def edge_logits(self, encoded: Tensor, hits: Tensor, cand: Tensor) -> Tensor:
        """Dense masked N x N MLP edge scores.

        The first layer is factorised so the largest intermediate is
        [B, N, N, edge_hidden_dim] rather than [B, N, N, 2 dim_embedding].
        """
        cfg = self.cfg
        m = hits[..., cfg.idx("m")]
        z = hits[..., cfg.idx("z")]
        r = hits[..., cfg.idx("r")]
        phi = hits[..., cfg.idx("phi")]

        def pair_delta(v: Tensor) -> Tensor:
            return v.unsqueeze(1) - v.unsqueeze(2)  # [B, N, N] = v_j - v_i

        dphi = pair_delta(phi)
        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        deltas = torch.stack([pair_delta(m), pair_delta(z), pair_delta(r), dphi], dim=-1)
        deltas = deltas / self.delta_scales_t

        # Factorised first layer + ReLU
        src = self.edge_src(encoded).unsqueeze(2)  # [B, N, 1, H] (i = source rows)
        tgt = self.edge_tgt(encoded).unsqueeze(1)  # [B, 1, N, H] (j = target cols)
        x = torch.relu(src + tgt + self.edge_delta(deltas))  # [B, N, N, H]
        x = self.edge_hidden(x)
        logits = self.edge_out(x).squeeze(-1)  # [B, N, N]

        return logits.masked_fill(~cand, float("-inf"))

    # ---------------------------------------------------------------- forward
    def forward(
        self, hits: Tensor, padding_mask: Tensor, bin_phi_center: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if padding_mask.dim() == 3:
            padding_mask = padding_mask.squeeze(-1)
        padding_mask = padding_mask.bool()

        encoded = self.embedding(hits, bin_phi_center)
        transformer_output = self.transformer(x=encoded, mask=padding_mask)

        cand = self.candidate_mask(hits, padding_mask)
        logits = self.edge_logits(transformer_output, hits, cand)

        # sigmoid maps the -inf mask to an exact 0 score
        scores = torch.sigmoid(logits)
        topk_scores, topk_targets = scores.topk(self.cfg.topk, dim=-1)  # [B, N, K]
        B, N, k = topk_scores.shape
        source = torch.arange(N, device=hits.device).view(1, N, 1).expand(B, N, k)
        triplets = torch.stack([source.to(topk_scores.dtype), topk_targets.to(topk_scores.dtype), topk_scores], dim=-1)

        return transformer_output, logits, cand, triplets

    # ---------------------------------------------------------------- persist
    def save(self, epoch: int, path: str, optimizer=None, scheduler=None) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "edge_transformer_config": self.cfg.to_dict(),
                "dtype": str(self.dtype).replace("torch.", ""),
            },
            path,
        )

    def load(self, path: str, device: torch.device) -> int:
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        cfg_dict = checkpoint.get("edge_transformer_config")
        if cfg_dict:
            self.cfg.from_dict(cfg_dict)
            self.device_acc = device
            self._setup_modules()
        self.load_state_dict(checkpoint["model_state_dict"])
        self.to(device)
        return checkpoint.get("epoch", -1) + 1

    def print_model_info(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        print("SeedEdgeTransformer Model Info:")
        print(f"  - Transformer layers: {self.cfg.nb_layers_t}")
        print(f"  - Embedding input dim: {self.embed_dim_in}")
        print(f"  - Edge hidden dim: {self.cfg.edge_hidden_dim}")
        print(f"  - Total parameters: {total}")

    # ------------------------------------------------------------------- onnx
    def export_onnx(
        self,
        path: str,
        example_hits: Tensor | None = None,
        example_mask: Tensor | None = None,
        example_center: Tensor | None = None,
    ) -> None:
        """Export (hits, padding_mask, bin_phi_center) -> (output, triplets).

        Fixed top-5 triplet shape keeps the ONNX interface static; the beam
        search consumes triplets exactly as with the previous model.
        """
        if example_hits is None:
            seq_len = 32
            example_hits = torch.zeros(1, seq_len, self.cfg.num_hit_features, dtype=torch.float32)
            # Make m strictly increasing so the candidate mask is non-degenerate
            example_hits[..., self.cfg.idx("m")] = torch.arange(seq_len, dtype=torch.float32) * 10.0
            example_mask = torch.zeros(1, seq_len, 1, dtype=torch.bool)
            example_center = torch.zeros(1, dtype=torch.float32)
        assert example_mask is not None and example_center is not None

        wrapper = _ExportWrapper(self).eval().cpu()
        was_training = self.training
        original_device = next(self.parameters()).device
        self.eval()
        self.to("cpu")
        try:
            torch.onnx.export(
                wrapper,
                (example_hits.float().cpu(), example_mask.cpu(), example_center.float().cpu()),
                path,
                input_names=["hits", "padding_mask", "bin_phi_center"],
                output_names=["output", "triplets"],
                dynamic_axes={
                    "hits": {0: "batch_size", 1: "seq_len"},
                    "padding_mask": {0: "batch_size", 1: "seq_len"},
                    "bin_phi_center": {0: "batch_size"},
                    "output": {0: "batch_size", 1: "seq_len"},
                    "triplets": {0: "batch_size", 1: "seq_len"},
                },
                opset_version=17,
                dynamo=False,  # classic tracer: stable with masked_fill(-inf) + topk
            )
            print(f"Model exported to ONNX at {path}")
        finally:
            self.to(original_device)
            if was_training:
                self.train()


class _ExportWrapper(nn.Module):
    """ONNX wrapper: keep only the interface outputs (output, triplets)."""

    def __init__(self, model: SeedEdgeTransformer):
        super().__init__()
        self.model = model

    def forward(self, hits: Tensor, padding_mask: Tensor, bin_phi_center: Tensor):
        transformer_output, _logits, _cand, triplets = self.model(hits, padding_mask, bin_phi_center)
        return transformer_output, triplets
