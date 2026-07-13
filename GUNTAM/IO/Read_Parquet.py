"""Parquet input for the GUNTAM training pipeline.

Reads the chunked Parquet files produced by the ACTS-side dump job
(``guntam_v2_parquet_pipeline/dump_spacepoints_parquet.py``): pairs of

    spacepoints_chunkNNNNN.parquet   one row per spacepoint
    particles_chunkNNNNN.parquet     one row per truth particle with >=1 spacepoint

Spacepoint columns (per row): event_id, x, y, z, r, phi, eta, m, varR, varZ,
volume, layer, layer_key, particle_id, particle_id_pv, badSP.
Particle columns: event_id, particle_id, particle_id_pv, pT, eta, phi, d0, z0.

``m = (R + rho) / 2`` with ``R = sqrt(x^2+y^2)`` and ``rho = sqrt(x^2+y^2+z^2)``
is the sort/edge metric of the v2 design. ``layer_key`` packs (volume, layer)
into a single integer so a same-layer test is one equality comparison.

particle_id values inside a chunk are already dense per event (0..P-1 assigned
by the dump job); -1 marks orphan spacepoints. This module additionally
re-orphanizes particles with fewer than ``cfg.min_sp_per_particle`` spacepoints
(v2 locked threshold: 4).
"""

import glob
from typing import List, Tuple

import numpy as np
import pandas as pd


def find_parquet_chunks(input_path: str) -> List[Tuple[str, str]]:
    """Return sorted (spacepoints, particles) parquet chunk file pairs.

    Args:
        input_path: Directory containing spacepoints_chunk*.parquet and
            particles_chunk*.parquet files.

    Returns:
        List of (spacepoint_file, particle_file) tuples, sorted by chunk id.

    Raises:
        FileNotFoundError: If no chunk pairs are found or files are unpaired.
    """
    sp_files = sorted(glob.glob(f"{input_path}/spacepoints_chunk*.parquet"))
    part_files = sorted(glob.glob(f"{input_path}/particles_chunk*.parquet"))

    if not sp_files:
        raise FileNotFoundError(f"No spacepoints_chunk*.parquet files found in {input_path}")

    def chunk_id(path: str, prefix: str) -> str:
        name = path.rsplit("/", 1)[-1]
        return name.replace(prefix, "").replace(".parquet", "")

    sp_ids = [chunk_id(f, "spacepoints_chunk") for f in sp_files]
    part_ids = [chunk_id(f, "particles_chunk") for f in part_files]
    if sp_ids != part_ids:
        raise FileNotFoundError(
            f"Unpaired parquet chunks in {input_path}: "
            f"spacepoints={sp_ids} vs particles={part_ids}"
        )
    return list(zip(sp_files, part_files))


def _ensure_derived_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add r, eta, phi, m columns if missing (defensive; the dump job writes them)."""
    x_sq = data["x"] ** 2
    y_sq = data["y"] ** 2
    z_sq = data["z"] ** 2
    if "r" not in data.columns:
        data["r"] = np.sqrt(x_sq + y_sq)
    if "phi" not in data.columns:
        data["phi"] = np.arctan2(data["y"], data["x"])
    if "eta" not in data.columns:
        theta = np.arctan2(data["r"], data["z"])
        data["eta"] = -np.log(np.tan(theta / 2))
    if "m" not in data.columns:
        rho = np.sqrt(x_sq + y_sq + z_sq)
        data["m"] = 0.5 * (data["r"] + rho)
    return data


def apply_min_sp_filter(
    data: pd.DataFrame, particles: pd.DataFrame, min_sp_per_particle: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orphanize particles with fewer than min_sp_per_particle spacepoints.

    Their spacepoints get particle_id = -1 and the particles are dropped from
    the particles table, mirroring what Read_ACTS_Csv did with
    min_hits_per_particle (v2 locked value: 4 spacepoints).

    Args:
        data: Spacepoint DataFrame with event_id / particle_id columns.
        particles: Particle DataFrame with event_id / particle_id columns.
        min_sp_per_particle: Minimum spacepoint count for a particle to survive.

    Returns:
        Tuple of (data, particles) after filtering.
    """
    if min_sp_per_particle <= 1:
        return data, particles

    # Vectorized per-(event, particle) spacepoint counts aligned with data rows
    sp_counts = data.groupby(["event_id", "particle_id"])["particle_id"].transform("size")
    orphanize = (sp_counts < min_sp_per_particle) & (data["particle_id"] >= 0)
    if orphanize.any():
        data = data.copy()
        data.loc[orphanize, "particle_id"] = -1

    counts = (
        data[data["particle_id"] >= 0]
        .groupby(["event_id", "particle_id"])
        .size()
        .rename("n_sp")
        .reset_index()
    )
    particles = particles.merge(counts, on=["event_id", "particle_id"], how="left")
    particles = particles[particles["n_sp"].fillna(0) >= min_sp_per_particle]
    particles = particles.drop(columns=["n_sp"]).reset_index(drop=True)

    return data, particles


def load_parquet_chunk(
    sp_file: str, particle_file: str, min_sp_per_particle: int = 4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load one (spacepoints, particles) parquet chunk pair for training.

    Applies the min-spacepoint particle filter and guarantees the derived
    columns (r, phi, eta, m) exist. Event ids are the true ACTS event numbers
    (not necessarily contiguous); the caller is responsible for densifying them
    if needed.

    Args:
        sp_file: Path to a spacepoints_chunk*.parquet file.
        particle_file: Path to the paired particles_chunk*.parquet file.
        min_sp_per_particle: Minimum spacepoints per particle (default 4).

    Returns:
        Tuple of (spacepoints DataFrame, particles DataFrame).
    """
    data = pd.read_parquet(sp_file)
    particles = pd.read_parquet(particle_file)

    data = _ensure_derived_columns(data)
    data, particles = apply_min_sp_filter(data, particles, min_sp_per_particle)

    return data, particles
