#!/usr/bin/env python3
"""Evaluate & write a density-stratified coastal spatial train/val/test split.

Builds per-tile boulder metadata from july14 GPKGs + tile footprints, groups
tiles into contiguous coastal blocks (~50–100 m along the PCA coast axis),
tiers blocks by boulder density, then assigns whole blocks to train / valid /
test (~70–75% / ~12.5–15% / ~12.5–15%) with an abutting buffer so adjacent
tiles cannot straddle splits.

Boulder counts default to Class=0 polygons with clipped area ≥ ``--min-area-m2``
(1.5). Deposits are never counted as boulders.

Example:
  python BoulderCalculator/scripts/evaluate_stratified_coastal_split.py \\
    --segmentation-dir segmentation \\
    --output-dir BoulderCalculator/experiments/geo_splits \\
    --tile-subdir tiling_used \\
    --min-area-m2 1.5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import box
from shapely.strtree import STRtree

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_coastal_splits import (  # noqa: E402
    TOUCH_TOL_M,
    abutting_buffer,
    expand_bounds,
    pca_axis_scores,
    split_dict,
    write_split_geojson,
    write_yaml,
)
from gpkg_to_coco import (  # noqa: E402
    canonical_key,
    footprint_overlaps,
    load_annotations,
    resolve_tile_path,
    resolve_tiles_by_year,
    year_key,
)

# Default class field matches gpkg_to_coco.
CLASS_FIELD = "Class"


@dataclass
class TileMeta:
    year_key: str
    year: int
    short: str
    bounds: tuple[float, float, float, float]
    pca_score: float = 0.0
    n_boulders: int = 0  # Class=0 meeting min_area_m2 (clipped to tile)
    n_boulders_all: int = 0  # Class=0 any size
    n_deposits: int = 0
    is_negative: bool = True


@dataclass
class Block:
    block_id: int
    year_keys: list[str] = field(default_factory=list)
    pca_lo: float = 0.0
    pca_hi: float = 0.0
    n_tiles: int = 0
    n_boulders: int = 0
    n_deposits: int = 0
    n_negative: int = 0
    density: float = 0.0  # boulders / tile
    tier: str = "negative"  # high | medium_low | negative


def load_bounds(
    tile_dir: Path, years: list[int], tiles_by_year: dict[int, list[str]]
) -> dict[str, tuple[float, float, float, float]]:
    cache: dict[str, tuple[float, float, float, float]] = {}
    for year in years:
        for key in tiles_by_year[year]:
            yk = year_key(year, key)
            with rasterio.open(resolve_tile_path(tile_dir, yk)) as ds:
                b = ds.bounds
                cache[yk] = (b.left, b.bottom, b.right, b.top)
    return cache


def count_instances_per_tile(
    tiles: dict[str, TileMeta],
    annotations: list[tuple],
    min_area_m2: float = 1.5,
) -> None:
    """Mutate tiles with boulder/deposit counts via STRtree intersection.

    Boulder counts use clipped intersection area vs ``min_area_m2`` (default
    1.5). Deposits (class 1) are never counted as boulders. Negatives = tiles
    with zero boulders meeting the area cutoff.
    """
    keys = sorted(tiles)
    geoms = [box(*tiles[k].bounds) for k in keys]
    tree = STRtree(geoms)
    # Shapely 2: query returns indices; Shapely 1 returns geoms — handle both.
    for geom, cls, year in annotations:
        if geom.is_empty:
            continue
        hits = tree.query(geom)
        for hit in hits:
            if isinstance(hit, (int, np.integer)):
                idx = int(hit)
                tile_geom = geoms[idx]
            else:
                try:
                    idx = geoms.index(hit)
                    tile_geom = hit
                except ValueError:
                    continue
            yk = keys[idx]
            meta = tiles[yk]
            if year is not None and meta.year != year:
                continue
            if not tile_geom.intersects(geom):
                continue
            # Require meaningful overlap (avoid pure boundary touches).
            inter = tile_geom.intersection(geom)
            if inter.is_empty or inter.area < 1e-6:
                continue
            if cls == 1:
                meta.n_deposits += 1
            else:
                meta.n_boulders_all += 1
                if inter.area >= min_area_m2:
                    meta.n_boulders += 1
    for meta in tiles.values():
        meta.is_negative = meta.n_boulders == 0


def form_coastal_blocks(
    tiles: dict[str, TileMeta],
    target_span_m: float = 75.0,
    min_tiles: int = 1,
    max_tiles: int = 4,
) -> list[Block]:
    """Group tiles ordered by PCA score into contiguous ~target_span_m blocks.

    PCA scores are along-coast metres (projection of centroids). A new block
    starts when adding the next tile would exceed ``target_span_m`` beyond the
    first tile's score, or when ``max_tiles`` is reached.
    """
    ordered = sorted(tiles.values(), key=lambda t: t.pca_score)
    blocks: list[Block] = []
    cur: list[TileMeta] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        bid = len(blocks)
        b = Block(
            block_id=bid,
            year_keys=[t.year_key for t in cur],
            pca_lo=cur[0].pca_score,
            pca_hi=cur[-1].pca_score,
            n_tiles=len(cur),
            n_boulders=sum(t.n_boulders for t in cur),
            n_deposits=sum(t.n_deposits for t in cur),
            n_negative=sum(1 for t in cur if t.is_negative),
        )
        b.density = b.n_boulders / max(b.n_tiles, 1)
        blocks.append(b)
        cur = []

    for t in ordered:
        if not cur:
            cur = [t]
            continue
        span = t.pca_score - cur[0].pca_score
        if len(cur) >= max_tiles or (len(cur) >= min_tiles and span > target_span_m):
            flush()
            cur = [t]
        else:
            cur.append(t)
    flush()
    return blocks


def assign_density_tiers(blocks: list[Block]) -> None:
    """Label blocks: negative (0 boulders), else high vs medium_low by median density."""
    pos = [b for b in blocks if b.n_boulders > 0]
    if not pos:
        for b in blocks:
            b.tier = "negative"
        return
    dens = sorted(b.density for b in pos)
    med = dens[len(dens) // 2]
    for b in blocks:
        if b.n_boulders == 0:
            b.tier = "negative"
        elif b.density >= med:
            b.tier = "high"
        else:
            b.tier = "medium_low"


def _split_counts(
    blocks: list[Block],
    assignment: dict[int, str],
) -> dict[str, dict[str, float]]:
    out = {
        s: {"tiles": 0, "boulders": 0, "negative": 0, "blocks": 0}
        for s in ("train", "valid", "test", "excluded")
    }
    for b in blocks:
        s = assignment[b.block_id]
        out[s]["tiles"] += b.n_tiles
        out[s]["boulders"] += b.n_boulders
        out[s]["negative"] += b.n_negative
        out[s]["blocks"] += 1
    return out


def score_assignment(
    blocks: list[Block],
    assignment: dict[int, str],
    *,
    train_lo: float = 0.70,
    train_hi: float = 0.75,
    hold_lo: float = 0.125,
    hold_hi: float = 0.15,
    neg_train_lo: float = 0.20,
    neg_train_hi: float = 0.30,
) -> tuple[float, dict]:
    """Lower is better. Soft penalties outside target bands."""
    counts = _split_counts(blocks, assignment)
    total_tiles = sum(c["tiles"] for c in counts.values() if True)
    total_boulders = sum(c["boulders"] for c in counts.values())
    # Ratios among train+valid+test only (excluded is intentional leakage buffer).
    usable = counts["train"]["tiles"] + counts["valid"]["tiles"] + counts["test"]["tiles"]
    if usable == 0 or counts["valid"]["tiles"] == 0 or counts["test"]["tiles"] == 0:
        return 1e9, counts
    if counts["train"]["tiles"] == 0:
        return 1e9, counts

    def frac(n: float, d: float) -> float:
        return n / d if d else 0.0

    tr = frac(counts["train"]["tiles"], usable)
    va = frac(counts["valid"]["tiles"], usable)
    te = frac(counts["test"]["tiles"], usable)
    br = frac(counts["train"]["boulders"], max(total_boulders, 1))
    bv = frac(counts["valid"]["boulders"], max(total_boulders, 1))
    bt = frac(counts["test"]["boulders"], max(total_boulders, 1))
    neg_tr = frac(counts["train"]["negative"], counts["train"]["tiles"])

    def band_pen(x: float, lo: float, hi: float, w: float = 1.0) -> float:
        if lo <= x <= hi:
            return 0.0
        if x < lo:
            return w * (lo - x) ** 2
        return w * (x - hi) ** 2

    pen = 0.0
    pen += band_pen(tr, train_lo, train_hi, 40)
    pen += band_pen(va, hold_lo, hold_hi, 40)
    pen += band_pen(te, hold_lo, hold_hi, 40)
    # Boulder instance balance: same bands as tiles.
    pen += band_pen(br, train_lo, train_hi, 50)
    pen += band_pen(bv, hold_lo, hold_hi, 50)
    pen += band_pen(bt, hold_lo, hold_hi, 50)
    pen += band_pen(neg_tr, neg_train_lo, neg_train_hi, 30)
    # Prefer few excluded tiles.
    pen += 2.0 * frac(counts["excluded"]["tiles"], max(total_tiles, 1))
    # Soft: holdouts should not be empty of boulders.
    if counts["valid"]["boulders"] == 0:
        pen += 5.0
    if counts["test"]["boulders"] == 0:
        pen += 5.0
    # Prefer spaced holdouts: penalize adjacent block_ids in different holdout sets.
    by_id = {b.block_id: b for b in blocks}
    for i in range(len(blocks) - 1):
        a, b = assignment[i], assignment[i + 1]
        if {a, b} <= {"valid", "test"} and a != b:
            pen += 0.5  # valid|test adjacency is ok but slightly discouraged
        if a in ("valid", "test") and b == "train" and i + 1 in by_id:
            pass  # buffer handles adjacency
    return pen, counts


def apply_tile_buffer(
    blocks: list[Block],
    block_assign: dict[int, str],
    bounds: dict[str, tuple[float, float, float, float]],
) -> dict[str, str]:
    """Map year_key → split; mark abutting train tiles as excluded."""
    tile_assign: dict[str, str] = {}
    for b in blocks:
        s = block_assign[b.block_id]
        for yk in b.year_keys:
            tile_assign[yk] = s
    holdout = {k for k, s in tile_assign.items() if s in ("valid", "test")}
    eligible = set(tile_assign)
    buf = abutting_buffer(bounds, holdout, eligible)
    for yk in buf:
        if tile_assign.get(yk) == "train":
            tile_assign[yk] = "excluded"
    return tile_assign


def candidate_holdout_sets(
    blocks: list[Block],
    *,
    n_valid_blocks: int,
    n_test_blocks: int,
    rng: random.Random,
    n_samples: int = 400,
) -> list[tuple[frozenset[int], frozenset[int]]]:
    """Sample geographically contiguous holdout windows along the coast.

    Contiguous windows minimize abutting-buffer loss versus scattered blocks.
    Density balance is handled by the scorer over many window placements.
    """
    ids = [b.block_id for b in blocks]  # already PCA-ordered by construction
    n = len(ids)
    results: list[tuple[frozenset[int], frozenset[int]]] = []
    seen: set[tuple[frozenset[int], frozenset[int]]] = set()

    def add(valid: frozenset[int], test: frozenset[int]) -> None:
        if not valid or not test or (valid & test):
            return
        key = (valid, test)
        if key in seen:
            return
        seen.add(key)
        results.append(key)

    # Exhaustive contiguous window placements (n is small: ~66).
    for v0 in range(0, n - n_valid_blocks + 1):
        valid = frozenset(ids[v0 : v0 + n_valid_blocks])
        for t0 in range(0, n - n_test_blocks + 1):
            if not (t0 + n_test_blocks <= v0 or t0 >= v0 + n_valid_blocks):
                continue  # overlapping windows
            # Require a ≥1-block gap so buffer isn't shared between holdouts.
            if abs(t0 - v0) < n_valid_blocks + 1 and abs(t0 - v0) < n_test_blocks + 1:
                gap_ok = t0 + n_test_blocks <= v0 - 1 or t0 >= v0 + n_valid_blocks + 1
                if not gap_ok:
                    continue
            test = frozenset(ids[t0 : t0 + n_test_blocks])
            add(valid, test)

    # Also sample two-window splits with a train gap in the middle (classic
    # coastal holdout geometry: valid near one end, test near other / mid).
    for _ in range(n_samples):
        # Pick two non-overlapping starts with gap
        if n < n_valid_blocks + n_test_blocks + 2:
            break
        v0 = rng.randint(0, n - n_valid_blocks)
        candidates = [
            t
            for t in range(0, n - n_test_blocks + 1)
            if t + n_test_blocks <= v0 - 1 or t >= v0 + n_valid_blocks + 1
        ]
        if not candidates:
            continue
        t0 = rng.choice(candidates)
        add(
            frozenset(ids[v0 : v0 + n_valid_blocks]),
            frozenset(ids[t0 : t0 + n_test_blocks]),
        )

    # Optional: swap one edge block with a nearby negative/medium block to
    # tweak density without breaking contiguity much — sample a few swaps.
    by_tier = defaultdict(list)
    for b in blocks:
        by_tier[b.tier].append(b.block_id)
    base = results[: min(200, len(results))]
    for valid, test in base:
        for hold_name, hold in (("valid", valid), ("test", test)):
            hold_list = sorted(hold)
            if not hold_list:
                continue
            edge = hold_list[0] if rng.random() < 0.5 else hold_list[-1]
            donor_pool = [
                i
                for i in by_tier["negative"] + by_tier["medium_low"]
                if i not in valid and i not in test
            ]
            if not donor_pool:
                continue
            donor = rng.choice(donor_pool)
            new_hold = (hold - {edge}) | {donor}
            if hold_name == "valid":
                add(frozenset(new_hold), test)
            else:
                add(valid, frozenset(new_hold))
    return results


def _block_level_counts(
    blocks: list[Block], block_assign: dict[int, str]
) -> dict[str, dict[str, float]]:
    """Fast pre-buffer scoring using whole-block assignment."""
    out = {
        s: {"tiles": 0, "boulders": 0, "negative": 0, "blocks": 0}
        for s in ("train", "valid", "test", "excluded")
    }
    for b in blocks:
        s = block_assign[b.block_id]
        out[s]["tiles"] += b.n_tiles
        out[s]["boulders"] += b.n_boulders
        out[s]["negative"] += b.n_negative
        out[s]["blocks"] += 1
    return out


def optimize_split(
    blocks: list[Block],
    bounds: dict[str, tuple[float, float, float, float]],
    seed: int = 42,
) -> tuple[dict[int, str], dict[str, str], float, dict]:
    n_blocks = len(blocks)
    rng = random.Random(seed)
    best_pen = 1e18
    best_block: dict[int, str] | None = None
    best_tile: dict[str, str] | None = None
    best_counts: dict | None = None
    n_tried = 0

    # Stage 1: cheap block-level search; keep top-K for buffer refinement.
    prelim: list[tuple[float, dict[int, str]]] = []
    budgets = sorted(
        {
            max(2, int(round(n_blocks * f)))
            for f in (0.12, 0.13, 0.14, 0.15)
        }
    )

    for n_valid in budgets:
        for n_test in budgets:
            if n_valid + n_test >= n_blocks - 2:
                continue
            for valid_ids, test_ids in candidate_holdout_sets(
                blocks,
                n_valid_blocks=n_valid,
                n_test_blocks=n_test,
                rng=rng,
                n_samples=120,
            ):
                if valid_ids & test_ids:
                    continue
                block_assign = {
                    b.block_id: (
                        "valid"
                        if b.block_id in valid_ids
                        else "test"
                        if b.block_id in test_ids
                        else "train"
                    )
                    for b in blocks
                }
                counts = _block_level_counts(blocks, block_assign)
                pen = _pen_from_tile_counts(counts)
                n_tried += 1
                prelim.append((pen, block_assign))

    prelim.sort(key=lambda t: t[0])
    # Deduplicate by frozenset of (valid,test) block ids
    uniq: list[tuple[float, dict[int, str]]] = []
    seen_keys: set[tuple[frozenset[int], frozenset[int]]] = set()
    for pen, ba in prelim:
        key = (
            frozenset(i for i, s in ba.items() if s == "valid"),
            frozenset(i for i, s in ba.items() if s == "test"),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        uniq.append((pen, ba))
        if len(uniq) >= 80:
            break

    print(f"Stage 1: {n_tried} samples → refining top {len(uniq)} with adjacency buffer…")

    for _pen0, block_assign in uniq:
        tile_assign = apply_tile_buffer(blocks, block_assign, bounds)
        tile_blocks = _counts_from_tiles(blocks, tile_assign)
        pen = _pen_from_tile_counts(tile_blocks)
        if pen < best_pen:
            best_pen = pen
            best_block = block_assign
            best_tile = tile_assign
            best_counts = tile_blocks

    if best_block is None or best_tile is None or best_counts is None:
        raise RuntimeError("No valid split candidates found")
    print(f"Best buffered penalty={best_pen:.4f}")
    return best_block, best_tile, best_pen, best_counts


def _counts_from_tiles(
    blocks: list[Block], tile_assign: dict[str, str]
) -> dict[str, dict[str, float]]:
    tile_to_stats = {}
    for b in blocks:
        for yk in b.year_keys:
            # recover per-tile from block aggregates is wrong; stash on side
            pass
    # Rebuild from block year_keys using equal share only if needed — better:
    # caller should pass tiles. Use block sums with per-tile assignment.
    out = {
        s: {"tiles": 0, "boulders": 0, "negative": 0, "blocks": 0}
        for s in ("train", "valid", "test", "excluded")
    }
    # We need per-tile boulder counts — encode in year_keys via a closure.
    # Stored on blocks only as aggregates; recompute via weighted: look up from
    # a module-level map set by main.
    global _TILE_LOOKUP  # noqa: PLW0603
    for b in blocks:
        splits_in_block = {tile_assign[yk] for yk in b.year_keys}
        if len(splits_in_block) == 1:
            out[next(iter(splits_in_block))]["blocks"] += 1
        for yk in b.year_keys:
            s = tile_assign[yk]
            t = _TILE_LOOKUP[yk]
            out[s]["tiles"] += 1
            out[s]["boulders"] += t.n_boulders
            out[s]["negative"] += int(t.is_negative)
    return out


_TILE_LOOKUP: dict[str, TileMeta] = {}


def _pen_from_tile_counts(counts: dict) -> float:
    usable = counts["train"]["tiles"] + counts["valid"]["tiles"] + counts["test"]["tiles"]
    # Instance % among usable only (excluded buffer boulders shouldn't inflate holdouts).
    usable_b = (
        counts["train"]["boulders"]
        + counts["valid"]["boulders"]
        + counts["test"]["boulders"]
    )
    total_tiles = sum(c["tiles"] for c in counts.values())
    if usable == 0 or counts["valid"]["tiles"] == 0 or counts["test"]["tiles"] == 0:
        return 1e9
    if counts["train"]["tiles"] == 0:
        return 1e9

    def frac(n, d):
        return n / d if d else 0.0

    def band_pen(x, lo, hi, w=1.0):
        if lo <= x <= hi:
            return 0.0
        return w * ((lo - x) ** 2 if x < lo else (x - hi) ** 2)

    tr = frac(counts["train"]["tiles"], usable)
    va = frac(counts["valid"]["tiles"], usable)
    te = frac(counts["test"]["tiles"], usable)
    br = frac(counts["train"]["boulders"], max(usable_b, 1))
    bv = frac(counts["valid"]["boulders"], max(usable_b, 1))
    bt = frac(counts["test"]["boulders"], max(usable_b, 1))
    neg_tr = frac(counts["train"]["negative"], counts["train"]["tiles"])
    pen = 0.0
    pen += band_pen(tr, 0.70, 0.75, 50)
    pen += band_pen(va, 0.125, 0.15, 80)
    pen += band_pen(te, 0.125, 0.15, 80)
    # Boulder balance is the primary objective.
    pen += band_pen(br, 0.70, 0.75, 140)
    pen += band_pen(bv, 0.125, 0.15, 120)
    pen += band_pen(bt, 0.125, 0.15, 120)
    # Dataset is ~45% negative at ≥1.5 m²; aim to pull train below that.
    pen += band_pen(neg_tr, 0.25, 0.40, 70)
    # Soften excluded penalty — contiguous windows already keep it small.
    pen += 0.5 * frac(counts["excluded"]["tiles"], max(total_tiles, 1))
    # Hard preference: keep usable tile fraction high (≥85% of all tiles).
    if total_tiles:
        usable_frac = usable / total_tiles
        if usable_frac < 0.85:
            pen += 40.0 * (0.85 - usable_frac) ** 2
    if counts["valid"]["boulders"] == 0:
        pen += 8.0
    if counts["test"]["boulders"] == 0:
        pen += 8.0
    # Soft symmetry between val and test boulder counts
    if usable_b:
        pen += 15.0 * abs(bv - bt) ** 2
    return pen


def verify_no_overlap(tile_assign: dict[str, str]) -> list[str]:
    flags = []
    sets = {s: {k for k, v in tile_assign.items() if v == s} for s in ("train", "valid", "test", "excluded")}
    for a, b in itertools.combinations(["train", "valid", "test", "excluded"], 2):
        inter = sets[a] & sets[b]
        if inter:
            flags.append(f"OVERLAP {a}&{b}: {len(inter)} tiles e.g. {sorted(inter)[:3]}")
    return flags


def verify_no_adjacent_leak(
    tile_assign: dict[str, str],
    bounds: dict[str, tuple[float, float, float, float]],
) -> list[str]:
    """Flag train↔holdout abutting pairs (should be 0 after buffer)."""
    flags = []
    hold = {k for k, s in tile_assign.items() if s in ("valid", "test")}
    train = {k for k, s in tile_assign.items() if s == "train"}
    leaks = []
    for t in train:
        be = expand_bounds(bounds[t], TOUCH_TOL_M)
        for h in hold:
            if footprint_overlaps(be, bounds[h], min_area=0.01):
                leaks.append((t, h, tile_assign[h]))
                break
    if leaks:
        flags.append(
            f"ADJACENCY LEAK train↔holdout: {len(leaks)} train tiles "
            f"e.g. {leaks[:3]}"
        )
    return flags


def write_metadata_csv(path: Path, tiles: dict[str, TileMeta], tile_assign: dict[str, str], block_of: dict[str, int], tiers: dict[int, str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "year_key",
                "year",
                "tile",
                "pca_score",
                "n_boulders",
                "n_boulders_all",
                "n_deposits",
                "is_negative",
                "block_id",
                "block_tier",
                "split",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ],
        )
        w.writeheader()
        for yk in sorted(tiles, key=lambda k: tiles[k].pca_score):
            t = tiles[yk]
            bid = block_of[yk]
            w.writerow(
                {
                    "year_key": yk,
                    "year": t.year,
                    "tile": t.short,
                    "pca_score": f"{t.pca_score:.3f}",
                    "n_boulders": t.n_boulders,
                    "n_boulders_all": t.n_boulders_all,
                    "n_deposits": t.n_deposits,
                    "is_negative": int(t.is_negative),
                    "block_id": bid,
                    "block_tier": tiers[bid],
                    "split": tile_assign[yk],
                    "xmin": t.bounds[0],
                    "ymin": t.bounds[1],
                    "xmax": t.bounds[2],
                    "ymax": t.bounds[3],
                }
            )


def write_block_csv(path: Path, blocks: list[Block], block_assign: dict[int, str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "block_id",
                "tier",
                "split",
                "n_tiles",
                "n_boulders",
                "n_deposits",
                "n_negative",
                "density",
                "pca_lo",
                "pca_hi",
                "span_m",
                "tiles",
            ],
        )
        w.writeheader()
        for b in blocks:
            w.writerow(
                {
                    "block_id": b.block_id,
                    "tier": b.tier,
                    "split": block_assign[b.block_id],
                    "n_tiles": b.n_tiles,
                    "n_boulders": b.n_boulders,
                    "n_deposits": b.n_deposits,
                    "n_negative": b.n_negative,
                    "density": f"{b.density:.2f}",
                    "pca_lo": f"{b.pca_lo:.3f}",
                    "pca_hi": f"{b.pca_hi:.3f}",
                    "span_m": f"{b.pca_hi - b.pca_lo:.3f}",
                    "tiles": ";".join(b.year_keys),
                }
            )


def print_report(
    blocks: list[Block],
    tiles: dict[str, TileMeta],
    tile_assign: dict[str, str],
    counts: dict,
    flags: list[str],
    outliers: list[str],
) -> str:
    usable = counts["train"]["tiles"] + counts["valid"]["tiles"] + counts["test"]["tiles"]
    lines = []
    lines.append("=" * 72)
    lines.append("STRATIFIED COASTAL SPATIAL SPLIT — SUMMARY REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Tiles total: {len(tiles)}  |  Blocks: {len(blocks)}  |  "
                 f"Usable (excl. buffer): {usable}")
    lines.append("")
    lines.append("Density tiers (blocks):")
    for tier in ("high", "medium_low", "negative"):
        bs = [b for b in blocks if b.tier == tier]
        lines.append(
            f"  {tier:12s}  blocks={len(bs):3d}  tiles={sum(b.n_tiles for b in bs):3d}  "
            f"boulders={sum(b.n_boulders for b in bs):5d}"
        )
    lines.append("")
    usable_b = (
        counts["train"]["boulders"]
        + counts["valid"]["boulders"]
        + counts["test"]["boulders"]
    )
    lines.append(f"{'Split':10s} {'Tiles':>7s} {'Tile%':>7s} {'Boulders':>9s} {'Bould%':>7s} "
                 f"{'NegTiles':>8s} {'Neg%':>7s} {'Blocks':>7s}")
    for s in ("train", "valid", "test", "excluded"):
        c = counts[s]
        if s != "excluded":
            tp = 100.0 * c["tiles"] / usable if usable else 0.0
            bp = 100.0 * c["boulders"] / usable_b if usable_b else 0.0
        else:
            tp = 100.0 * c["tiles"] / len(tiles) if tiles else 0.0
            bp = 100.0 * c["boulders"] / max(sum(x["boulders"] for x in counts.values()), 1)
        neg_pct = 100.0 * c["negative"] / c["tiles"] if c["tiles"] else 0.0
        lines.append(
            f"{s:10s} {c['tiles']:7.0f} {tp:6.1f}% {c['boulders']:9.0f} {bp:6.1f}% "
            f"{c['negative']:8.0f} {neg_pct:6.1f}% {c['blocks']:7.0f}"
        )
    lines.append("")
    lines.append(
        "Target bands: tiles/instances train 70–75%, val/test 12.5–15%; "
        "train negatives ~25–40% (dataset ~45% negative at ≥1.5 m²)."
    )
    lines.append("")
    lines.append("Spatial integrity:")
    if not flags:
        lines.append("  ✓ No tile overlap between splits.")
        lines.append("  ✓ No train↔holdout adjacency (buffer applied).")
    else:
        for fl in flags:
            lines.append(f"  ✗ {fl}")
    lines.append("")
    lines.append("Outliers / flags:")
    if not outliers:
        lines.append("  (none)")
    else:
        for o in outliers:
            lines.append(f"  • {o}")
    lines.append("=" * 72)
    text = "\n".join(lines)
    print(text)
    return text


def find_outliers(blocks: list[Block], tiles: dict[str, TileMeta], counts: dict) -> list[str]:
    out = []
    dens = [b.density for b in blocks if b.n_boulders > 0]
    if dens:
        thr = np.percentile(dens, 95)
        for b in blocks:
            if b.density >= thr and b.n_boulders >= 8:
                out.append(
                    f"High-density block {b.block_id} ({b.tier}, split later): "
                    f"{b.n_boulders} boulders / {b.n_tiles} tiles "
                    f"(density={b.density:.1f})"
                )
    # Isolated positive tiles vs same-year PCA neighbors
    for year in (24, 25):
        ordered = sorted(
            (t for t in tiles.values() if t.year == year),
            key=lambda t: t.pca_score,
        )
        for i, t in enumerate(ordered):
            if t.n_boulders < 12:
                continue
            neigh = []
            if i > 0:
                neigh.append(ordered[i - 1])
            if i + 1 < len(ordered):
                neigh.append(ordered[i + 1])
            if neigh and all(n.is_negative for n in neigh):
                out.append(
                    f"Isolated high-count tile {t.year_key}: {t.n_boulders} boulders "
                    f"with negative same-year PCA neighbors"
                )
    usable = counts["train"]["tiles"] + counts["valid"]["tiles"] + counts["test"]["tiles"]
    if usable:
        for s in ("valid", "test"):
            share = counts[s]["boulders"] / max(sum(c["boulders"] for c in counts.values()), 1)
            if share < 0.08:
                out.append(f"Severe boulder under-representation in {s}: {share:.1%}")
            if share > 0.25:
                out.append(f"Boulder over-concentration in {s}: {share:.1%}")
        neg = counts["train"]["negative"] / max(counts["train"]["tiles"], 1)
        if neg < 0.15 or neg > 0.40:
            out.append(f"Train negative fraction outside 15–40% soft band: {neg:.1%}")
    return out


def main() -> None:
    global _TILE_LOOKUP

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=Path("segmentation"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("BoulderCalculator/experiments/geo_splits"),
    )
    parser.add_argument(
        "--tile-subdir",
        default="tiling_used",
        help="Subdir under segmentation with year tiles (default: tiling_used).",
    )
    parser.add_argument("--block-span-m", type=float, default=75.0)
    parser.add_argument("--max-tiles-per-block", type=int, default=3)
    parser.add_argument("--setup-id", default="stratified_coastal")
    parser.add_argument("--class-field", default=CLASS_FIELD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=1.5,
        help="Only Class=0 polygons with clipped area ≥ this (m²) count as "
        "boulders for density / balance (deposits never count). Default 1.5.",
    )
    args = parser.parse_args()

    seg = args.segmentation_dir
    tile_dir = seg / args.tile_subdir
    if not tile_dir.is_dir():
        tile_dir = seg / "tiling"
        print(f"WARNING: {args.tile_subdir} missing; falling back to {tile_dir}")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles_by_year = resolve_tiles_by_year(seg, None)
    years = [24, 25]
    print("Loading tile footprints…")
    bounds = load_bounds(tile_dir, years, tiles_by_year)
    scores = pca_axis_scores(bounds)

    tiles: dict[str, TileMeta] = {}
    for yk, b in bounds.items():
        parts = yk.split("_")
        year = int(parts[0])
        short = canonical_key(f"{int(parts[1])}_{int(parts[2])}")
        tiles[yk] = TileMeta(
            year_key=yk,
            year=year,
            short=short,
            bounds=b,
            pca_score=scores[yk],
        )

    print("Loading annotations…")
    ann_dir = seg / "annotations"
    annotations: list[tuple] = []
    for year in years:
        gpkg = ann_dir / f"july14_{year}.gpkg"
        if not gpkg.exists():
            raise FileNotFoundError(gpkg)
        annotations.extend(
            load_annotations(gpkg, None, args.class_field, boulder_only=True, year=year)
        )

    print(
        f"Counting instances per tile (min_area_m2={args.min_area_m2}, "
        "deposits excluded from boulder counts)…"
    )
    count_instances_per_tile(tiles, annotations, min_area_m2=args.min_area_m2)
    _TILE_LOOKUP = tiles

    n_neg = sum(1 for t in tiles.values() if t.is_negative)
    n_b = sum(t.n_boulders for t in tiles.values())
    n_all = sum(t.n_boulders_all for t in tiles.values())
    n_dep = sum(t.n_deposits for t in tiles.values())
    print(
        f"Dataset: {len(tiles)} tiles, {n_b} boulders ≥{args.min_area_m2} m² "
        f"({n_all} Class=0 any size, {n_dep} deposits ignored), "
        f"{n_neg} negative tiles ({100*n_neg/len(tiles):.1f}%)"
    )

    blocks = form_coastal_blocks(
        tiles,
        target_span_m=args.block_span_m,
        max_tiles=args.max_tiles_per_block,
    )
    assign_density_tiers(blocks)
    print(f"Formed {len(blocks)} coastal blocks "
          f"(span≈{args.block_span_m} m, max {args.max_tiles_per_block} tiles)")

    print("Searching stratified block assignments…")
    block_assign, tile_assign, pen, counts = optimize_split(
        blocks, bounds, seed=args.seed
    )

    # After buffer, some blocks may be mixed — report tile-level truth
    flags = verify_no_overlap(tile_assign) + verify_no_adjacent_leak(tile_assign, bounds)
    outliers = find_outliers(blocks, tiles, counts)
    # Annotate high-density outliers with final split
    for i, o in enumerate(outliers):
        if o.startswith("High-density block"):
            bid = int(o.split()[2])
            outliers[i] = o.replace("split later", f"split={block_assign.get(bid, '?')}")

    report = print_report(blocks, tiles, tile_assign, counts, flags, outliers)

    block_of = {}
    tiers = {}
    for b in blocks:
        tiers[b.block_id] = b.tier
        for yk in b.year_keys:
            block_of[yk] = b.block_id

    # YAML in repo format
    valid = {k for k, s in tile_assign.items() if s == "valid"}
    test = {k for k, s in tile_assign.items() if s == "test"}
    excluded = {k for k, s in tile_assign.items() if s == "excluded"}
    data = split_dict(
        args.setup_id,
        (
            f"Density-stratified coastal blocks (~{args.block_span_m:.0f} m PCA span); "
            f"boulder counts = Class=0 clipped area ≥{args.min_area_m2} m² "
            "(deposits excluded); tiers=high/medium_low/negative; abutting buffer; "
            f"search penalty={pen:.4f}."
        ),
        "geographic",
        valid,
        test,
        excluded,
    )
    write_yaml(out_dir / f"{args.setup_id}.yaml", data)
    write_split_geojson(
        out_dir / f"tile_extents_{args.setup_id}.geojson", bounds, tile_assign
    )

    meta_csv = out_dir / f"{args.setup_id}_dataset_metadata.csv"
    write_metadata_csv(meta_csv, tiles, tile_assign, block_of, tiers)
    print(f"Wrote {meta_csv}")
    block_csv = out_dir / f"{args.setup_id}_blocks.csv"
    write_block_csv(block_csv, blocks, block_assign)
    print(f"Wrote {block_csv}")

    report_path = out_dir / f"{args.setup_id}_report.txt"
    report_path.write_text(report + "\n")
    print(f"Wrote {report_path}")

    summary = {
        "setup_id": args.setup_id,
        "penalty": pen,
        "counts": counts,
        "n_tiles": len(tiles),
        "n_blocks": len(blocks),
        "flags": flags,
        "outliers": outliers,
    }
    (out_dir / f"{args.setup_id}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
