"""Quick visualization for boulder matching results.

Two modes:
  1. Screenshots: write overview + per-match crop PNGs (no GUI needed).
  2. GUI: matplotlib browser to flip through matches with optional ortho/DSM.

Examples:
  python -m matching.visualize --results-dir .../matching/results --outdir .../screenshots
  python -m matching.visualize --results-dir .../matching/results --gui \\
      --before-ortho .../2024/...Orthomosaic.tif --after-ortho .../2025/25IniSouthOrt.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
except ImportError:  # pragma: no cover
    rasterio = None


COLORS = {
    "match": "#2ecc71",
    "appeared": "#3498db",
    "disappeared": "#e74c3c",
    "vector": "#f39c12",
    "before": "#c0392b",
    "after": "#27ae60",
}


def _read_layer(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:25829")
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:25829")
    return gdf


def load_results(results_dir: Path) -> dict[str, gpd.GeoDataFrame]:
    return {
        "matches": _read_layer(results_dir / "matched_boulders.geojson"),
        "appeared": _read_layer(results_dir / "appeared_boulders.geojson"),
        "disappeared": _read_layer(results_dir / "disappeared_boulders.geojson"),
        "vectors": _read_layer(results_dir / "movement_vectors.geojson"),
    }


def load_inputs(
    before_path: Path | None, after_path: Path | None
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    before = gpd.read_file(before_path) if before_path and before_path.exists() else None
    after = gpd.read_file(after_path) if after_path and after_path.exists() else None
    return before, after


def _window_rgb(path: str | Path, bounds, max_size: int = 800) -> tuple[np.ndarray, tuple]:
    """Return (H,W,3) uint8 preview and extent (left, right, bottom, top)."""
    if rasterio is None:
        raise RuntimeError("rasterio is required for ortho/DSM previews")

    with rasterio.open(path) as src:
        left, bottom, right, top = bounds
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            raise ValueError("empty window")

        scale = max(window.width, window.height) / max_size
        if scale < 1:
            scale = 1.0
        out_h = max(1, int(window.height / scale))
        out_w = max(1, int(window.width / scale))

        count = min(3, src.count)
        data = src.read(
            indexes=list(range(1, count + 1)),
            window=window,
            out_shape=(count, out_h, out_w),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
        transform = src.window_transform(window)
        # Adjust transform for downsampling
        transform = transform * transform.scale(
            window.width / out_w, window.height / out_h
        )
        extent = (
            transform.c,
            transform.c + transform.a * out_w,
            transform.f + transform.e * out_h,
            transform.f,
        )

        if count == 1:
            band = data[0].astype(np.float32)
            # DSM / single-band stretch
            valid = band[np.isfinite(band) & (band != 0)]
            if valid.size:
                lo, hi = np.percentile(valid, [2, 98])
                if hi <= lo:
                    hi = lo + 1
                band = np.clip((band - lo) / (hi - lo), 0, 1)
            else:
                band = np.zeros_like(band)
            rgb = np.stack([band, band, band], axis=-1)
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = np.transpose(data, (1, 2, 0)).astype(np.float32)
            for i in range(rgb.shape[2]):
                band = rgb[:, :, i]
                valid = band[band > 0]
                if valid.size:
                    lo, hi = np.percentile(valid, [2, 98])
                    if hi <= lo:
                        hi = lo + 1
                    rgb[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)
                else:
                    rgb[:, :, i] = 0
            if rgb.shape[2] == 2:
                rgb = np.concatenate([rgb, rgb[:, :, :1]], axis=2)
            rgb = (rgb * 255).astype(np.uint8)

        return rgb, extent


def plot_overview(results: dict[str, gpd.GeoDataFrame], ax=None, title: str | None = None):
    ax = ax or plt.gca()
    matches = results["matches"]
    appeared = results["appeared"]
    disappeared = results["disappeared"]
    vectors = results["vectors"]

    if not disappeared.empty:
        disappeared.plot(ax=ax, facecolor="none", edgecolor=COLORS["disappeared"], linewidth=0.6, alpha=0.7)
    if not appeared.empty:
        appeared.plot(ax=ax, facecolor="none", edgecolor=COLORS["appeared"], linewidth=0.6, alpha=0.7)
    if not matches.empty:
        matches.plot(ax=ax, facecolor="none", edgecolor=COLORS["match"], linewidth=1.0, alpha=0.9)
    if not vectors.empty:
        vectors.plot(ax=ax, color=COLORS["vector"], linewidth=0.8, alpha=0.85)

    ax.set_aspect("equal")
    ax.set_title(
        title
        or (
            f"Matches={len(matches)}  Appeared={len(appeared)}  "
            f"Disappeared={len(disappeared)}"
        )
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(
        handles=[
            Patch(edgecolor=COLORS["match"], facecolor="none", label="Matched (after geom)"),
            Patch(edgecolor=COLORS["appeared"], facecolor="none", label="Appeared"),
            Patch(edgecolor=COLORS["disappeared"], facecolor="none", label="Disappeared"),
            Patch(facecolor=COLORS["vector"], label="Movement vector"),
        ],
        loc="upper right",
        fontsize=8,
    )
    return ax


def _match_pair_geoms(
    match_row,
    before: gpd.GeoDataFrame | None,
    after: gpd.GeoDataFrame | None,
):
    before_geom = None
    after_geom = match_row.geometry
    if before is not None and "before_id" in match_row.index:
        hits = before.index.isin([int(match_row["before_id"])])
        # before_id may be a column added later; fall back to positional
        if "before_id" in before.columns:
            subset = before[before["before_id"] == match_row["before_id"]]
            if not subset.empty:
                before_geom = subset.iloc[0].geometry
        elif int(match_row["before_id"]) < len(before):
            before_geom = before.iloc[int(match_row["before_id"])].geometry

    if after is not None and "after_id" in match_row.index:
        if "after_id" in after.columns:
            subset = after[after["after_id"] == match_row["after_id"]]
            if not subset.empty:
                after_geom = subset.iloc[0].geometry
        elif int(match_row["after_id"]) < len(after):
            after_geom = after.iloc[int(match_row["after_id"])].geometry

    return before_geom, after_geom


def plot_match_detail(
    match_row,
    before: gpd.GeoDataFrame | None,
    after: gpd.GeoDataFrame | None,
    before_raster: Path | None = None,
    after_raster: Path | None = None,
    pad_m: float = 8.0,
    ax=None,
):
    ax = ax or plt.gca()
    before_geom, after_geom = _match_pair_geoms(match_row, before, after)

    geoms = [g for g in (before_geom, after_geom) if g is not None]
    if not geoms:
        ax.set_title("No geometry")
        return ax

    minx = min(g.bounds[0] for g in geoms) - pad_m
    miny = min(g.bounds[1] for g in geoms) - pad_m
    maxx = max(g.bounds[2] for g in geoms) + pad_m
    maxy = max(g.bounds[3] for g in geoms) + pad_m
    bounds = (minx, miny, maxx, maxy)

    # Prefer after ortho; fall back to before; else blank
    shown = False
    for raster in (after_raster, before_raster):
        if raster and Path(raster).exists():
            try:
                rgb, extent = _window_rgb(raster, bounds)
                ax.imshow(rgb, extent=extent, origin="upper")
                shown = True
                break
            except Exception as exc:  # noqa: BLE001
                ax.text(
                    0.02,
                    0.02,
                    f"raster preview failed: {exc}",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="red",
                )

    if not shown:
        ax.set_facecolor("#1a1a1a")

    if before_geom is not None:
        gpd.GeoSeries([before_geom], crs="EPSG:25829").plot(
            ax=ax, facecolor="none", edgecolor=COLORS["before"], linewidth=2.0
        )
    if after_geom is not None:
        gpd.GeoSeries([after_geom], crs="EPSG:25829").plot(
            ax=ax, facecolor="none", edgecolor=COLORS["after"], linewidth=2.0
        )

    if before_geom is not None and after_geom is not None:
        xs = [before_geom.centroid.x, after_geom.centroid.x]
        ys = [before_geom.centroid.y, after_geom.centroid.y]
        ax.plot(xs, ys, color=COLORS["vector"], linewidth=1.5, marker="o", markersize=4)

    score = match_row.get("match_score", np.nan)
    dist = match_row.get("distance_m", np.nan)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_title(
        f"before={match_row.get('before_id')} → after={match_row.get('after_id')}  "
        f"score={score:.3f}  dist={dist:.2f}m"
    )
    ax.legend(
        handles=[
            Patch(edgecolor=COLORS["before"], facecolor="none", label="Before"),
            Patch(edgecolor=COLORS["after"], facecolor="none", label="After"),
        ],
        loc="upper right",
        fontsize=8,
    )
    return ax


def export_screenshots(
    results: dict[str, gpd.GeoDataFrame],
    outdir: Path,
    before: gpd.GeoDataFrame | None = None,
    after: gpd.GeoDataFrame | None = None,
    before_raster: Path | None = None,
    after_raster: Path | None = None,
    max_matches: int = 50,
    pad_m: float = 8.0,
):
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    plot_overview(results, ax=ax)
    fig.tight_layout()
    overview_path = outdir / "overview.png"
    fig.savefig(overview_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {overview_path}")

    matches = results["matches"]
    if matches.empty:
        print("No matches to screenshot.")
        return

    ranked = matches.sort_values("match_score", ascending=False).head(max_matches)
    for i, (_, row) in enumerate(ranked.iterrows()):
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_match_detail(
            row,
            before=before,
            after=after,
            before_raster=before_raster,
            after_raster=after_raster,
            pad_m=pad_m,
            ax=ax,
        )
        fig.tight_layout()
        path = outdir / f"match_{i:03d}_b{int(row['before_id'])}_a{int(row['after_id'])}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Wrote {path}")


def run_gui(
    results: dict[str, gpd.GeoDataFrame],
    before: gpd.GeoDataFrame | None = None,
    after: gpd.GeoDataFrame | None = None,
    before_raster: Path | None = None,
    after_raster: Path | None = None,
    pad_m: float = 8.0,
):
    matches = results["matches"].sort_values("match_score", ascending=False).reset_index(drop=True)
    if matches.empty:
        print("No matches — showing overview only.")
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_overview(results, ax=ax)
        plt.show()
        return

    state = {"idx": 0}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.canvas.manager.set_window_title("Boulder match browser")

    def redraw():
        axes[0].cla()
        axes[1].cla()
        plot_overview(results, ax=axes[0], title="Overview (n=next, p=prev, o=overview focus)")
        row = matches.iloc[state["idx"]]
        # Highlight current vector if present
        if not results["vectors"].empty:
            cur = results["vectors"][
                (results["vectors"]["before_id"] == row["before_id"])
                & (results["vectors"]["after_id"] == row["after_id"])
            ]
            if not cur.empty:
                cur.plot(ax=axes[0], color="yellow", linewidth=2.5)
        plot_match_detail(
            row,
            before=before,
            after=after,
            before_raster=before_raster,
            after_raster=after_raster,
            pad_m=pad_m,
            ax=axes[1],
        )
        axes[1].set_xlabel(f"Match {state['idx'] + 1} / {len(matches)}")
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("n", "right"):
            state["idx"] = (state["idx"] + 1) % len(matches)
            redraw()
        elif event.key in ("p", "left"):
            state["idx"] = (state["idx"] - 1) % len(matches)
            redraw()
        elif event.key == "o":
            # zoom overview to current match
            row = matches.iloc[state["idx"]]
            before_geom, after_geom = _match_pair_geoms(row, before, after)
            geoms = [g for g in (before_geom, after_geom, row.geometry) if g is not None]
            if geoms:
                minx = min(g.bounds[0] for g in geoms) - 25
                miny = min(g.bounds[1] for g in geoms) - 25
                maxx = max(g.bounds[2] for g in geoms) + 25
                maxy = max(g.bounds[3] for g in geoms) + 25
                axes[0].set_xlim(minx, maxx)
                axes[0].set_ylim(miny, maxy)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    print("GUI keys: n/→ next, p/← previous, o zoom overview to current match")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize boulder matching results")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--outdir", type=Path, default=None, help="Screenshot output dir")
    parser.add_argument("--before", type=Path, default=None, help="Before polygons (for outlines)")
    parser.add_argument("--after", type=Path, default=None, help="After polygons (for outlines)")
    parser.add_argument("--before-ortho", type=Path, default=None)
    parser.add_argument("--after-ortho", type=Path, default=None)
    parser.add_argument("--before-dsm", type=Path, default=None, help="Fallback single-band preview")
    parser.add_argument("--after-dsm", type=Path, default=None)
    parser.add_argument("--max-matches", type=int, default=40)
    parser.add_argument("--pad-m", type=float, default=8.0)
    parser.add_argument("--gui", action="store_true", help="Open interactive browser")
    parser.add_argument("--no-screenshots", action="store_true")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    before, after = load_inputs(args.before, args.after)

    # Attach id columns if missing (matcher indices)
    if before is not None and "before_id" not in before.columns:
        before = before.copy()
        before["before_id"] = before.index
    if after is not None and "after_id" not in after.columns:
        after = after.copy()
        after["after_id"] = after.index

    before_raster = args.before_ortho or args.before_dsm
    after_raster = args.after_ortho or args.after_dsm

    if not args.no_screenshots:
        outdir = args.outdir or (args.results_dir / "screenshots")
        export_screenshots(
            results,
            outdir=outdir,
            before=before,
            after=after,
            before_raster=before_raster,
            after_raster=after_raster,
            max_matches=args.max_matches,
            pad_m=args.pad_m,
        )

    if args.gui:
        run_gui(
            results,
            before=before,
            after=after,
            before_raster=before_raster,
            after_raster=after_raster,
            pad_m=args.pad_m,
        )
    elif results["matches"].empty:
        # still print a text summary when no screenshots were requested?
        print(
            f"Matches={len(results['matches'])} "
            f"Appeared={len(results['appeared'])} "
            f"Disappeared={len(results['disappeared'])}"
        )


if __name__ == "__main__":
    main()
