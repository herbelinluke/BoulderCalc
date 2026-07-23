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
        "missed_candidates": _read_layer(results_dir / "missed_candidates.geojson"),
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


def _match_bounds(before_geom, after_geom, pad_m: float):
    geoms = [g for g in (before_geom, after_geom) if g is not None]
    if not geoms:
        return None
    minx = min(g.bounds[0] for g in geoms) - pad_m
    miny = min(g.bounds[1] for g in geoms) - pad_m
    maxx = max(g.bounds[2] for g in geoms) + pad_m
    maxy = max(g.bounds[3] for g in geoms) + pad_m
    return (minx, miny, maxx, maxy)


def _pick_pair_rasters(
    bounds,
    before_raster: Path | None,
    after_raster: Path | None,
    pair_tiles: list[tuple[str, str]] | None,
) -> tuple[Path | None, Path | None]:
    """Choose 2024/2025 rasters whose footprint covers the match (if pair list given)."""
    if not pair_tiles or bounds is None or rasterio is None:
        return before_raster, after_raster

    cx = 0.5 * (bounds[0] + bounds[2])
    cy = 0.5 * (bounds[1] + bounds[3])
    for b24, b25 in pair_tiles:
        p24, p25 = Path(b24), Path(b25)
        if not p24.exists() or not p25.exists():
            continue
        try:
            with rasterio.open(p25) as src:
                left, bottom, right, top = src.bounds
            if left <= cx <= right and bottom <= cy <= top:
                return p24, p25
        except Exception:  # noqa: BLE001
            continue
    return Path(pair_tiles[0][0]), Path(pair_tiles[0][1])


def _draw_displacement_vector(ax, before_geom, after_geom, color=None):
    """Draw before→after centroid arrow (may extend past panel crop)."""
    if before_geom is None or after_geom is None:
        return
    color = color or COLORS["vector"]
    x0, y0 = before_geom.centroid.x, before_geom.centroid.y
    x1, y1 = after_geom.centroid.x, after_geom.centroid.y
    if abs(x1 - x0) < 1e-6 and abs(y1 - y0) < 1e-6:
        return
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=2.0,
            mutation_scale=14,
        ),
        clip_on=False,
        zorder=5,
    )
    ax.plot([x0], [y0], marker="o", color=COLORS["before"], markersize=4, zorder=6)
    ax.plot([x1], [y1], marker="o", color=COLORS["after"], markersize=4, zorder=6)


def _draw_panel(
    ax,
    raster: Path | None,
    geom,
    bounds,
    edgecolor: str,
    title: str,
    before_geom=None,
    after_geom=None,
    draw_vector: bool = False,
):
    shown = False
    if raster and Path(raster).exists() and bounds is not None:
        try:
            rgb, extent = _window_rgb(raster, bounds)
            ax.imshow(rgb, extent=extent, origin="upper")
            shown = True
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

    if geom is not None:
        gpd.GeoSeries([geom], crs="EPSG:25829").plot(
            ax=ax, facecolor="none", edgecolor=edgecolor, linewidth=2.0
        )
        ax.plot(
            [geom.centroid.x],
            [geom.centroid.y],
            marker="o",
            color=edgecolor,
            markersize=5,
        )

    if draw_vector:
        _draw_displacement_vector(ax, before_geom, after_geom)

    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_title(title)


def plot_match_detail(
    match_row,
    before: gpd.GeoDataFrame | None,
    after: gpd.GeoDataFrame | None,
    before_raster: Path | None = None,
    after_raster: Path | None = None,
    pad_m: float = 8.0,
    ax=None,
    side_by_side: bool = True,
    pair_tiles: list[tuple[str, str]] | None = None,
    axes=None,
    draw_vector: bool = True,
):
    """Plot one match. Default is side-by-side 2024 | 2025 panels."""
    before_geom, after_geom = _match_pair_geoms(match_row, before, after)
    bounds = _match_bounds(before_geom, after_geom, pad_m)
    before_raster, after_raster = _pick_pair_rasters(
        bounds, before_raster, after_raster, pair_tiles
    )

    score = match_row.get("match_score", np.nan)
    dist = match_row.get("distance_m", np.nan)
    subtitle = (
        f"before={match_row.get('before_id')} → after={match_row.get('after_id')}  "
        f"score={score:.3f}  dist={dist:.2f}m"
    )

    if side_by_side:
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        else:
            fig = axes[0].figure
        _draw_panel(
            axes[0],
            before_raster,
            before_geom,
            bounds,
            COLORS["before"],
            "2024 (before)",
            before_geom=before_geom,
            after_geom=after_geom,
            draw_vector=draw_vector,
        )
        _draw_panel(
            axes[1],
            after_raster,
            after_geom,
            bounds,
            COLORS["after"],
            "2025 (after)",
            before_geom=before_geom,
            after_geom=after_geom,
            draw_vector=draw_vector,
        )
        fig.suptitle(subtitle, fontsize=11)
        return axes

    # Legacy single-panel overlay
    ax = ax or plt.gca()
    if bounds is None:
        ax.set_title("No geometry")
        return ax
    shown = False
    for raster in (after_raster, before_raster):
        if raster and Path(raster).exists():
            try:
                rgb, extent = _window_rgb(raster, bounds)
                ax.imshow(rgb, extent=extent, origin="upper")
                shown = True
                break
            except Exception as exc:  # noqa: BLE001
                ax.text(0.02, 0.02, f"raster preview failed: {exc}", transform=ax.transAxes, fontsize=7, color="red")
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
    if draw_vector:
        _draw_displacement_vector(ax, before_geom, after_geom)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_title(subtitle)
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
    side_by_side: bool = True,
    pair_tiles: list[tuple[str, str]] | None = None,
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
        if side_by_side:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            plot_match_detail(
                row,
                before=before,
                after=after,
                before_raster=before_raster,
                after_raster=after_raster,
                pad_m=pad_m,
                side_by_side=True,
                pair_tiles=pair_tiles,
                axes=axes,
            )
        else:
            fig, ax = plt.subplots(figsize=(7, 7))
            plot_match_detail(
                row,
                before=before,
                after=after,
                before_raster=before_raster,
                after_raster=after_raster,
                pad_m=pad_m,
                side_by_side=False,
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
    side_by_side: bool = True,
    pair_tiles: list[tuple[str, str]] | None = None,
    overview_pad_m: float = 25.0,
):
    matches = results["matches"].sort_values("match_score", ascending=False).reset_index(drop=True)
    if matches.empty:
        print("No matches — showing overview only.")
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_overview(results, ax=ax)
        plt.show()
        return

    state = {"idx": 0, "overview_zoomed": True}

    if side_by_side:
        fig = plt.figure(figsize=(16, 7))
        ax_overview = fig.add_subplot(1, 3, 1)
        ax_before = fig.add_subplot(1, 3, 2)
        ax_after = fig.add_subplot(1, 3, 3)
        detail_axes = [ax_before, ax_after]
    else:
        fig, (ax_overview, ax_detail) = plt.subplots(1, 2, figsize=(14, 7))
        detail_axes = None

    fig.canvas.manager.set_window_title("Boulder match browser")

    def _overview_full_extent():
        geoms = []
        for key in ("matches", "appeared", "disappeared", "vectors"):
            gdf = results.get(key)
            if gdf is not None and not gdf.empty:
                geoms.extend(list(gdf.geometry))
        if not geoms:
            return None
        minx = min(g.bounds[0] for g in geoms)
        miny = min(g.bounds[1] for g in geoms)
        maxx = max(g.bounds[2] for g in geoms)
        maxy = max(g.bounds[3] for g in geoms)
        pad = max(5.0, 0.05 * max(maxx - minx, maxy - miny))
        return (minx - pad, miny - pad, maxx + pad, maxy + pad)

    full_extent = _overview_full_extent()

    def _apply_overview_zoom(row):
        if state["overview_zoomed"]:
            before_geom, after_geom = _match_pair_geoms(row, before, after)
            geoms = [g for g in (before_geom, after_geom, row.geometry) if g is not None]
            if geoms:
                minx = min(g.bounds[0] for g in geoms) - overview_pad_m
                miny = min(g.bounds[1] for g in geoms) - overview_pad_m
                maxx = max(g.bounds[2] for g in geoms) + overview_pad_m
                maxy = max(g.bounds[3] for g in geoms) + overview_pad_m
                ax_overview.set_xlim(minx, maxx)
                ax_overview.set_ylim(miny, maxy)
                return
        if full_extent is not None:
            ax_overview.set_xlim(full_extent[0], full_extent[2])
            ax_overview.set_ylim(full_extent[1], full_extent[3])

    def redraw():
        ax_overview.cla()
        zoom_hint = "zoomed" if state["overview_zoomed"] else "full"
        plot_overview(
            results,
            ax=ax_overview,
            title=f"Overview ({zoom_hint})  n/p flip, o toggle zoom",
        )
        row = matches.iloc[state["idx"]]
        before_geom, after_geom = _match_pair_geoms(row, before, after)

        # Highlight current pair polygons on the left map
        if before_geom is not None:
            gpd.GeoSeries([before_geom], crs="EPSG:25829").plot(
                ax=ax_overview,
                facecolor="none",
                edgecolor="yellow",
                linewidth=2.5,
                zorder=6,
            )
        if after_geom is not None:
            gpd.GeoSeries([after_geom], crs="EPSG:25829").plot(
                ax=ax_overview,
                facecolor="none",
                edgecolor="cyan",
                linewidth=2.5,
                zorder=6,
            )
        if not results["vectors"].empty:
            cur = results["vectors"][
                (results["vectors"]["before_id"] == row["before_id"])
                & (results["vectors"]["after_id"] == row["after_id"])
            ]
            if not cur.empty:
                cur.plot(ax=ax_overview, color="yellow", linewidth=2.5, zorder=7)
        elif before_geom is not None and after_geom is not None:
            _draw_displacement_vector(ax_overview, before_geom, after_geom, color="yellow")

        _apply_overview_zoom(row)

        if side_by_side:
            for a in detail_axes:
                a.cla()
            plot_match_detail(
                row,
                before=before,
                after=after,
                before_raster=before_raster,
                after_raster=after_raster,
                pad_m=pad_m,
                side_by_side=True,
                pair_tiles=pair_tiles,
                axes=detail_axes,
                draw_vector=True,
            )
            detail_axes[1].set_xlabel(f"Match {state['idx'] + 1} / {len(matches)}")
        else:
            ax_detail.cla()
            plot_match_detail(
                row,
                before=before,
                after=after,
                before_raster=before_raster,
                after_raster=after_raster,
                pad_m=pad_m,
                side_by_side=False,
                ax=ax_detail,
                draw_vector=True,
            )
            ax_detail.set_xlabel(f"Match {state['idx'] + 1} / {len(matches)}")

        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("n", "right"):
            state["idx"] = (state["idx"] + 1) % len(matches)
            redraw()
        elif event.key in ("p", "left"):
            state["idx"] = (state["idx"] - 1) % len(matches)
            redraw()
        elif event.key in ("o", "O"):
            state["overview_zoomed"] = not state["overview_zoomed"]
            redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    print(
        "GUI keys: n/→ next, p/← previous, o toggle overview zoom "
        "(starts zoomed on current boulder pair)"
    )
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
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Single panel with both outlines overlaid (default: side-by-side 2024|2025).",
    )
    parser.add_argument(
        "--pair-tiles",
        type=str,
        default=None,
        help="Comma-separated 24tif:25tif pairs for per-match background picking.",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    before, after = load_inputs(args.before, args.after)

    if before is not None and "before_id" not in before.columns:
        before = before.copy()
        before["before_id"] = before.index
    if after is not None and "after_id" not in after.columns:
        after = after.copy()
        after["after_id"] = after.index

    before_raster = args.before_ortho or args.before_dsm
    after_raster = args.after_ortho or args.after_dsm
    pair_tiles = None
    if args.pair_tiles:
        pair_tiles = []
        for item in args.pair_tiles.split(","):
            left, right = item.split(":")
            pair_tiles.append((left.strip(), right.strip()))

    side_by_side = not args.overlay
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
            side_by_side=side_by_side,
            pair_tiles=pair_tiles,
        )

    if args.gui:
        run_gui(
            results,
            before=before,
            after=after,
            before_raster=before_raster,
            after_raster=after_raster,
            pad_m=args.pad_m,
            side_by_side=side_by_side,
            pair_tiles=pair_tiles,
        )
    elif results["matches"].empty:
        print(
            f"Matches={len(results['matches'])} "
            f"Appeared={len(results['appeared'])} "
            f"Disappeared={len(results['disappeared'])}"
        )


if __name__ == "__main__":
    main()
