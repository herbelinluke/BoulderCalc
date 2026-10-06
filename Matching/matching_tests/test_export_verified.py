# test_export_verified.py

from __future__ import annotations

import json
from pathlib import Path

from matching.evaluate_matches import LABEL_CONFIRMED, LABEL_NOT, save_labels_db
from matching.export_verified import export_verified


def test_export_verified_writes_confirmed_only(tmp_path: Path):
    outdir = tmp_path / "run"
    eval_dir = outdir / "eval"
    eval_dir.mkdir(parents=True)

    db = {
        "schema_version": 1,
        "labels": {
            "b0_a1": {
                "label_id": "b0_a1",
                "before_id": 0,
                "after_id": 1,
                "before_fid": 10,
                "after_fid": 20,
                "label": LABEL_CONFIRMED,
                "match_score": 0.9,
                "distance_m": 12.0,
                "dx": 10.0,
                "dy": 2.0,
                "before_area": 2.0,
                "after_area": 2.1,
                "before_volume": 1.2,
                "after_volume": 1.3,
                "after": {
                    "wkt": "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                    "geojson": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                    },
                },
            },
            "b2_a3": {
                "label_id": "b2_a3",
                "before_id": 2,
                "after_id": 3,
                "label": LABEL_NOT,
                "match_score": 0.4,
                "after": {"wkt": "POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))"},
            },
        },
    }
    labels_path = eval_dir / "match_labels.json"
    save_labels_db(db, labels_path, also_geojson=False)

    verified_dir = export_verified(outdir)
    assert (verified_dir / "verified_matches.geojson").exists()
    assert (verified_dir / "verified_pairs.csv").exists()
    audit = json.loads((verified_dir / "match_labels_confirmed.json").read_text())
    assert audit["n_confirmed"] == 1
    assert audit["labels"][0]["label_id"] == "b0_a1"
