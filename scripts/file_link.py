#!/usr/bin/env python3
"""Hard-link / symlink / copy helpers shared by COCO builders.

Prefer hard links on Windows guest (same NTFS volume, no admin, no extra bytes).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def unlink_dst(dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()


def link_or_copy(src: Path, dst: Path, mode: str = "hard") -> str:
    """Create dst referring to src. Returns the mode actually used.

    Modes:
      hard    — os.link (same volume; no admin; no extra bytes)
      symlink — os.symlink (may need admin / Developer Mode on Windows)
      copy    — shutil.copy2 (full duplicate; avoid for large pools)
      auto    — hard → symlink → copy
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    unlink_dst(dst)
    src = src.resolve()

    def try_hard() -> bool:
        try:
            os.link(src, dst)
            return True
        except OSError:
            return False

    def try_symlink() -> bool:
        try:
            os.symlink(src, dst)
            return True
        except OSError:
            return False

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "hard":
        if not try_hard():
            raise OSError(
                f"Hard link failed for {dst.name}. Source and destination must "
                "share one NTFS volume. Use --link-mode copy only as a last resort."
            )
        return "hard"
    if mode == "symlink":
        if not try_symlink():
            raise OSError(
                f"Symlink failed for {dst.name}. On Windows guest enable Developer "
                "Mode or use --link-mode hard (default)."
            )
        return "symlink"
    if try_hard():
        return "hard"
    if try_symlink():
        return "symlink"
    shutil.copy2(src, dst)
    return "copy"


def add_link_mode_argument(parser, *, default: str = "hard") -> None:
    parser.add_argument(
        "--link-mode",
        choices=("hard", "symlink", "copy", "auto"),
        default=default,
        help=(
            "How to place images in the output dir (default: hard). "
            "Hard links reuse bytes on the same volume — preferred on Windows guest."
        ),
    )
