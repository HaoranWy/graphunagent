#!/usr/bin/env python3
"""Build the algorithm-only Q-RASC overview from the user's vector draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "figures/q-rasc-algorithm-overview.tex"
DEFAULT_DRAFT = ROOT / "AuthorKit27/Figures/overview-draft.pdf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-6000:]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the algorithm-only Q-RASC overview PDF."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--tex-figure-dir", type=Path, default=ROOT / "AuthorKit27/Figures")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    draft = DEFAULT_DRAFT.resolve()
    if not source.exists() or not draft.exists():
        raise FileNotFoundError("algorithm overlay source or user draft is missing")
    for executable in ("pdflatex", "gs", "pdftoppm"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable is unavailable: {executable}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tex_figure_dir.mkdir(parents=True, exist_ok=True)
    editable_pdf = args.output_dir / "q-rasc-algorithm-overview.pdf"
    paper_pdf = args.tex_figure_dir / "q-rasc-algorithm-overview.pdf"
    preview_png = args.output_dir / "q-rasc-algorithm-overview.png"

    with tempfile.TemporaryDirectory(prefix="qrasc-algorithm-", dir="/tmp") as temp_name:
        temp_dir = Path(temp_name)
        _run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={temp_dir}",
                source.name,
            ],
            source.parent,
        )
        built_pdf = temp_dir / f"{source.stem}.pdf"
        shutil.copyfile(built_pdf, editable_pdf)
        _run(
            [
                "gs",
                "-q",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                "-dPDFSETTINGS=/prepress",
                "-dNoOutputFonts",
                f"-sOutputFile={paper_pdf}",
                str(editable_pdf),
            ],
            ROOT,
        )
        preview_prefix = temp_dir / "q-rasc-algorithm-overview"
        _run(
            [
                "pdftoppm",
                "-png",
                "-r",
                "300",
                "-singlefile",
                str(paper_pdf),
                str(preview_prefix),
            ],
            ROOT,
        )
        shutil.copyfile(preview_prefix.with_suffix(".png"), preview_png)

    sidecar = {
        "schema": "q-rasc-algorithm-overview-v1",
        "paper_label": "Q-RASC",
        "source_draft": {"path": _relative(draft), "sha256": _sha256(draft)},
        "overlay_source": {"path": _relative(source), "sha256": _sha256(source)},
        "transform": {
            "crop_left_pt": 96.50,
            "source_page_pt": [493.92, 187.20],
            "output_page_pt": [397.42, 187.20],
            "vector_corrections": [
                "rename PWSC-CI to Q-RASC",
                "state q_i = c_i v_i explicitly in the local-quality block",
                "remove the uncertainty-to-answer-selection arrow",
                "keep answer selection structurally detached as an independent output",
            ],
        },
        "artifacts": {
            "editable_pdf": {"path": _relative(editable_pdf), "sha256": _sha256(editable_pdf)},
            "paper_pdf": {"path": _relative(paper_pdf), "sha256": _sha256(paper_pdf)},
            "preview_png": {"path": _relative(preview_png), "sha256": _sha256(preview_png)},
        },
    }
    sidecar_path = args.output_dir / "q-rasc-algorithm-overview-sidecar.json"
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "editable_pdf": _relative(editable_pdf),
                "paper_pdf": _relative(paper_pdf),
                "sidecar": _relative(sidecar_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
