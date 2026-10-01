from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "configs/main.yaml",
    "configs/scaling.yaml",
    "configs/ablation.yaml",
    "configs/confirmation.yaml",
    "configs/trec_covid.yaml",
)


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> None:
    for relative in CONFIGS:
        config_path = ROOT / relative
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output_dir = Path(config["output_dir"])
        run("scripts/run_experiments.py", "--config", relative)
        run(
            "scripts/analyze_results.py",
            "--input",
            str(output_dir / "results_full.csv"),
            "--output",
            str(output_dir / "analysis"),
        )
    for part in ("weights", "bid", "scale"):
        run(
            "scripts/run_sensitivity.py",
            "--part",
            part,
            "--output-dir",
            f"results/sensitivity/{part}",
        )


if __name__ == "__main__":
    main()
