#!/usr/bin/env python3
"""
Submit SLURM jobs to sweep over KL ramp hyperparameters and seeds.

Reads a base .slurm script (default: _bc.slurm), generates one .slurm file per
(seed, discovery_kl_loss_weight, discovery_kl_loss_warmup_steps) combination,
and sbatch-submits each. Checkpoints and meta_controller checkpoints are written
to unique dirs per run so runs do not overwrite each other.

Usage:
  python run_kl_ramp_sweep.py                    # use defaults
  python run_kl_ramp_sweep.py --dry_run          # only write .slurm files, do not sbatch
  python run_kl_ramp_sweep.py --base_slurm _bc_bm.slurm
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


# Default sweep grid (KL ramp study + seeds)
SEEDS = [42, 456, 1011]
DISCOVERY_KL_LOSS_WEIGHTS = [0.01, 0.1, 0.2]
DISCOVERY_KL_WARMUP_STEPS = [1000, 2000, 4000]

# Subdir under checkpoints for this sweep
SWEEP_CHECKPOINT_DIR = "kl_ramp_sweep"


def main():
    parser = argparse.ArgumentParser(description="Submit KL ramp sweep jobs")
    parser.add_argument(
        "--base_slurm",
        type=Path,
        default=Path("_bc.slurm"),
        help="Base .slurm script to use (same SBATCH header and train args, we override seed/kl/warmup/paths)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("slurm_kl_ramp_sweep"),
        help="Directory to write generated .slurm files",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help=f"Seeds to run (default: {SEEDS})",
    )
    parser.add_argument(
        "--discovery_kl_loss_weight",
        type=float,
        nargs="+",
        default=DISCOVERY_KL_LOSS_WEIGHTS,
        help=f"Final KL loss weights (default: {DISCOVERY_KL_LOSS_WEIGHTS})",
    )
    parser.add_argument(
        "--discovery_kl_loss_warmup_steps",
        type=int,
        nargs="+",
        default=DISCOVERY_KL_WARMUP_STEPS,
        help=f"KL warmup steps (default: {DISCOVERY_KL_WARMUP_STEPS})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only write .slurm files, do not run sbatch",
    )
    args = parser.parse_args()

    base_slurm = args.base_slurm
    if not base_slurm.is_file():
        raise FileNotFoundError(f"Base slurm script not found: {base_slurm}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_content = base_slurm.read_text()

    # Remove commented-out arguments that have trailing backslashes (e.g. "    # --flag \")
    base_content = re.sub(r"\s*#.*\\\s*$", "", base_content, flags=re.MULTILINE)

    # Remove dangling continuation backslash on the last non-blank line
    lines = base_content.rstrip().split("\n")
    if lines and lines[-1].rstrip().endswith("\\"):
        lines[-1] = lines[-1].rstrip()[:-1].rstrip()
    base_content = "\n".join(lines) + "\n"

    submitted = []
    for seed in args.seeds:
        for kl_weight in args.discovery_kl_loss_weight:
            for warmup in args.discovery_kl_loss_warmup_steps:
                run_id = f"seed{seed}_kl{kl_weight}_w{warmup}"
                job_name = f"bc-kl-{run_id}"
                checkpoint_subdir = f"{SWEEP_CHECKPOINT_DIR}/{run_id}"
                checkpoint_path = f"checkpoints/{checkpoint_subdir}/model.pt"
                meta_path = f"checkpoints/{checkpoint_subdir}/meta_controller.pt"

                # Patch base script: job name and run-specific args (single source of truth)
                slurm_content = re.sub(
                    r"#SBATCH\s+-J\s+\S+",
                    f"#SBATCH -J {job_name}",
                    base_content,
                    count=1,
                )
                slurm_content = re.sub(
                    r"--checkpoint_path\s+\S+",
                    f"--checkpoint_path {checkpoint_path}",
                    slurm_content,
                )
                slurm_content = re.sub(
                    r"--meta_controller_checkpoint_path\s+\S+",
                    f"--meta_controller_checkpoint_path {meta_path}",
                    slurm_content,
                )
                slurm_content = re.sub(
                    r"--discovery_kl_loss_weight\s+[\d.]+",
                    f"--discovery_kl_loss_weight {kl_weight}",
                    slurm_content,
                )
                slurm_content = re.sub(
                    r"--discovery_kl_loss_warmup_steps\s+\d+",
                    f"--discovery_kl_loss_warmup_steps {warmup}",
                    slurm_content,
                )
                if re.search(r"--run_seed\s+\d+", slurm_content):
                    slurm_content = re.sub(r"--run_seed\s+\d+", f"--run_seed {seed}", slurm_content)
                else:
                    # Insert --run_seed after --env_id; use lambda to avoid re.sub escape issues
                    slurm_content = re.sub(
                        r"(--env_id\s+\S+)\s*\\",
                        lambda m: f"{m.group(1)} \\\n    --run_seed {seed} \\",
                        slurm_content,
                        count=1,
                    )
                # Ensure the checkpoint directory exists before the job runs
                Path(f"checkpoints/{checkpoint_subdir}").mkdir(parents=True, exist_ok=True)

                out_slurm = args.out_dir / f"_bc_kl_ramp_{run_id}.slurm"
                out_slurm.write_text(slurm_content)
                submitted.append((out_slurm, run_id))

                if not args.dry_run:
                    subprocess.run(["sbatch", str(out_slurm)], check=True, cwd=Path.cwd())
                    print(f"Submitted {run_id} -> {out_slurm}")

    if args.dry_run:
        print(f"Dry run: wrote {len(submitted)} .slurm files under {args.out_dir}/")
        for p, rid in submitted:
            print(f"  {p}  ({rid})")
    else:
        print(f"Submitted {len(submitted)} jobs.")


if __name__ == "__main__":
    main()
