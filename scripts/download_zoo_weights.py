"""
Download pretrained weights for all CREDIT model zoo models.

Usage:
  python scripts/download_zoo_weights.py

Downloads to /glade/derecho/scratch/schreck/credit_zoo/<model>/
"""

import sys
import subprocess
import time
from pathlib import Path

DEST = Path("/glade/derecho/scratch/schreck/credit_zoo")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def hf_download(repo_id, filename, dest_dir):
    """Download one file from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(filename).name
    if dest_path.exists():
        print(f"  [skip] {dest_path.name} already exists")
        return dest_path
    print(f"  [dl]   {repo_id}/{filename}  →  {dest_path}")
    sys.stdout.flush()
    t0 = time.time()
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    # hf_hub_download preserves subdir structure inside local_dir;
    # move file to dest_dir root if it landed in a subdir
    downloaded = dest_dir / filename
    if downloaded.exists() and downloaded != dest_path:
        downloaded.rename(dest_path)
    elapsed = time.time() - t0
    size_gb = dest_path.stat().st_size / 1e9
    print(f"         {size_gb:.2f} GB in {elapsed:.0f}s")
    return dest_path


def wget_download(url, dest_dir):
    """Download one file via wget."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(url).name
    dest_path = dest_dir / fname
    if dest_path.exists():
        print(f"  [skip] {fname} already exists")
        return dest_path
    print(f"  [dl]   {url}  →  {dest_path}")
    sys.stdout.flush()
    t0 = time.time()
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(dest_path), url],
        check=True,
    )
    elapsed = time.time() - t0
    size_gb = dest_path.stat().st_size / 1e9
    print(f"         {size_gb:.2f} GB in {elapsed:.0f}s")
    return dest_path


def section(name):
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    sys.stdout.flush()


# ──────────────────────────────────────────────────────────────────────────────
# Stormer  (tungnd/stormer, MIT, 1.40625°)
# ──────────────────────────────────────────────────────────────────────────────


def download_stormer():
    section("Stormer  [tungnd/stormer, MIT]")
    d = DEST / "stormer"
    for f in [
        "stormer_1.40625_patch_size_2.ckpt",
        "stormer_1.40625_patch_size_4.ckpt",
    ]:
        hf_download("tungnd/stormer", f, d)


# ──────────────────────────────────────────────────────────────────────────────
# ClimaX  (tungnd/climax, MIT)
# ──────────────────────────────────────────────────────────────────────────────


def download_climax():
    section("ClimaX  [tungnd/climax, MIT]")
    d = DEST / "climax"
    for f in [
        "1.40625deg.ckpt",  # ERA5 fine-tune at 1.40625°
        "5.625deg.ckpt",  # ERA5 fine-tune at 5.625°
        "vits.ckpt",  # small CMIP6 pretrained
        "vitb.ckpt",  # base CMIP6 pretrained
        "vitl.ckpt",  # large CMIP6 pretrained
    ]:
        hf_download("tungnd/climax", f, d)


# ──────────────────────────────────────────────────────────────────────────────
# FourCastNet v1  (NERSC, BSD-3, 20-ch 0.25°)
# ──────────────────────────────────────────────────────────────────────────────


def download_fourcastnet():
    section("FourCastNet v1  [NERSC, BSD-3]")
    d = DEST / "fourcastnet"
    base = "https://portal.nersc.gov/project/m4134/FCN_weights_v0"
    wget_download(f"{base}/backbone.ckpt", d)
    stats_base = f"{base}/stats_v0"
    for fname in [
        "global_means.npy",
        "global_stds.npy",
        "land_sea_mask.npy",
        "latitude.npy",
        "longitude.npy",
        "time_means.npy",
    ]:
        wget_download(f"{stats_base}/{fname}", d / "stats")


# ──────────────────────────────────────────────────────────────────────────────
# FourCastNet v3  (nvidia/fourcastnet3, Apache-2, 72-ch 0.25°)
# ──────────────────────────────────────────────────────────────────────────────


def download_fourcastnet3():
    section("FourCastNet v3  [nvidia/fourcastnet3, Apache-2]")
    d = DEST / "fourcastnet3"
    # Main checkpoint (inside subdir on HF)
    from huggingface_hub import hf_hub_download

    ckpt_src = Path(
        hf_hub_download(
            repo_id="nvidia/fourcastnet3",
            filename="training_checkpoints/best_ckpt_mp0.tar",
            local_dir=str(d),
            local_dir_use_symlinks=False,
        )
    )
    # Move to root if needed
    dest_ckpt = d / "best_ckpt_mp0.tar"
    if ckpt_src != dest_ckpt and ckpt_src.exists():
        ckpt_src.rename(dest_ckpt)
        # Clean up empty subdir
        try:
            (d / "training_checkpoints").rmdir()
        except OSError:
            pass
    print(f"  checkpoint: {dest_ckpt}")

    # Ancillary files
    for f in [
        "global_means.npy",
        "global_stds.npy",
        "maxs.npy",
        "mins.npy",
        "land_mask.nc",
        "orography.nc",
        "config.json",
        "metadata.json",
    ]:
        hf_download("nvidia/fourcastnet3", f, d)


# ──────────────────────────────────────────────────────────────────────────────
# Aurora  (microsoft/aurora, MIT, 0.25°)
# ──────────────────────────────────────────────────────────────────────────────


def download_aurora():
    section("Aurora  [microsoft/aurora, MIT]")
    d = DEST / "aurora"
    for f in [
        "aurora-0.25-pretrained.ckpt",  # main 0.25° pretrained
        "aurora-0.25-finetuned.ckpt",  # ERA5 finetuned
        "aurora-0.25-small-pretrained.ckpt",  # small variant for testing
        "aurora-0.25-static.pickle",  # static fields needed at runtime
        "aurora-0.25-12h-pretrained.ckpt",  # 12h lead pretrained
    ]:
        hf_download("microsoft/aurora", f, d)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Download destination: {DEST}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    download_stormer()
    download_climax()
    download_fourcastnet()
    download_fourcastnet3()
    download_aurora()

    print(f"\n{'─' * 60}")
    print("All downloads complete.")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Print manifest
    print(f"\nManifest ({DEST}):")
    for p in sorted(DEST.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size / 1e9
            rel = p.relative_to(DEST)
            print(f"  {str(rel):<65}  {sz:.3f} GB")
