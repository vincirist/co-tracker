# BOP onboarding: CoTracker3 tracks + SfM

Camera poses and a point cloud of an object from a BOP onboarding video (HOPE static/dynamic), without depth and
without a pose network: CoTracker3 tracks points inside `mask_visib` through the video, COLMAP's incremental mapper
reconstructs poses and points from these tracks with the known intrinsics.

## Setup

Needs [uv](https://docs.astral.sh/uv/), Linux x86_64 and an NVIDIA driver for CUDA 12.1 (no CUDA toolkit or
compiler). The environment is pinned in `pyproject.toml` / `uv.lock`:

```bash
uv sync
mkdir -p checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth -O checkpoints/cotracker3_scaled_offline.pth
```

The BOP dataset is expected at `/home/vincent/2_resources/bop_h3/hope` (`--src` / `--data` to change it).

## Usage

```bash
uv run python scripts/track_video.py --object 2 --onboarding dynamic --stride 2
uv run python scripts/run_tracks.py --src output/hope/dynamic/obj_000002/video
```

- `track_video.py`: fixed 512x384 window around the object, CoTracker3 offline with adaptive keyframes ->
  `output/hope/<onboarding>/<object>/video/` (`tracks.npz`, `video.json`, `tracks.mp4` to check the tracks)
- `run_tracks.py`: COLMAP database from the tracks + incremental mapping with K fixed ->
  `output/hope/<onboarding>/<object>/sfm/` (`cameras.json`, `points.npz`, `run.json`, COLMAP model), then a viser
  viewer at http://localhost:8080 (`--no-viz` to skip)

The offline tracker needs the whole video in GPU memory: ~140 frames at 512x384 fit next to ~9 GB of other GPU
use, the full 276 frames of HOPE dynamic object 2 did not (`--stride 2`).
