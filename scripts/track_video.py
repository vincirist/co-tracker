"""Track the object through a whole BOP onboarding video with CoTracker3 (offline) and render the tracks.

The video is cut to one fixed window per sequence: the union of all mask_visib boxes + 5% padding, made 4:3, at
least 512x384 original pixels, written at CoTracker3's model resolution 512x384 (so its internal resize is a
no-op; downscaling only). Full images, no background fill: the mask only decides which points are tracked.

Queries are a grid (every --grid px) inside the eroded mask of keyframes. Keyframes are chosen adaptively: frame
0 first, then the next frame in which fewer than --reseed of the current keyframe's points are still observed.
Each keyframe's points are tracked forwards and backwards through the whole video, so neighbouring keyframes
share points and a point seen again later keeps its identity. An observation counts where CoTracker says
visible and the point lies inside mask_visib of that frame (drops the hand and the background).

Writes <out>/<onboarding>/<object>/video/:
  video.json   sequence, frame ids, window [x0, y0, x1, y1] in original pixels, K of the window at 512x384,
               keyframes, settings
  tracks.npz   tracks (T,P,2) float32 in window pixels, observed (T,P) bool, query_frame (P,) int, colors (P,3)
               uint8 from the query frame
  tracks.mp4   the window video with the observed points, coloured by keyframe, and the mask outline

Usage:
  .venv/bin/python scripts/track_video.py --object 2 --onboarding dynamic [--stride 1] [--grid 6]
"""
import argparse
import json
import os
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from cotracker.predictor import CoTrackerPredictor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = '/home/vincent/2_resources/bop_h3/hope'
WINDOW_SIZE = (512, 384)  # CoTracker3 model resolution (W, H)
WINDOW_PAD = 0.05  # relative padding per side of the union mask box
MASK_ERODE = 2  # px at 512x384: queries keep this distance to the mask border
VIS_COLORS = (np.array([cv2.applyColorMap(np.uint8([[int(255 * i / 11)]]), cv2.COLORMAP_TURBO)[0, 0] for i in range(12)])
              [:, ::-1])  # RGB, cycled per keyframe


def crop_box(mask):
    """4:3 window around the mask, at least WINDOW_SIZE, inside the image; floats in original pixels."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    W0, H0 = WINDOW_SIZE
    w = max((x1 - x0) * (1 + 2 * WINDOW_PAD), (y1 - y0) * (1 + 2 * WINDOW_PAD) * W0 / H0, W0)
    h = w * H0 / W0
    H, W = mask.shape
    if w > W or h > H:
        raise SystemExit(f'window {w:.0f}x{h:.0f} exceeds the {W}x{H} image')
    left = min(max((x0 + x1 - w) / 2, 0), W - w)
    top = min(max((y0 + y1 - h) / 2, 0), H - h)
    if w == W0:  # no scaling: keep the window on the pixel grid
        left, top = round(left), round(top)
    return [float(left), float(top), float(left + w), float(top + h)]


def window_intrinsics(cam_K, window):
    fx, _, cx, _, fy, cy = cam_K[:6]
    s = WINDOW_SIZE[0] / (window[2] - window[0])
    f = s * (fx + fy) / 2
    return [[f, 0.0, s * (cx - window[0])], [0.0, f, s * (cy - window[1])], [0.0, 0.0, 1.0]]


def grid_queries(mask, spacing):
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * MASK_ERODE + 1,) * 2, np.uint8)) > 0
    ys, xs = np.mgrid[spacing // 2:mask.shape[0]:spacing, spacing // 2:mask.shape[1]:spacing]
    keep = m[ys, xs]
    return np.stack([xs[keep], ys[keep]], 1).astype(np.float32)


def observed_in_mask(tracks, vis, masks):
    """visible and inside the frame's mask (T,P)."""
    T, H, W = masks.shape
    xy = np.round(tracks).astype(int)
    on = (xy[..., 0] >= 0) & (xy[..., 0] < W) & (xy[..., 1] >= 0) & (xy[..., 1] < H)
    inside = np.zeros_like(on)
    t_idx = np.broadcast_to(np.arange(T)[:, None], on.shape)
    inside[on] = masks[t_idx[on], xy[..., 1][on], xy[..., 0][on]]
    return vis & inside


def render(path, video, masks, tracks, observed, query_kf, fps):
    with imageio.get_writer(path, fps=fps, codec='libx264', quality=8, macro_block_size=8) as w:
        for t in range(len(video)):
            img = np.ascontiguousarray(video[t])
            contours, _ = cv2.findContours(masks[t].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
            for p in np.nonzero(observed[t])[0]:
                c = tuple(int(v) for v in VIS_COLORS[query_kf[p] % len(VIS_COLORS)])
                cv2.circle(img, (int(round(tracks[t, p, 0])), int(round(tracks[t, p, 1]))), 2, c, -1)
            cv2.putText(img, f'frame {t}  observed {observed[t].sum()}', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            w.append_data(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--object', type=int, required=True, help='BOP object id, e.g. 2')
    p.add_argument('--onboarding', default='dynamic', choices=['static', 'dynamic'])
    p.add_argument('--src', default=DEFAULT_SRC, help='BOP dataset root (containing onboarding_<onboarding>/)')
    p.add_argument('--out', default=None, help='default: <repo>/output/<dataset>')
    p.add_argument('--stride', type=int, default=1, help='use every stride-th frame of the video')
    p.add_argument('--grid', type=int, default=6, help='query spacing in px at 512x384')
    p.add_argument('--reseed', type=float, default=0.5,
                   help='new keyframe once fewer than this fraction of the current keyframe\'s points are observed')
    p.add_argument('--weights', default=os.path.join(REPO, 'checkpoints/cotracker3_scaled_offline.pth'))
    p.add_argument('--fps', type=int, default=15, help='of tracks.mp4')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    src = os.path.abspath(args.src)
    dataset = os.path.basename(os.path.normpath(src))
    out_root = os.path.abspath(args.out or os.path.join(REPO, 'output', dataset))
    obj = f'obj_{args.object:06d}'
    seq_name = f'{obj}_up' if args.onboarding == 'static' else obj
    seq = os.path.join(src, f'onboarding_{args.onboarding}', seq_name)
    out = os.path.join(out_root, args.onboarding, obj, 'video')
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(seq, 'scene_camera.json')) as fh:
        cams = json.load(fh)
    ids = sorted(int(k) for k in cams)[::args.stride]
    raw_masks = [np.array(Image.open(os.path.join(seq, 'mask_visib', f'{i:06d}_000000.png'))) > 0 for i in ids]
    if not all(m.any() for m in raw_masks):
        raise SystemExit(f'empty mask_visib in {seq}')
    window = crop_box(np.logical_or.reduce(raw_masks))
    video = np.stack([np.asarray(Image.open(os.path.join(seq, 'rgb', f'{i:06d}.jpg')).convert('RGB')
                                 .resize(WINDOW_SIZE, Image.LANCZOS, box=window)) for i in ids])  # T,H,W,3
    masks = np.stack([np.asarray(Image.fromarray(m).resize(WINDOW_SIZE, Image.NEAREST, box=window))
                      for m in raw_masks])  # T,H,W bool
    print(f'{seq_name}: {len(ids)} frames, window {[round(v) for v in window]} -> {WINDOW_SIZE}')

    model = CoTrackerPredictor(checkpoint=args.weights, offline=True).to(args.device)
    video_t = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(args.device)  # 1,T,3,H,W, 0..255

    t0 = time.perf_counter()
    tracks, observed, query_kf, query_frame, colors, keyframes = [], [], [], [], [], []
    kf = 0
    while kf is not None:
        q = grid_queries(masks[kf], args.grid)
        queries = torch.from_numpy(np.concatenate([np.full((len(q), 1), kf, np.float32), q], 1))[None]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):  # full precision OOMs on ~300 frames
            tr, vis = model(video_t, queries=queries.to(args.device), backward_tracking=True)
        tr, vis = tr[0].float().cpu().numpy(), vis[0].cpu().numpy()
        if vis.ndim == 3:
            vis = vis[..., 0]
        obs = observed_in_mask(tr, vis > 0.5 if vis.dtype != bool else vis, masks)
        obs[kf] = True  # the queries themselves
        tracks.append(tr)
        observed.append(obs)
        query_kf.append(np.full(len(q), len(keyframes)))
        query_frame.append(np.full(len(q), kf))
        colors.append(video[kf][q[:, 1].astype(int), q[:, 0].astype(int)])
        keyframes.append(kf)
        frac = obs.mean(1)
        later = np.nonzero(frac[kf + 1:] < args.reseed)[0]
        print(f'  keyframe {kf}: {len(q)} points, observed fraction min {frac.min():.2f}, '
              f'next keyframe {kf + 1 + later[0] if len(later) else "-"}')
        kf = int(kf + 1 + later[0]) if len(later) else None
    t_track = time.perf_counter() - t0

    tracks = np.concatenate(tracks, 1).astype(np.float32)
    observed = np.concatenate(observed, 1)
    query_kf, query_frame = np.concatenate(query_kf), np.concatenate(query_frame)
    colors = np.concatenate(colors).astype(np.uint8)
    np.savez_compressed(os.path.join(out, 'tracks.npz'), tracks=tracks, observed=observed, query_frame=query_frame,
                        colors=colors)
    K = window_intrinsics(cams[str(ids[0])]['cam_K'], window)
    with open(os.path.join(out, 'video.json'), 'w') as fh:
        json.dump(dict(dataset=dataset, onboarding=args.onboarding, sequence=seq_name, frame_ids=ids,
                       window=window, size=list(WINDOW_SIZE), K=K, keyframes=keyframes, grid=args.grid,
                       reseed=args.reseed, stride=args.stride, n_points=int(tracks.shape[1]),
                       time_tracking_s=round(t_track, 1)), fh, indent=1)
    render(os.path.join(out, 'tracks.mp4'), video, masks, tracks, observed, query_kf, args.fps)
    n_obs = observed.sum(0)
    print(f'{len(keyframes)} keyframes, {tracks.shape[1]} points, observations per point median {np.median(n_obs):.0f} '
          f'(of {len(ids)} frames), observed points per frame min {observed.sum(1).min()} | '
          f'tracking {t_track:.0f}s -> {out}')


if __name__ == '__main__':
    main()
