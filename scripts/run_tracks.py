"""Structure from motion from the CoTracker tracks of track_video.py, and show the result next to the BOP GT in viser.

No depth and no pose network: the tracks go into a COLMAP database (per frame the observed track positions as
keypoints; per frame pair the shared observed tracks as matches, verified by RANSAC on the essential matrix with the
known K; pairs: each frame with its next --pair_gap frames plus all pairs among every --loop_every-th frame, since
matching all pairs made the mapping several minutes slower without new information: CoTracker's tracks are ~98%
RANSAC inliers and COLMAP links the observations of a track across the neighbour pairs) and COLMAP's incremental mapper (pycolmap) reconstructs poses and points with K fixed (one shared PINHOLE
camera, focal and principal point not refined). Since only points inside mask_visib are observed, the rigid object
is the static scene and the camera moves around it.

--src is the video folder of track_video.py (video.json + tracks.npz). Writes <src>/../sfm/ (or --out):
  images/       the window frames (512x384), for colours and the viewer
  database.db   the COLMAP database built from the tracks; model/ the reconstruction (COLMAP binary)
  cameras.json  per registered frame: frame_id, image, cam2world (OpenCV, arbitrary scale), K, focal, mask_depth
                (median depth of the frame's observed points, for anchoring the dynamic viewer)
  points.npz    triangulated points (N,3) float32 and colors (N,3) uint8
  run.json      settings, registered frames, points, observations, reprojection error, timings
SfM is skipped if cameras.json, points.npz and run.json exist (use --force). The viewer (http://localhost:8080, or
the next free port) shows every --viz_every-th registered camera, the GT cameras and the point cloud in the object
frame in metres; stop it with Ctrl+C. Static onboarding is Sim(3)-aligned to all GT cameras; dynamic onboarding only
has GT for frame 0, so the reconstruction is anchored there: rotation and position from the frame-0 GT pose,
scale = GT object distance / mask_depth of frame 0.

Usage:
  .venv/bin/python scripts/run_tracks.py --src output/hope/dynamic/obj_000002/video [--force] [--no-viz]
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import PIL.Image
import pycolmap
import viser
import viser.transforms as vtf

DEFAULT_SRC = '/home/vincent/2_resources/bop_h3/hope'
VIZ_MAX_POINTS = 500_000
VIZ_IMAGE_WIDTH = 256
EST_COLOR, GT_COLOR, LINE_COLOR = (255, 120, 0), (0, 150, 255), (220, 30, 30)


def write_frames(video, seq, images_dir):
    os.makedirs(images_dir, exist_ok=True)
    names = []
    for fid in video['frame_ids']:
        name = f'{fid:06d}.jpg'
        path = os.path.join(images_dir, name)
        if not os.path.isfile(path):
            PIL.Image.open(os.path.join(seq, 'rgb', name)).convert('RGB').resize(
                tuple(video['size']), PIL.Image.LANCZOS, box=tuple(video['window'])).save(path, quality=95)
        names.append(name)
    return names


def frame_pairs(n, gap, loop_every):
    """Neighbours up to gap frames apart, plus all pairs among every loop_every-th frame (long baselines)."""
    pairs = {(i, j) for i in range(n) for j in range(i + 1, min(i + gap + 1, n))}
    loop = range(0, n, loop_every)
    pairs |= {(i, j) for i in loop for j in loop if i < j}
    return sorted(pairs)


def build_database(path, video, names, tracks, observed, min_matches, pairs):
    """Tracks -> COLMAP database: keypoints per frame, matches + verified two-view geometry per frame pair."""
    if os.path.exists(path):
        os.remove(path)
    W, H = video['size']
    K = np.array(video['K'])
    camera = pycolmap.Camera(model='PINHOLE', width=W, height=H, params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                             has_prior_focal_length=True)
    db = pycolmap.Database(path)
    camera.camera_id = db.write_camera(camera)
    image_ids, kp_index = [], []
    for t, name in enumerate(names):
        image_ids.append(db.write_image(pycolmap.Image(name=name, camera_id=camera.camera_id)))
        idx = np.full(tracks.shape[1], -1)
        idx[observed[t]] = np.arange(observed[t].sum())
        kp_index.append(idx)
        db.write_keypoints(image_ids[-1], tracks[t, observed[t]].astype(np.float32))
    options = pycolmap.TwoViewGeometryOptions()
    n_pairs = n_verified = 0
    for i, j in pairs:
        shared = observed[i] & observed[j]
        if shared.sum() < min_matches:
            continue
        matches = np.stack([kp_index[i][shared], kp_index[j][shared]], 1).astype(np.uint32)
        db.write_matches(image_ids[i], image_ids[j], matches)
        geometry = pycolmap.estimate_calibrated_two_view_geometry(
            camera, tracks[i, observed[i]].astype(np.float64), camera, tracks[j, observed[j]].astype(np.float64),
            matches, options)
        db.write_two_view_geometry(image_ids[i], image_ids[j], geometry)
        n_pairs += 1
        n_verified += len(geometry.inlier_matches) >= min_matches
    db.close()
    return n_pairs, n_verified


def run_sfm(args, video, src, out):
    seq = os.path.join(args.data, f'onboarding_{video["onboarding"]}', video['sequence'])
    tr = np.load(os.path.join(src, 'tracks.npz'))
    tracks, observed = tr['tracks'], tr['observed']
    images_dir = os.path.join(out, 'images')
    names = write_frames(video, seq, images_dir)

    t0 = time.perf_counter()
    db_path = os.path.join(out, 'database.db')
    pairs = frame_pairs(len(names), args.pair_gap, args.loop_every)
    n_pairs, n_verified = build_database(db_path, video, names, tracks, observed, args.min_matches, pairs)
    t_db = time.perf_counter() - t0

    t0 = time.perf_counter()
    options = pycolmap.IncrementalPipelineOptions()
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    # the tracks are clean enough that COLMAP's frequent global BAs only cost time: 130s -> 20s for 138 frames,
    # same registration, reprojection error and point count
    options.ba_global_images_ratio = options.ba_global_points_ratio = 1.5
    options.ba_global_max_num_iterations, options.ba_local_max_num_iterations = 20, 10
    options.ba_global_max_refinements = options.ba_local_max_refinements = 1
    model_dir = os.path.join(out, 'model')
    shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir)
    models = pycolmap.incremental_mapping(db_path, images_dir, model_dir, options)
    t_sfm = time.perf_counter() - t0
    if not models:
        raise SystemExit('incremental mapping found no reconstruction')
    rec = max(models.values(), key=lambda r: r.num_reg_images())
    rec.write(model_dir)

    name_to_t = {n: t for t, n in enumerate(names)}
    ids = np.array(sorted(rec.points3D))
    xyz = np.array([rec.points3D[j].xyz for j in ids])
    cameras = []
    for image in sorted(rec.images.values(), key=lambda im: name_to_t[im.name]):
        T = np.eye(4)
        T[:3] = image.cam_from_world.matrix()
        seen = [p.point3D_id for p in image.points2D if p.has_point3D()]
        z = (xyz[np.searchsorted(ids, seen)] @ T[:3, :3].T + T[:3, 3])[:, 2]
        fid = video['frame_ids'][name_to_t[image.name]]
        cameras.append(dict(frame_id=fid, image=os.path.join(images_dir, image.name), cam2world=np.linalg.inv(T).tolist(),
                            size=video['size'], K=video['K'], focal=video['K'][1][1], mask_depth=float(np.median(z))))
    points = xyz.astype(np.float32)
    colors = np.array([rec.points3D[j].color for j in ids], dtype=np.uint8)

    with open(os.path.join(out, 'cameras.json'), 'w') as fh:
        json.dump(cameras, fh, indent=1)
    np.savez_compressed(os.path.join(out, 'points.npz'), points=points, colors=colors)
    missing = [video['frame_ids'][name_to_t[n]] for n in names if n not in {im.name for im in rec.images.values()}]
    run = dict(vars(args), src=src, out=out, n_frames=len(names), n_tracks=int(tracks.shape[1]), n_pairs=n_pairs,
               n_verified_pairs=int(n_verified), n_models=len(models), n_registered=rec.num_reg_images(),
               unregistered_frames=missing, n_points=len(points),
               n_observations=int(sum(rec.points3D[j].track.length() for j in ids)),
               reproj_error_px=round(rec.compute_mean_reprojection_error(), 3),
               time_database_s=round(t_db, 1), time_sfm_s=round(t_sfm, 1))
    with open(os.path.join(out, 'run.json'), 'w') as fh:
        json.dump(run, fh, indent=1)
    print(f'{len(names)} frames, {n_pairs} pairs ({n_verified} verified) -> {len(models)} model(s); largest: '
          f'{rec.num_reg_images()} registered, {len(points)} points, {run["n_observations"]} observations, '
          f'reprojection error {run["reproj_error_px"]:.2f} px | database {t_db:.0f}s, mapping {t_sfm:.0f}s -> {out}')
    if missing:
        print(f'unregistered frames: {missing}')
    return cameras, points, colors


def gt_cam2world(frame):
    """BOP model-to-camera pose (mm) -> camera-to-object pose (m)."""
    R = np.array(frame['cam_R_m2c']).reshape(3, 3)
    t = np.array(frame['cam_t_m2c']) / 1000
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.T, -R.T @ t
    return T


def umeyama(X, Y):
    """Similarity (s, R, t) minimizing |Y - (s R X + t)|^2."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (Xc ** 2).sum(1).mean()
    return s, R, my - s * R @ mx


def visualize(meta, cameras, points, colors, viz_every):
    est = [np.array(c['cam2world']) for c in cameras]
    gt = {i: gt_cam2world(f) for i, f in enumerate(meta['frames']) if f['cam_R_m2c'] is not None}
    if not gt:
        raise SystemExit('no registered frame has a GT pose (dynamic: frame 0 was not registered)')
    if meta['onboarding'] == 'static':
        s, R, t = umeyama(np.array([est[i][:3, 3] for i in gt]), np.array([g[:3, 3] for g in gt.values()]))
        alignment = f'Sim(3) fit to {len(gt)} GT cameras'
    else:  # anchor: estimated camera i0 := GT camera i0, scale from the object distance
        i0 = min(gt)
        s = np.linalg.norm(meta['frames'][i0]['cam_t_m2c']) / 1000 / cameras[i0]['mask_depth']
        R = gt[i0][:3, :3] @ est[i0][:3, :3].T
        t = gt[i0][:3, 3] - s * R @ est[i0][:3, 3]
        alignment = f'anchored at GT frame {meta["frames"][i0]["frame_id"]}'
    est_aligned = []
    for e in est:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R @ e[:3, :3], s * R @ e[:3, 3] + t
        est_aligned.append(T)
    points = s * points @ R.T + t

    n_total = len(points)
    if n_total > VIZ_MAX_POINTS:
        idx = np.random.default_rng(0).choice(n_total, VIZ_MAX_POINTS, replace=False)
        points, colors = points[idx], colors[idx]

    server = viser.ViserServer()
    up = -np.mean([g[:3, 1] for g in gt.values()], axis=0)  # OpenCV cameras: -y is up
    server.scene.set_up_direction(up / np.linalg.norm(up))
    radius = float(np.median([np.linalg.norm(g[:3, 3]) for g in gt.values()]))
    frustum_scale = 0.12 * radius

    server.scene.add_frame('/object', axes_length=0.15 * radius, axes_radius=0.004 * radius)
    groups = {'estimated cameras': [], 'GT cameras': []}
    if len(gt) > 1:
        groups['error lines'] = []
    shown = set(range(0, len(cameras), viz_every)) | set(gt)
    for i, (c, f) in enumerate(zip(cameras, meta['frames'])):
        if i not in shown:
            continue
        img = PIL.Image.open(f['image']).convert('RGB')
        img = np.asarray(img.resize((VIZ_IMAGE_WIDTH, round(VIZ_IMAGE_WIDTH * img.height / img.width))))
        w, h = c['size']
        groups['estimated cameras'].append(server.scene.add_camera_frustum(
            f'/est/{i}', fov=2 * np.arctan(h / (2 * c['focal'])), aspect=w / h, scale=frustum_scale,
            color=EST_COLOR, image=img, wxyz=vtf.SO3.from_matrix(est_aligned[i][:3, :3]).wxyz,
            position=est_aligned[i][:3, 3]))
        if i not in gt:
            continue
        W, H = meta['size']
        groups['GT cameras'].append(server.scene.add_camera_frustum(
            f'/gt/{i}', fov=2 * np.arctan(H / (2 * f['K'][1][1])), aspect=W / H,
            scale=frustum_scale, color=GT_COLOR, wxyz=vtf.SO3.from_matrix(gt[i][:3, :3]).wxyz,
            position=gt[i][:3, 3]))
        if 'error lines' in groups:
            groups['error lines'].append(server.scene.add_spline_catmull_rom(
                f'/lines/{i}', np.stack([est_aligned[i][:3, 3], gt[i][:3, 3]]), line_width=2, color=LINE_COLOR))

    server.gui.add_markdown(
        f"**{meta['onboarding']} · {meta['sequence']}** · {meta['n']} registered frames "
        f"(every {viz_every}th shown)\n\n"
        f"{alignment} · scale {s:.4f} m per SfM unit\n\n"
        f"points: {n_total:,} total, {len(points):,} shown\n\n"
        f"orange = estimated, blue = GT")
    show_cloud = server.gui.add_checkbox('point cloud', True)
    for label, handles in groups.items():
        box = server.gui.add_checkbox(label, True)
        box.on_update(lambda _, b=box, hs=handles: [setattr(h, 'visible', b.value) for h in hs])
    size = server.gui.add_slider('point size (m)', 0.0005, 0.02, 0.0005, round(0.004 * radius / 0.0005) * 0.0005)

    def draw_cloud(_=None):  # viser 0.2.7 can't change point size in place, so re-add under the same name
        server.scene.add_point_cloud('/points', points, colors, point_size=size.value, visible=show_cloud.value)

    draw_cloud()
    show_cloud.on_update(draw_cloud)
    size.on_update(draw_cloud)

    print(f'viewer at http://localhost:{server.get_port()} (Ctrl+C to stop)')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='video folder of track_video.py, e.g. output/hope/dynamic/obj_000002/video')
    p.add_argument('--out', default=None, help='default: <src>/../sfm')
    p.add_argument('--data', default=DEFAULT_SRC, help='BOP dataset root (for the frames and the GT)')
    p.add_argument('--min_matches', type=int, default=15, help='shared observed tracks needed to match a frame pair')
    p.add_argument('--pair_gap', type=int, default=8, help='match each frame with the next pair_gap frames')
    p.add_argument('--loop_every', type=int, default=10,
                   help='additionally match all pairs among every loop_every-th frame (long baselines, loop closure)')
    p.add_argument('--viz_every', type=int, default=10, help='show every n-th registered camera in the viewer')
    p.add_argument('--force', action='store_true', help='regenerate even if results already exist in --out')
    p.add_argument('--no-viz', action='store_true', help='skip the viser viewer (e.g. for batch runs)')
    args = p.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out or os.path.join(src, '..', 'sfm'))
    with open(os.path.join(src, 'video.json')) as fh:
        video = json.load(fh)
    os.makedirs(out, exist_ok=True)

    if not args.force and all(os.path.isfile(os.path.join(out, f)) for f in ('run.json', 'cameras.json', 'points.npz')):
        with open(os.path.join(out, 'cameras.json')) as fh:
            cameras = json.load(fh)
        cloud = np.load(os.path.join(out, 'points.npz'))
        points, colors = cloud['points'], cloud['colors']
        print(f'results already in {out}, skipping SfM; use --force to regenerate')
    else:
        cameras, points, colors = run_sfm(args, video, src, out)

    if not args.no_viz:
        seq = os.path.join(args.data, f'onboarding_{video["onboarding"]}', video['sequence'])
        with open(os.path.join(seq, 'scene_gt.json')) as fh:
            gts = json.load(fh)
        frames = []
        for c in cameras:
            gt = gts.get(str(c['frame_id']), [{}])[0]
            frames.append(dict(frame_id=c['frame_id'], image=c['image'], K=c['K'],
                               cam_R_m2c=gt.get('cam_R_m2c'), cam_t_m2c=gt.get('cam_t_m2c')))
        meta = dict(onboarding=video['onboarding'], sequence=video['sequence'], n=len(cameras), size=video['size'],
                    frames=frames)
        visualize(meta, cameras, points, colors, args.viz_every)


if __name__ == '__main__':
    main()
