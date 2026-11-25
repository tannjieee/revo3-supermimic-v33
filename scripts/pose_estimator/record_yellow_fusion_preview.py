"""CPU inference on a complete saved policy trajectory, with synchronized evidence video."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from viserdex.collectors.rgbd_fusion_network import RGBDKeypointNetwork
from viserdex.utils.pose_metrics import fit_rigid_pose


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--training_dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    os.nice(15)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    # Trainer replaces best.pt atomically. Copy one opened inode for a fixed evaluation.
    snapshot = args.output / 'checkpoint_snapshot.pt'
    shutil.copyfile(args.checkpoint, snapshot)
    payload = torch.load(snapshot, map_location='cpu', weights_only=True)
    source = json.loads((args.dataset / 'manifest.json').read_text())
    training = json.loads((args.training_dataset / 'manifest.json').read_text())
    assert source['completed'] and source['mode'] == 'policy_rollout_rgbd'
    assert source['capture']['calibration_sha256'] == training['calibration_sha256']
    assert source['policy']['checkpoint_sha256'] == training['policy_sha256']
    assert source['seed'] not in {s['seed'] for s in training['shards']}
    used_hashes = set()
    for split in ('train', 'validation'):
        index = args.training_dataset / f'{split}_index.jsonl'
        assert digest(index) == training['index_sha256'][split]
        used_hashes.update(json.loads(line)['pair_sha256'] for line in index.read_text().splitlines())
    paths = sorted(args.dataset.glob('[0-9][0-9][0-9][0-9][0-9]/pair.npz'))
    assert len(paths) == source['samples'] == 200
    source_hashes = {row['frame']: row['pair_sha256'] for row in source['frames']}
    frames = []
    mask = np.zeros((360, 640), dtype=bool)
    for i, path in enumerate(paths):
        h = digest(path)
        assert h == source_hashes[i] and h not in used_hashes
        with np.load(path, allow_pickle=False) as d:
            assert int(d['frame_id']) == i
            points = d['points_camera_m'].astype(np.float32)
            # This earlier continuous preview predates point-noise augmentation.
            assert not source['capture'].get('point_noise')
            ref = d['keypoints_object_m'].astype(np.float32)
            assert np.array_equal(ref, payload['reference'].numpy())
            valid = d['point_valid'].astype(bool)
            uv = d['point_uv_rgb'].astype(np.float32)
            projected = points @ d['K_rgb'].T
            assert np.max(np.abs(projected[valid, :2] / projected[valid, 2:] - uv[valid])) < 1e-4
            mask |= d['rgb_visible']
            frames.append(dict(rgb=d['rgb_clean'].copy(), play=d['play_raw_rgb'].copy(),
                               points=points, valid=valid, uv=uv, k=d['K_rgb'].copy(),
                               gt=d['T_camera_object'].copy(), target=d['keypoints_camera_m'].copy(),
                               time=float(d['sim_time_s']), sequence=int(d['sequence_id'])))
    times = np.array([f['time'] for f in frames])
    fps = float(source['capture_fps'])
    assert np.allclose(np.diff(times), 1 / fps)
    yy, xx = np.nonzero(mask)
    side = int(max(np.ptp(xx), np.ptp(yy)) + 48)
    cx, cy = (int(xx.min()) + int(xx.max())) // 2, (int(yy.min()) + int(yy.max())) // 2
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)
    model = RGBDKeypointNetwork(payload['reference']).eval()
    model.load_state_dict(payload['model'], strict=True)
    predictions = []
    start = time.monotonic()
    for i in range(0, len(frames), 2):
        batch = frames[i:i+2]
        rgb = torch.from_numpy(np.stack([f['rgb'] for f in batch])).permute(0, 3, 1, 2).float() / 255
        points = torch.from_numpy(np.stack([f['points'] for f in batch]))
        uv = torch.from_numpy(np.stack([f['uv'] for f in batch]))
        valid = torch.from_numpy(np.stack([f['valid'] for f in batch]))
        pred = model(rgb, points, uv, valid)
        assert torch.isfinite(pred).all()
        rotation, translation = fit_rigid_pose(model.reference.double(), pred.double())
        for j, frame in enumerate(batch):
            rot, trans = rotation[j].numpy(), translation[j].numpy()
            angle = float(np.degrees(np.arccos(np.clip((np.trace(rot.T @ frame['gt'][:3, :3]) - 1) / 2, -1, 1))))
            position = float(np.linalg.norm(trans - frame['gt'][:3, 3]) * 1000)
            keypoint = float(np.linalg.norm(pred[j].numpy() - frame['target'], axis=-1).mean() * 1000)
            predictions.append(dict(rotation=rot, translation=trans, keypoints=pred[j].numpy(),
                                    translation_mm=position, rotation_deg=angle, keypoint_mm=keypoint))
        if i % 20 == 0:
            print(f'INFERENCE {i+len(batch)}/{len(frames)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    np.savez_compressed(args.output / 'predictions.npz',
                        rotation=np.stack([p['rotation'] for p in predictions]),
                        translation_m=np.stack([p['translation'] for p in predictions]),
                        keypoints_camera_m=np.stack([p['keypoints'] for p in predictions]),
                        time_s=times, **{k: np.array([p[k] for p in predictions]) for k in
                                        ('translation_mm', 'rotation_deg', 'keypoint_mm')})
    fonts = {s: ImageFont.truetype(str(args.font), s) for s in (18, 20, 24, 28, 34)}
    ref = model.reference.numpy()
    edges = [(i, j) for i in range(1, 9) for j in range(i+1, 9)
             if np.count_nonzero(np.abs(ref[i] - ref[j]) > 1e-6) == 1]
    low = min(float(f['points'][f['valid'], 2].min()) for f in frames)
    high = max(float(f['points'][f['valid'], 2].max()) for f in frames)
    from matplotlib import colormaps

    def project(p, k):
        h = p @ k.T
        return h[:, :2] / h[:, 2:]

    def display_uv(uv):
        # Both projection and render use edge coordinates; no extra half-pixel shift.
        return (uv - np.array(box[:2])) * (400 / side) - 0.5

    def crop(rgb):
        return Image.fromarray(rgb).crop(box).resize((400, 400), Image.Resampling.NEAREST)

    def draw_pose(image, rotation, translation, k, color, width):
        draw = ImageDraw.Draw(image)
        uv = display_uv(project(ref @ rotation.T + translation, k))
        for i, j in edges:
            draw.line([tuple(uv[i]), tuple(uv[j])], fill=color, width=width)
        axes = np.vstack([np.zeros(3), np.eye(3) * 0.028]) @ rotation.T + translation
        a = display_uv(project(axes, k))
        for j in range(1, 4):
            draw.line([tuple(a[0]), tuple(a[j])], fill=color, width=width)
            draw.text(tuple(a[j]), 'XYZ'[j-1], font=fonts[18], fill=color)
        return image

    video = args.output / 'yellow_predictor_preview.mp4'
    snapshots = []
    selected = set(np.linspace(0, len(frames)-1, 4, dtype=int).tolist())
    writer = imageio.get_writer(str(video), fps=fps, codec='libx264', quality=8,
                                macro_block_size=16, ffmpeg_params=['-threads', '2', '-movflags', '+faststart'])
    try:
        for i, (frame, pred) in enumerate(zip(frames, predictions)):
            cloud = Image.new('RGB', (400, 400), 'black')
            draw = ImageDraw.Draw(cloud)
            pts = frame['points'][frame['valid']]
            uv = display_uv(project(pts, frame['k']))
            colors = (colormaps['viridis']((pts[:, 2] - low) / (high - low))[..., :3] * 255).astype('uint8')
            for j in np.argsort(-pts[:, 2]):
                x, y = uv[j]
                draw.ellipse((x-1.4, y-1.4, x+1.4, y+1.4), fill=tuple(colors[j]))
            overlay = crop(frame['play'])
            draw_pose(overlay, frame['gt'][:3, :3], frame['gt'][:3, 3], frame['k'], '#42e69b', 4)
            draw_pose(overlay, pred['rotation'], pred['translation'], frame['k'], '#ffab49', 2)
            panels = [Image.fromarray(frame['play']).resize((720, 405)), crop(frame['rgb']), cloud, overlay]
            titles = ['01  原始 Policy Play 回放', '02  实际 RGB 输入',
                      '03  可见点云：同相机视角', '04  预测位姿与真值叠加']
            notes = ['32100 轮 Actor 的连续轨迹；手、骰子、支架均保留',
                     '干净 RGB；手指和背景被掩码排除；输入为完整 640×360',
                     '1024 个采样点；不补全遮挡面；颜色表示深度',
                     '绿色：仿真真值   橙色：预测器输出   XYZ：物体语义轴']
            canvas = Image.new('RGB', (1600, 1200), '#0a1220')
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 16), '黄色骰子 RGB＋点云预测器 · 连续轨迹效果', font=fonts[34], fill='white')
            draw.text((24, 64), f'最佳权重 epoch {payload["epoch"]}  |  独立轨迹 seed {source["seed"]}  |  20 秒 / 10 fps 正常速度  |  CPU 离线推理',
                      font=fonts[24], fill='#adc3dc')
            for j, panel in enumerate(panels):
                x, y = 24 + (j % 2) * 788, 110 + (j // 2) * 494
                draw.rounded_rectangle((x, y, x+764, y+480), radius=12, fill='#172335', outline='#34475f')
                draw.text((x+16, y+10), titles[j], font=fonts[28], fill='white')
                canvas.paste(panel, (x + (764-panel.width)//2, y+49))
                draw.text((x+16, y+455), notes[j], font=fonts[18], fill='#b5c9de')
            draw.text((24, 1100), f'帧 {i+1:03d}/200   t={frame["time"]:05.2f}s   轨迹段 {frame["sequence"]+1}',
                      font=fonts[24], fill='white')
            draw.text((680, 1100), f'位置误差 {pred["translation_mm"]:.2f} mm     旋转误差 {pred["rotation_deg"]:.2f}°',
                      font=fonts[28], fill='#ffcf8b')
            draw.text((24, 1142), '02/03/04 使用同一固定裁剪放大；预测未反馈给 Actor；没有平滑、没有对称姿态取最小误差。',
                      font=fonts[20], fill='#aabed5')
            draw.text((24, 1171), '本段未进入训练/验证集；包含初始预热帧。仿真回放与虚拟深度，非真实相机或实时性能测试。',
                      font=fonts[18], fill='#93a9c1')
            writer.append_data(np.asarray(canvas))
            if i in selected:
                canvas.save(args.output / f'frame_{i:03d}.png')
                snapshots.append(canvas.resize((800, 600)))
    finally:
        writer.close()
    reader = imageio.get_reader(str(video))
    count = 0
    for image in reader:
        assert image.shape == (1200, 1600, 3)
        count += 1
    reader.close()
    assert count == len(frames)
    sheet = Image.new('RGB', (1600, 1200))
    for i, image in enumerate(snapshots):
        sheet.paste(image, ((i % 2)*800, (i // 2)*600))
    sheet.save(args.output / 'contact_sheet.png')

    def metrics(indices):
        return {k: {'mean': float(np.mean([predictions[i][k] for i in indices])),
                    'p95': float(np.quantile([predictions[i][k] for i in indices], 0.95))}
                for k in ('translation_mm', 'rotation_deg', 'keypoint_mm')}

    report = dict(completed=True, source_dataset=str(args.dataset.resolve()), source_seed=source['seed'],
                  source_manifest_sha256=digest(args.dataset/'manifest.json'),
                  excluded_from_train_and_validation=True, overlapping_pair_hashes=0,
                  checkpoint_source=str(args.checkpoint.resolve()), checkpoint_sha256=digest(snapshot),
                  epoch=payload['epoch'], optimizer_step=payload['step'], frames=count, fps=fps,
                  duration_s=count/fps, all_frames_decoded=True, video_sha256=digest(video),
                  input_rgb='rgb_clean', input_points='points_camera_m (source has no point noise)',
                  extra_augmentation=False, smoothing=False, symmetry_minimization=False,
                  roi_box_rgb=box, mean_and_p95_all_frames=metrics(range(count)),
                  mean_and_p95_after_initial_2s=metrics(np.flatnonzero(times >= 2)),
                  original_training_modified=False, device='cpu', elapsed_wall_s=time.monotonic()-start,
                  limitation='Offline simulation replay; not newly executed policy or real-camera/real-time benchmark')
    (args.output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('COMPLETED', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
