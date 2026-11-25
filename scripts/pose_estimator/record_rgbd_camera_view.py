"""Display saved RGB and measured points through the same RGB camera intrinsics."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists')
    source = json.loads((a.dataset/'manifest.json').read_text())
    assert source['completed'] and source['mode'] == 'policy_rollout_rgbd'
    files = sorted(a.dataset.glob('[0-9][0-9][0-9][0-9][0-9]/pair.npz'))
    mask = np.zeros((360, 640), dtype=bool)
    zmin, zmax = float('inf'), -float('inf')
    for path in files:
        with np.load(path) as d:
            mask |= d['rgb_visible']
            z = d['points_camera_m'][d['point_valid'], 2]
            zmin, zmax = min(zmin, float(z.min())), max(zmax, float(z.max()))
    yy, xx = np.nonzero(mask)
    side = int(max(xx.max()-xx.min(), yy.max()-yy.min())+32)
    cx, cy = int((xx.max()+xx.min())//2), int((yy.max()+yy.min())//2)
    box = (cx-side//2, cy-side//2, cx-side//2+side, cy-side//2+side)
    size = 540
    fps = float(source['capture_fps'])
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    fonts = {s: ImageFont.truetype(font_path, s) for s in (16, 20, 22, 28)}
    a.output.mkdir(parents=True)
    snapshots = []
    snapshot_ids = set(np.linspace(0, len(files)-1, 4, dtype=int).tolist())
    max_projection_error = 0.
    writer = imageio.get_writer(str(a.output/'rgb_pointcloud_camera_view.mp4'), fps=fps,
        codec='libx264', quality=8, macro_block_size=16,
        ffmpeg_params=['-pix_fmt','yuv420p','-threads','2','-movflags','+faststart'])
    scene_writer = imageio.get_writer(str(a.output/'policy_scene.mp4'), fps=fps,
        codec='libx264', quality=8, macro_block_size=16,
        ffmpeg_params=['-pix_fmt','yuv420p','-threads','2','-movflags','+faststart'])
    try:
        for i, path in enumerate(files):
            with np.load(path) as d:
                points = d['points_camera_m'][d['point_valid']]
                homogeneous = points @ d['K_rgb'].T
                uv = homogeneous[:, :2] / homogeneous[:, 2:]
                error = float(np.abs(uv-d['point_uv_rgb'][d['point_valid']]).max())
                assert error < 1e-4
                max_projection_error = max(max_projection_error, error)
                # Identical fixed crop and pixel-center resize convention on both panels.
                xy = (uv-np.asarray(box[:2])+.5)*(size/side)-.5
                colors = (colormaps['viridis']((points[:,2]-zmin)/(zmax-zmin))[:,:3]*255).astype('uint8')
                cloud = Image.new('RGB', (size, size), 'black')
                cd = ImageDraw.Draw(cloud)
                # Far points first, nearest point wins; dots only, no surface completion.
                for j in np.argsort(-points[:,2]):
                    x, y = xy[j]
                    cd.ellipse((x-1.5,y-1.5,x+1.5,y+1.5), fill=tuple(colors[j]))
                rgb = Image.fromarray(d['rgb']).crop(box).resize((size,size), Image.Resampling.NEAREST)
                canvas = Image.new('RGB', (1280,720), '#101720')
                canvas.paste(rgb, (50,108))
                canvas.paste(cloud, (690,108))
                draw = ImageDraw.Draw(canvas)
                draw.text((24,18), 'RGB + POINT CLOUD | SAME CAMERA VIEW', font=fonts[28], fill='white')
                draw.text((50,69), 'RGB: native ViserDex SH augmentation', font=fonts[22], fill='#d8e3ee')
                draw.text((690,69), f'Point cloud: {len(points)} visible points', font=fonts[22], fill='#d8e3ee')
                draw.line((640,65,640,650), fill='#3c4a59', width=2)
                draw.text((24,665), f"Frame {i:03d} | t={float(d['sim_time_s']):.2f}s | Policy {source['policy']['iteration']} | 1x speed", font=fonts[20], fill='white')
                draw.text((24,695), 'Same RGB camera intrinsics + same crop/zoom | Points colored by depth | No surface completion', font=fonts[16], fill='#b2c3d2')
                writer.append_data(np.asarray(canvas))
                scene = Image.new('RGB', (640,416), '#101720')
                with Image.open(path.parent/'scene.png') as source_scene:
                    scene.paste(source_scene.convert('RGB'), (0,56))
                sd = ImageDraw.Draw(scene)
                sd.text((12,6), f"CAMERA VIEW | Policy {source['policy']['iteration']} | 1x speed", font=fonts[20], fill='white')
                sd.text((12,32), f"Frame {i:03d} | t={float(d['sim_time_s']):.2f}s | Full RGB camera frame", font=fonts[16], fill='#b2c3d2')
                scene_writer.append_data(np.asarray(scene))
                if i in snapshot_ids:
                    canvas.save(a.output/f'frame_{i:03d}.png')
                    scene.save(a.output/f'scene_{i:03d}.png')
                    snapshots.append(canvas.resize((640,360)))
            if i%50 == 0:
                print(f'Encoded {i+1}/{len(files)}', flush=True)
    finally:
        writer.close()
        scene_writer.close()
    contact = Image.new('RGB', (1280,720))
    for i, frame in enumerate(snapshots):
        contact.paste(frame, ((i%2)*640, (i//2)*360))
    contact.save(a.output/'contact_sheet.png')
    reader = imageio.get_reader(str(a.output/'rgb_pointcloud_camera_view.mp4'))
    count = 0
    for frame in reader:
        assert frame.shape == (720,1280,3)
        count += 1
    reader.close()
    assert count == len(files)
    reader = imageio.get_reader(str(a.output/'policy_scene.mp4'))
    scene_count = 0
    for frame in reader:
        assert frame.shape == (416,640,3)
        scene_count += 1
    reader.close()
    assert scene_count == count
    result = dict(completed=True, source_dataset=str(a.dataset.resolve()), frames=count,
        fps=fps, duration_s=count/fps, camera='original RGB optical camera', projection='K_rgb @ points_camera_m',
        identical_crop_xyxy=box, identical_panel_size=[size,size], point_radius_display_px=1.5,
        max_projection_error_px=max_projection_error, depth_color_range_m=[zmin,zmax],
        source_data_modified=False, gpu_used=False, simulation_started=False,
        policy_iteration=source['policy']['iteration'], decoded_all_frames=True,
        calibration_sha256=source['capture']['calibration_sha256'], scene_decoded_frames=scene_count)
    (a.output/'manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
