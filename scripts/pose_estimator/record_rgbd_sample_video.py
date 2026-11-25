"""Record synchronized saved RGB/visible-point samples with native SH augmentation.

This re-renders existing poses and preserves measured points/labels. It does
not start physics, a policy or an optimizer. Video pauses and zooms are display
only, recorded in the manifest.
"""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/viserdex"))
from viserdex.renderer.splat_randomization import SplatRandomizationCfg, ViserDexSplatRandomizer, render_splats, digest
from viserdex.utils.load_utils import load_splat_file


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--saved_rgb", action="store_true", help="CPU playback of existing RGB without re-rendering")
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite output")
    torch.set_num_threads(2)
    paths = sorted(args.dataset.glob("[0-9][0-9][0-9][0-9][0-9]/pair.npz"))
    if not paths:
        p.error("No captured samples found")
    source_manifest = json.loads((args.dataset / "manifest.json").read_text())
    policy_play = source_manifest.get("mode") == "policy_rollout_rgbd"
    fps = float(source_manifest.get("capture_fps", 10))
    repeats = 1 if policy_play else 5
    sh_enabled = bool(source_manifest.get("capture", {}).get("splat_augmentation")) or not args.saved_rgb
    records = []
    for path in paths:
        with np.load(path) as data:
            records.append({key:data[key].copy() for key in data.files})
    sequence_count = len({int(data["sequence_id"]) for data in records})
    snapshot_ids = set(np.linspace(0, len(records)-1, 4, dtype=int).tolist())
    clouds = [data["points_camera_m"][data["point_valid"]].astype(float)*1000 for data in records]
    all_points = np.concatenate(clouds)
    lower, upper = all_points.min(0), all_points.max(0)
    center, span = (lower+upper)/2, float((upper-lower).max())*1.15
    lower, upper = center-span/2, center+span/2
    depth_range = (float(all_points[:,2].min()), float(all_points[:,2].max()))
    # A fixed ROI for all frames avoids introducing artificial display motion.
    all_masks = np.logical_or.reduce([data["rgb_visible"] for data in records])
    yy, xx = np.nonzero(all_masks)
    side = int(max(xx.max()-xx.min(), yy.max()-yy.min())+32)
    cx, cy = int((xx.max()+xx.min())//2), int((yy.max()+yy.min())//2)
    box = (cx-side//2, cy-side//2, cx-side//2+side, cy-side//2+side)
    ply = ROOT / "assets/rounded_die/gaussians.ply"
    splats = {} if args.saved_rgb else load_splat_file(str(ply), device="cuda:0")
    baseline = {key:value.clone() for key,value in splats.items()}
    augmenter = None if args.saved_rgb else ViserDexSplatRandomizer(splats, SplatRandomizationCfg(), ply)
    args.output.mkdir(parents=True)
    manifest = dict(completed=False, training_ready=False,
                    mode=("original_saved_RGB_playback" if args.saved_rgb else "saved_zero_action_samples_re_rendered_with_native_SH"),
                    source_dataset=str(args.dataset.resolve()), native_augmentation=augmenter.contract if augmenter else source_manifest.get("capture", {}).get("native_augmentation"),
                    policy=source_manifest.get("policy"),
                    point_source="unchanged captured first-hit visible points; no completion",
                    playback_fps=fps, display_repeats_per_sample=repeats, slowdown_factor=repeats,
                    reset_card_duration_s=.5, rgb_display_crop=box,
                    cloud_display="fixed camera-coordinate axes, millimeters; color is optical depth",
                    frames=[])
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    writer = imageio.get_writer(str(args.output/"rgb_pointcloud.mp4"), fps=fps, codec="libx264",
                               quality=8, macro_block_size=16, ffmpeg_params=["-pix_fmt","yuv420p","-threads","2","-movflags","+faststart"])
    scene_writer = (imageio.get_writer(str(args.output/"policy_scene.mp4"),fps=fps,codec="libx264",
                    quality=8,macro_block_size=16,ffmpeg_params=["-pix_fmt","yuv420p","-threads","2","-movflags","+faststart"])
                    if policy_play else None)
    video_frames = 0
    snapshots=[]
    try:
        for index,(path,data,cloud) in enumerate(zip(paths,records,clouds,strict=True)):
            seq=int(data["sequence_id"])
            if index and seq != int(records[index-1]["sequence_id"]):
                card=Image.new("RGB",(1280,720),"#111820")
                cd=ImageDraw.Draw(card)
                cd.text((310,290),f"RESET - sequence {seq+1}",font=font(36),fill="white")
                cd.text((245,355),("Environment reset during policy play" if policy_play else "Independent short capture; not continuous motion"),font=font(24),fill="#b8c8d8")
                for _ in range(round(.5*fps)):
                    writer.append_data(np.asarray(card)); video_frames+=1
                    if scene_writer is not None: scene_writer.append_data(np.asarray(card.resize((640,416))))
            source_hash=digest(path)
            if args.saved_rgb:
                rgb=data["rgb"].copy()
                aug=None
            else:
                original_rng=torch.cuda.get_rng_state().clone()
                varied,aug=augmenter(20261102+index)
                assert torch.equal(original_rng,torch.cuda.get_rng_state())
                for key in splats:
                    assert torch.equal(splats[key],baseline[key]),key
                    if key not in ["sh0","shN"]: assert torch.equal(varied[key],baseline[key]),key
                view=torch.tensor(data["T_camera_object"],dtype=torch.float32,device="cuda:0")[None]
                k=torch.tensor(data["K_rgb"],dtype=torch.float32,device="cuda:0")[None]
                mask=torch.tensor(data["rgb_visible"],device="cuda:0",dtype=torch.bool)[None]
                rendered=render_splats(varied,view,k)
                assert torch.isfinite(rendered).all()
                rgb=(rendered*mask[...,None]).mul(255).round().byte()[0].cpu().numpy()
            assert not rgb[~data["rgb_visible"]].any()
            folder=args.output/f"{index:05d}"
            folder.mkdir()
            Image.fromarray(rgb).save(folder/"rgb.png")
            copied=dict(data,rgb=rgb,rgb_clean=data.get("rgb_clean",data["rgb"]))
            np.savez_compressed(folder/"pair.npz",**copied)
            with np.load(folder/"pair.npz") as saved:
                for key in data:
                    if key != "rgb": assert np.array_equal(saved[key],data[key],equal_nan=True),key
            assert digest(path)==source_hash
            row=dict(frame=index,sequence=seq,episode_time_s=float(data["episode_time_s"]),
                     valid_points=len(cloud),source_pair_sha256=source_hash,
                     outside_mask_nonblack_pixels=0,points_and_labels_unchanged=True,augmentation=aug)
            (folder/"metadata.json").write_text(json.dumps(row,indent=2))
            manifest["frames"].append(row)
            fig=plt.figure(figsize=(6.4,5.5),dpi=100,facecolor="#141d27")
            ax=fig.add_subplot(111,projection="3d",facecolor="#141d27")
            scatter=ax.scatter(cloud[:,0],cloud[:,1],cloud[:,2],c=cloud[:,2],s=6,
                               cmap="viridis",vmin=depth_range[0],vmax=depth_range[1],depthshade=False)
            for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
                axis.pane.fill=False
                axis.label.set_color("#d8e3ee")
                axis.line.set_color("#637589")
                axis.set_tick_params(colors="#bac8d5",labelsize=8)
            ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),zlim=(lower[2],upper[2]),
                   xlabel="X right (mm)",ylabel="Y down (mm)",zlabel="Z forward (mm)")
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=22,azim=-65)
            fig.subplots_adjust(left=.02,right=.85,bottom=.07,top=.98)
            bar=fig.colorbar(scatter,ax=ax,shrink=.55,pad=.06)
            bar.ax.tick_params(colors="#bac8d5",labelsize=8)
            bar.set_label("Depth (mm)",color="#d8e3ee")
            fig.canvas.draw()
            cloud_image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
            plt.close(fig)
            canvas=Image.new("RGB",(1280,720),"#101720")
            canvas.paste(Image.fromarray(rgb).crop(box).resize((500,500),Image.Resampling.NEAREST),(52,95))
            canvas.paste(cloud_image,(640,90))
            inset=(Image.open(path.parent/"scene.png").convert("RGB") if policy_play else Image.fromarray(rgb)).resize((192,108))
            canvas.paste(inset,(438,540))
            draw=ImageDraw.Draw(canvas)
            draw.line((639,70,639,655),fill="#3c4a59",width=2)
            draw.text((24,18),"RGB + VISIBLE POINT CLOUD",font=font(28),fill="white")
            draw.text((24,62),("RGB: native ViserDex SH augmentation" if sh_enabled else "RGB: original capture (SH augmentation off)"),font=font(21),fill="#d8e3ee")
            draw.text((668,62),f"Captured point cloud: {len(cloud)} valid points",font=font(21),fill="#d8e3ee")
            draw.text((24,603),("RGB ROI zoom | inset: policy scene" if policy_play else "Fixed ROI zoom | inset: full 640 x 360 RGB"),font=font(17),fill="#b2c3d2")
            draw.text((24,665),f"Sequence {seq+1}/{sequence_count}   Frame {index:03d}   t={float(data['sim_time_s']):.2f}s   " + ("Policy play / 1x speed" if policy_play else "Playback 5x slower"),font=font(20),fill="white")
            draw.text((24,695),("Policy checkpoint " + str(source_manifest["policy"]["iteration"]) + " | same-frame RGB / points | depth-colored points; no surface completion" if policy_play else "Saved zero-action preview | same-frame RGB / points | depth-colored points; no surface completion"),font=font(14),fill="#9bafc2")
            if index in snapshot_ids:
                canvas.save(folder/"video_frame.png"); snapshots.append(canvas.resize((640,360)))
            for _ in range(repeats): writer.append_data(np.asarray(canvas)); video_frames+=1
            if scene_writer is not None:
                full_scene=Image.new("RGB",(640,416),"#101720")
                full_scene.paste(Image.open(path.parent/"scene.png").convert("RGB"),(0,56))
                sd=ImageDraw.Draw(full_scene)
                sd.text((12,6),f"POLICY PLAY | checkpoint {source_manifest['policy']['iteration']} | 1x speed",font=font(19),fill="white")
                sd.text((12,31),f"t={float(data['sim_time_s']):.2f}s | frame {index:03d} | RGB-D captured in this state",font=font(16),fill="#b2c3d2")
                for _ in range(repeats): scene_writer.append_data(np.asarray(full_scene))
            print(json.dumps(dict(frame=index,valid_points=len(cloud),sequence=seq)),flush=True)
    finally:
        writer.close()
        if scene_writer is not None: scene_writer.close()
    manifest.update(completed=True,video_frames=video_frames,duration_s=video_frames/fps,
                    limitation=("Short actor-only policy rollout; not a complete training dataset" if policy_play else "Saved zero-action diagnostic samples replayed slowly"))
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    contact=Image.new("RGB",(1280,720))
    for i,snapshot in enumerate(snapshots): contact.paste(snapshot,((i%2)*640,(i//2)*360))
    contact.save(args.output/"contact_sheet.png")
    # Decode the entire delivery video to confirm encoding, size and frame count.
    reader=imageio.get_reader(str(args.output/"rgb_pointcloud.mp4"))
    count=0
    for decoded in reader:
        assert decoded.shape==(720,1280,3),decoded.shape
        count+=1
    reader.close()
    assert count==video_frames,(count,video_frames)
    (args.output/"video_checks.json").write_text(json.dumps(dict(passed=True,decoded_frames=count,
        size=[1280,720],fps=fps,duration_s=count/fps,all_points_and_labels_preserved=True,
        outside_mask_nonblack_pixels=0),indent=2))


if __name__=="__main__": main()
