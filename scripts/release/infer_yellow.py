"""Infer camera-frame die pose from a synchronized RGB / visible-point NPZ."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/viserdex'))
import numpy as np
import torch
from viserdex.collectors.rgbd_fusion_network import RGBDKeypointNetwork
from viserdex.utils.pose_metrics import fit_rigid_pose


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/yellow_rgbd/yellow_fusion_latest.pt')
    parser.add_argument('--pair', type=Path, default=ROOT/'examples/yellow_rgbd/pair.npz')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = RGBDKeypointNetwork(payload['reference']).to(args.device).eval()
    model.load_state_dict(payload['model'], strict=True)
    with np.load(args.pair, allow_pickle=False) as pair:
        rgb = pair['rgb_clean']
        points = pair['points_camera_clean_m'] if 'points_camera_clean_m' in pair else pair['points_camera_m']
        uv, valid = pair['point_uv_rgb'], pair['point_valid'].astype(bool)
        if rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB [360,640,3]')
        if points.shape != (1024, 3) or uv.shape != (1024, 2) or valid.shape != (1024,):
            raise ValueError('Expected 1024 camera-frame points, pixel UV and validity mask')
        if not valid.any() or not np.isfinite(points).all() or not np.isfinite(uv).all():
            raise ValueError('Invalid point input')
        pred = model(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().to(args.device)/255,
                     torch.from_numpy(points.copy())[None].float().to(args.device),
                     torch.from_numpy(uv.copy())[None].float().to(args.device),
                     torch.from_numpy(valid.copy())[None].to(args.device))
        if not torch.isfinite(pred).all():
            raise ValueError('Nonfinite prediction')
        rotation, translation = fit_rigid_pose(model.reference.double(), pred.double())
        pose = np.eye(4)
        pose[:3,:3] = rotation[0].cpu().numpy()
        pose[:3,3] = translation[0].cpu().numpy()
        report = {'epoch':payload['epoch'], 'step':payload['step'],
                  'coordinate_frame':'optical camera: x right, y down, z forward; meters',
                  'T_camera_object':pose.tolist(), 'keypoints_camera_m':pred[0].cpu().tolist()}
        # Ground truth is used only after inference for optional evaluation.
        if 'T_camera_object' in pair:
            gt=pair['T_camera_object']
            report['translation_error_mm']=float(np.linalg.norm(pose[:3,3]-gt[:3,3])*1000)
            report['rotation_error_deg']=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@gt[:3,:3])-1)/2,-1,1))))
    output=json.dumps(report,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output+'\n')
    print(output)


if __name__ == '__main__':
    main()
