"""Show the production student camera input while an existing teacher moves the hand.

Run with the official Isaac Lab play CLI arguments, including --checkpoint.
This records RGB and preprocessing, and does not claim to evaluate a student CNN.
"""

import json
import os
from pathlib import Path
import sys

import gymnasium as gym
from gymnasium.envs.registration import load_env_creator
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

import viserdex.tasks  # noqa: F401
from isaaclab.utils import configclass
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from viserdex.distillation.hikrobot_images import HikrobotImageProcessor
from viserdex.renderer.hikrobot_calibration import (
    DEFAULT_CALIBRATION_DIRECTORY, audit_calibrated_camera, load_calibration,
)
from viserdex.tasks.manipulation.inhand.config.revo3_hand.revo3_env_cfg import ViserDexRevo3HikrobotEnvCfg
from viserdex.tasks.manipulation.inhand.pose_estimator.transforms import get_transforms

OUT = Path(os.environ["STUDENT_CAMERA_PREVIEW_OUT"])
OUT.mkdir(parents=True, exist_ok=True)


@configclass
class StudentCameraPreviewCfg(ViserDexRevo3HikrobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.video_recorders = [VideoRecorderCfg(
            source="sensor:hikrobot_camera:rgb", output_dir=str(OUT / "native_video"),
            fps=30, video_length=300, video_interval=0, frame_stride=1,
            output_filename_prefix="student_hikrobot_native",
        )]


class CameraInputRecorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        raw = env.unwrapped
        assert raw.num_envs == 1 and abs(raw.step_dt - 1 / 30) < 1e-12
        self.camera = raw.scene["hikrobot_camera"]
        self.camera.update(0., force_recompute=True)
        self.audit = audit_calibrated_camera(self.camera, raw.scene["robot"])
        previous = Path(os.environ["STUDENT_CAMERA_COMPARISON_AUDIT"])
        previous = json.loads(previous.read_text())
        np.testing.assert_allclose(self.audit["T_world_camera_env0"], previous["T_world_camera_actual"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(self.audit["intrinsics"], previous["intrinsics"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(self.audit["distortion"], previous["distortion"], atol=1e-12, rtol=0)
        intrinsics, _, _ = load_calibration(DEFAULT_CALIBRATION_DIRECTORY)
        self.processor = HikrobotImageProcessor(intrinsics, raw.device)
        self.preprocess = get_transforms(["binary_opening"], raw.device)
        self.writer = imageio.get_writer(str(OUT / "student_camera_input_preview.mp4"), fps=30,
                                        codec="libx264", pixelformat="yuv420p", quality=8, macro_block_size=16)
        self.steps = 0
        self.closed = False
        self.audit.update({"same_calibrated_view_as_previous_video": True,
                           "controller": "existing teacher, not a student policy",
                           "cnn_weights_used": False, "student_weights_used": False,
                           "image_processing": self.processor.metadata})
        (OUT / "camera_runtime_audit.json").write_text(json.dumps(self.audit, indent=2) + "\n")

    def step(self, action):
        with torch.inference_mode():
            native = self.camera.data.output["rgb"].torch[..., :3]
            rectified = self.processor(native)
            cnn_input = self.preprocess(rectified)
            assert cnn_input.shape == (1, 3, 120, 120) and torch.isfinite(cnn_input).all()
            pixels = native[0].cpu().numpy()
            rectified_pixels = (rectified[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
            cnn_pixels = (cnn_input[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
            canvas = Image.new("RGB", (1280, 704), "#101820")
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 12), "STUDENT CAMERA INPUT | Hikrobot DB0447679 | 30 Hz | teacher drives hand", fill="white")
            draw.text((16, 40), "Native calibrated RGB 1280x1024", fill="#b6c5d4")
            draw.text((832, 40), "Actual CNN input: rectified full frame", fill="#b6c5d4")
            canvas.paste(Image.fromarray(pixels).resize((800, 640)), (16, 64))
            canvas.paste(Image.fromarray(cnn_pixels).resize((416, 416), Image.Resampling.NEAREST), (832, 64))
            draw.text((832, 505), "120x120 | no crop | corrected lens distortion", fill="white")
            draw.text((832, 535), f"t={self.steps / 30:.2f}s | original calibrated viewpoint", fill="#6ae0d0")
            draw.text((832, 565), "Camera/input validation; no trained student/CNN", fill="#b6c5d4")
            self.writer.append_data(np.asarray(canvas))
            if self.steps in (0, 150, 299):
                Image.fromarray(pixels).save(OUT / f"native_{self.steps:04d}.png")
                Image.fromarray(rectified_pixels).save(OUT / f"rectified_{self.steps:04d}.png")
                Image.fromarray(cnn_pixels).save(OUT / f"cnn_input_{self.steps:04d}.png")
                canvas.save(OUT / f"preview_{self.steps:04d}.png")
            self.steps += 1
        return self.env.step(action)

    def close(self):
        if not self.closed:
            self.writer.close()
            self.closed = True
            (OUT / "camera_input_validation.json").write_text(json.dumps({
                "steps": self.steps, "duration_s": self.steps / 30,
                "passed": self.steps == 300, "controller": "teacher",
                "same_calibrated_view_as_previous_video": True,
                "student_cnn_closed_loop_validated": False,
            }, indent=2) + "\n")
        return self.env.close()


base = gym.spec("Isaac-ViserDex-Repose-Revo3-Hikrobot")
creator = load_env_creator(base.entry_point) if isinstance(base.entry_point, str) else base.entry_point


def create_recorded_env(**kwargs):
    return CameraInputRecorder(creator(**kwargs))


gym.register(
    id="Isaac-ViserDex-Repose-Revo3-Hikrobot-Input-Preview", entry_point=create_recorded_env,
    disable_env_checker=True, kwargs={**base.kwargs, "env_cfg_entry_point": StudentCameraPreviewCfg},
)

from isaaclab_rl.entrypoints import run_play_cli

if __name__ == "__main__":
    raise SystemExit(run_play_cli(sys.argv[1:]))
