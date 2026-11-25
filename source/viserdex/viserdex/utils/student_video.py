"""Compose synchronized simulator, RGB inference and policy-observation frames."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


class StudentDashboard:
    def __init__(self, mode, iteration):
        self.mode = mode
        self.iteration = iteration
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        self.font = ImageFont.truetype(str(path), 18) if path.exists() else ImageFont.load_default()
        self.small = ImageFont.truetype(str(path), 15) if path.exists() else ImageFont.load_default()
        self.title = ImageFont.truetype(str(path), 25) if path.exists() else ImageFont.load_default()

    def frame(self, scene, masked_rgb, processed_rgb, prediction, truth, record):
        canvas = Image.new("RGB", (1280, 800), "#101820")
        d = ImageDraw.Draw(canvas)
        d.text(
            (20, 14),
            "Revo3 student | "
            + ("Hikrobot RGB pose closed loop" if self.mode == "vision" else "Simulated pose-noise baseline"),
            font=self.title,
            fill="#f1f5f9",
        )
        d.text(
            (20, 50),
            f"Student {self.iteration} | env {record['env_id']} | t={record['time_s']:.2f}s | "
            f"successes={record['successes']} | resets={record['resets']}",
            font=self.font,
            fill="#b6c5d4",
        )
        native_view = ImageOps.contain(Image.fromarray(scene[..., :3]), (640, 480))
        canvas.paste(native_view, (16 + (640 - native_view.width) // 2, 112))
        d.text((20, 86), "NATIVE CALIBRATED HIKROBOT RGB (same state)", font=self.small, fill="#b6c5d4")
        for x, title, pixels in (
            (672, "Rectified full frame (120x120)", masked_rgb),
            (968, "CNN input + predicted keypoints", processed_rgb),
        ):
            d.text((x, 88), title, font=self.small, fill="#b6c5d4")
            canvas.paste(Image.fromarray(pixels).resize((280, 280), Image.Resampling.NEAREST), (x, 112))
        for p in np.asarray(truth).reshape(9, 3):
            x, y = 968 + p[0] * 280, 112 + p[1] * 280
            if 968 <= x < 1248 and 112 <= y < 392:
                d.line((x - 3, y, x + 3, y), fill="#44ee88", width=2)
                d.line((x, y - 3, x, y + 3), fill="#44ee88", width=2)
        for p in np.asarray(prediction).reshape(9, 3):
            x, y = 968 + p[0] * 280, 112 + p[1] * 280
            if 968 <= x < 1248 and 112 <= y < 392:
                d.ellipse((x - 3, y - 3, x + 3, y + 3), outline="#ffb454", width=2)
        d.text(
            (672, 403), "Orange: prediction  |  Green: ground truth (diagnostic only)", font=self.small, fill="#b6c5d4"
        )
        d.text((672, 437), "ACTUAL STUDENT EXTEROCEPTIVE INPUT [11]", font=self.font, fill="#6ae0d0")
        e = record["extero"]
        rows = [
            f"position [mm]   {e[0] * 1000:+8.2f} {e[1] * 1000:+8.2f} {e[2] * 1000:+8.2f}",
            "quaternion xyzw  " + " ".join(f"{x:+.3f}" for x in e[3:7]),
            "relative quat    " + " ".join(f"{x:+.3f}" for x in e[7:11]),
            "Frame: right_hand_base_link | xyzw",
            f"CNN pose error: {record['cnn_translation_mm']:.2f} mm / {record['cnn_rotation_deg']:.2f} deg",
            f"Input pose error: {record['input_translation_mm']:.2f} mm / {record['input_rotation_deg']:.2f} deg",
            "Hand occlusion: native rendered geometry",
            f"CNN + pose decode: {record['inference_ms']:.1f} ms (batch)",
        ]
        for i, line in enumerate(rows):
            d.text((672, 471 + i * 29), line, font=self.small, fill="#e2e8f0")
        d.text((20, 612), "PROPRIOCEPTION [113] + recurrent memory", font=self.font, fill="#6ae0d0")
        d.text((20, 649), "21 noisy joints | 4 previous actions | goal | base | time", font=self.font, fill="#b6c5d4")
        d.text(
            (20, 681),
            f"Joint RMS noise/delay: {record['joint_error_rms']:.4f} (normalized)",
            font=self.font,
            fill="#b6c5d4",
        )
        d.text((20, 713), "Full policy inputs and actions saved in observations.npz", font=self.small, fill="#b6c5d4")
        path = (
            "Hikrobot RGB -> rectify -> frozen CNN -> base pose -> student -> simulation"
            if self.mode == "vision"
            else "Noisy simulated pose -> exported student -> simulation | CNN: monitor only"
        )
        d.text((20, 764), path, font=self.font, fill="#6ae0d0")
        return np.asarray(canvas)
