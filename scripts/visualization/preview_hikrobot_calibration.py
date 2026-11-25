"""Play the current teacher with the delivered calibrated Newton camera."""

from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path('/home/zhangke/viserDex_revo3')
OUT = Path(os.environ.get('HIKROBOT_PREVIEW_OUT', str(ROOT / 'outputs/hikrobot_calibrated_preview_20261022')))
selection = json.loads((OUT / 'preview_selection.json').read_text())

import pyglet
pyglet.options['headless'] = True
pyglet.options['headless_device'] = 0

import gymnasium as gym
from gymnasium.envs.registration import load_env_creator
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warp as wp
from PIL import Image
import cv2

import viserdex.tasks
from viserdex.renderer.hikrobot_calibration import (
    load_calibration, make_calibrated_camera,
)
from viserdex.tasks.manipulation.inhand.config.revo3_hand.revo3_env_cfg import ViserDexRevo3EnvCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg


@configclass
class HikrobotPreviewEnvCfg(ViserDexRevo3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.robot.spawn.usd_path = selection['robot_usd']
        self.scene.object.spawn.usd_path = selection['object_usd']
        self.decimation = 4
        self.sim.dt = 1 / 120.
        self.sim.render_interval = 4
        self.commands.object_pose.debug_vis = False
        initial = self.scene.robot.init_state
        self.scene.hikrobot_camera = make_calibrated_camera(
            selection['calibration_directory'], initial.pos, initial.rot,
        )
        self.scene.calibration_fixture = AssetBaseCfg(
            prim_path='{ENV_REGEX_NS}/CalibrationFixture',
            spawn=sim_utils.UsdFileCfg(usd_path=selection['fixture_usd']),
            init_state=AssetBaseCfg.InitialStateCfg(pos=initial.pos, rot=initial.rot),
        )
        # Frustum geometry is logged only to the GL viewer and cannot enter the RGB sensor.
        self.sim.default_visualizer_cfg = NewtonGLVisualizerCfg(
            headless=True, window_width=1280, window_height=1024,
            streaming_view=False, show_collision=False, enable_picking=False,
            enable_live_plots=False, enable_sky=False, enable_shadows=True,
            background_color=(.93, .95, .98),
            eye=(.67, -.45, .98), lookat=(.105, .025, .61),
            max_visible_envs=1,
        )
        self.video_recorders = [
            VideoRecorderCfg(
                source=source, output_dir=str(OUT / 'video'), fps=30,
                video_length=300, video_interval=0, frame_stride=1,
                output_filename_prefix=name,
            )
            for source, name in [
                ('sensor:hikrobot_camera:rgb', 'hikrobot_native_1280x1024'),
                ('visualizer:newton_gl', 'camera_placement_overview'),
            ]
        ]


def matrix_from_pose(position, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = position
    return matrix


def tube_mesh(segments, radius=.0008):
    vertices, faces = [], []
    for start, end in segments:
        start, end = np.asarray(start), np.asarray(end)
        z = end - start
        z /= np.linalg.norm(z)
        u = np.cross(z, [0., 0., 1.] if abs(z[2]) < .9 else [1., 0., 0.])
        u /= np.linalg.norm(u)
        v = np.cross(z, u)
        base = len(vertices)
        for pos in [start, end]:
            for a in np.arange(8) * (2 * np.pi / 8):
                vertices.append(pos + radius * (np.cos(a) * u + np.sin(a) * v))
        for i in range(8):
            j = (i + 1) % 8
            faces += [[base+i, base+j, base+8+j], [base+i, base+8+j, base+8+i]]
    return np.asarray(vertices, np.float32), np.asarray(faces, np.int32).reshape(-1)


def log_mesh(viz, name, vertices, indices, color):
    viz.log_mesh(
        name, wp.array(vertices, dtype=wp.vec3, device='cuda:0'),
        wp.array(indices, dtype=wp.int32, device='cuda:0'),
        color=color, backface_culling=False,
    )


def add_camera_guides(viz, world_camera, world_base, transforms, intrinsics):
    camera_origin = world_camera[:3, 3]
    pixels = np.array([[0., 0.], [1280., 0.], [1280., 1024.], [0., 1024.]], np.float64)
    xy = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), np.asarray(intrinsics['camera_matrix']),
        np.asarray(intrinsics['dist_coeffs']),
    ).reshape(-1, 2)
    corners = np.column_stack((xy, np.ones(4))) * .46
    corners = corners @ world_camera[:3, :3].T + camera_origin
    segments = [(camera_origin, point) for point in corners]
    segments += [(corners[i], corners[(i+1) % 4]) for i in range(4)]
    vertices, indices = tube_mesh(segments)
    log_mesh(viz, '/CalibrationDisplay/CameraFrustum', vertices, indices, (1., .63, .05))
    # Small wire box represents the camera body without assuming a physical housing model.
    box = np.array([[x, y, z] for z in [-.035, -.006] for y in [-.019, .019]
                    for x in [-.023, .023]])
    box = box @ world_camera[:3, :3].T + camera_origin
    edges = [(i, j) for i in range(8) for j in range(i+1, 8)
             if bin(i ^ j).count('1') == 1]
    vertices, indices = tube_mesh([(box[i], box[j]) for i, j in edges], .0015)
    log_mesh(viz, '/CalibrationDisplay/CameraSymbol', vertices, indices, (.1, .32, .55))
    for label, transform, length in [
        ('Camera', world_camera, .045),
        ('Palm', world_base @ transforms['T_base_palm'], .05),
        ('Base', world_base, .035),
    ]:
        for axis, color in enumerate([(1., .12, .1), (.1, .72, .2), (.12, .28, 1.)]):
            start = transform[:3, 3]
            end = start + transform[:3, axis] * length
            vertices, indices = tube_mesh([(start, end)], .001)
            log_mesh(viz, '/CalibrationDisplay/' + label + str(axis), vertices, indices, color)


base = gym.spec('Isaac-ViserDex-Repose-Revo3')
creator = load_env_creator(base.entry_point) if isinstance(base.entry_point, str) else base.entry_point


def create_checked_env(**kwargs):
    import newton
    from isaaclab_newton.physics import NewtonManager

    cfg = kwargs['cfg']
    assert cfg.scene.num_envs == 1 and cfg.decimation == 4
    assert abs(cfg.sim.dt - 1/120.) < 1e-12
    assert cfg.sim.physics.num_substeps == 2
    assert cfg.scene.robot.spawn.usd_path == selection['robot_usd']
    assert cfg.scene.object.spawn.usd_path == selection['object_usd']
    for key in ['robot', 'object']:
        assert hashlib.sha256(Path(selection[key+'_usd']).read_bytes()).hexdigest() == selection[key+'_sha256']
    env = creator(**kwargs)
    try:
        raw = env.unwrapped
        robot = raw.scene['robot']
        camera = raw.scene['hikrobot_camera']
        intrinsics, _, transforms = load_calibration(selection['calibration_directory'])
        camera.update(0., force_recompute=True)
        world_base = matrix_from_pose(
            robot.data.root_pos_w.torch[0].cpu().numpy(),
            robot.data.root_quat_w.torch[0].cpu().numpy(),
        )
        expected = world_base @ transforms['T_base_palm'] @ transforms['T_palm_camera']
        actual = matrix_from_pose(camera.data.pos_w.torch[0].cpu().numpy(),
                                  camera.data.quat_w_ros.torch[0].cpu().numpy())
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(
            camera.data.intrinsic_matrices.torch[0].cpu().numpy(),
            intrinsics['camera_matrix'], atol=2e-4,
        )
        model = NewtonManager.get_model()
        flags, types = model.shape_flags.numpy(), model.shape_type.numpy()
        hand = [i for i, label in enumerate(model.shape_label)
                if '/Robot/' in label and flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES)]
        obj = [i for i, label in enumerate(model.shape_label)
               if '/object/' in label and flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES)]
        fixture = [i for i, label in enumerate(model.shape_label) if '/CalibrationFixture/' in label]
        assert len(hand) == 116 and len(obj) == 1
        assert int(types[obj[0]]) == int(newton.GeoType.BOX)
        assert not any(flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES) for i in fixture)
        for viz in raw.sim.visualizers:
            if viz.cfg.visualizer_type == 'newton_gl':
                add_camera_guides(viz, actual, world_base, transforms, intrinsics)

        # Save a controlled zero-joint view for comparison with the static calibration photo.
        # Restore the live state before the official teacher rollout begins.
        saved_q = robot.data.joint_pos.torch.clone()
        saved_qd = robot.data.joint_vel.torch.clone()
        object_state = raw.scene['object'].data.root_state_w.torch.clone()
        joint_limits = robot.data.soft_joint_pos_limits.torch
        zero = torch.zeros_like(saved_q).clamp(joint_limits[..., 0], joint_limits[..., 1])
        robot.write_joint_state_to_sim(zero, torch.zeros_like(saved_qd))
        moved = object_state.clone()
        moved[:, 2] -= 10.
        raw.scene['object'].write_root_pose_to_sim(moved[:, :7])
        raw.sim.forward()
        raw.scene.update(0.)
        camera.update(0., force_recompute=True)
        static_rgb = camera.data.output['rgb'].torch[0].cpu().numpy()[..., :3]
        Image.fromarray(static_rgb).save(OUT / 'calibration_static_camera.png')
        np.save(OUT / 'calibration_static_depth.npy',
                camera.data.output['distance_to_image_plane'].torch[0].cpu().numpy())
        for viz in raw.sim.visualizers:
            if viz.cfg.visualizer_type == 'newton_gl':
                Image.fromarray(viz.render_rgb_array()).save(OUT / 'calibration_camera_placement.png')
        robot.write_joint_state_to_sim(saved_q, saved_qd)
        raw.scene['object'].write_root_state_to_sim(object_state)
        raw.sim.forward()
        raw.scene.update(0.)
        camera.update(0., force_recompute=True)
        Image.fromarray(camera.data.output['rgb'].torch[0].cpu().numpy()[..., :3]).save(
            OUT / 'policy_initial_camera.png')

        record = {
            'source': selection,
            'physics': str(cfg.sim.physics.class_type), 'num_envs': 1,
            'policy_hz': 1/raw.step_dt, 'physics_dt': cfg.sim.dt,
            'substeps': cfg.sim.physics.num_substeps,
            'nconmax': cfg.sim.physics.solver_cfg.nconmax,
            'njmax': cfg.sim.physics.solver_cfg.njmax,
            'use_cuda_graph': cfg.sim.physics.use_cuda_graph,
            'hand_colliders': len(hand), 'object_colliders': len(obj),
            'visual_only_fixture_shapes': len(fixture),
            'image_size': [1280, 1024], 'intrinsics': intrinsics['camera_matrix'],
            'distortion_order': intrinsics['distortion_order'],
            'distortion': intrinsics['dist_coeffs'],
            'native_opencv_distortion': True, 'renderer': 'NewtonWarpRenderer',
            'T_world_base': world_base.tolist(),
            'T_world_camera_expected': expected.tolist(),
            'T_world_camera_actual': actual.tolist(),
            'camera_transform_max_abs_error': float(np.max(np.abs(actual-expected))),
            'base_to_palm_m': transforms['T_base_palm'][:3, 3].tolist(),
            'static_snapshot': 'zero joints clamped to limits; object moved out of view; full state restored',
            'zero_joint_positions': zero[0].cpu().tolist(),
            'rollout_observation_noise_enabled': cfg.observations.policy.enable_corruption,
            'episode_length_s': cfg.episode_length_s,
            'video_seconds': 10, 'fps': 30, 'speed_multiplier': 1,
            'camera_location_is_relative_to_hand_base': True,
            'world_table_pose_not_calibrated': True,
        }
        (OUT / 'camera_runtime_audit.json').write_text(json.dumps(record, indent=2) + '\n')
        print('HIKROBOT_CAMERA_RUNTIME_AUDIT_OK', json.dumps(record), flush=True)
    except BaseException:
        env.close()
        raise
    if os.environ.get('HIKROBOT_RECORD_SUCCESSES') == '1':
        from viserdex.utils.goal_counting import GoalCountingWrapper
        return GoalCountingWrapper(env, OUT)
    return env


gym.register(
    id='Isaac-ViserDex-Repose-Revo3-Hikrobot-Preview', entry_point=create_checked_env,
    disable_env_checker=True, kwargs={**base.kwargs, 'env_cfg_entry_point': HikrobotPreviewEnvCfg},
)

from isaaclab_rl.entrypoints import run_play_cli
raise SystemExit(run_play_cli(sys.argv[1:]))
