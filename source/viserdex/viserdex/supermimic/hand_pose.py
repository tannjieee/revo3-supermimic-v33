"""Wuji hand_pose semantics on the official Newton asset tensor interface.

Reference: wuji-technology/wuji-mjlab b2b98d7, reorient/mdp/rewards.py.
Sum squared actual joint deviations; the reward manager applies weight and dt.
"""

def hand_pose_penalty(env):
    robot = env.scene["robot"]
    return (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch).square().sum(dim=-1)
