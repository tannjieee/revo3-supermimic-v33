"""ViserDex Appendix B-B defaults, adapted to the released Revo3 teacher."""

from dataclasses import dataclass


@dataclass
class StudentCfg:
    num_actions: int = 21
    action_history: int = 4
    proprio_dim: int = 113  # joints + four actions + goal quaternion + palm position + time
    extero_dim: int = 11
    privileged_dim: int = 27  # released teacher: joint velocity + object linear/angular velocity
    extero_hidden: tuple[int, ...] = (256, 256)
    extero_latent: int = 64
    belief_dim: int = 256
    rnn_hidden: int = 256
    rnn_layers: int = 2
    latent_hidden: tuple[int, ...] = (256, 256)
    gate_hidden: tuple[int, ...] = (128, 128)
    decoder_hidden: tuple[int, ...] = (256, 256)
    actor_hidden: tuple[int, ...] = (1024, 1024, 512, 512)


@dataclass
class DistillationCfg:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01  # AdamW default; unspecified in paper
    epochs: int = 32
    rollout_steps: int = 45
    beta_initial: float = 0.9
    beta_decay: float = 0.95
    reconstruction_coef: float = 0.2
    decoder_l1_coef: float = 0.2
    decoder_extero_coef: float = 2.0
    action_noise_std: float = 0.02
    max_grad_norm: float = 1.0
    update_env_batch_size: int | None = None  # Gradient accumulation; all environments still participate.
    optimizer_env_batch_size: int | None = None  # Logical optimizer batch, independent of rollout parallelism.
    refresh_hidden_after_update: bool = False  # Legacy checkpoints preserve their original state handling.
    teacher_quaternion_sign: str = "raw"

    def __post_init__(self):
        if self.epochs < 1 or self.rollout_steps < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, rollout_steps and learning_rate must be positive")
        if not 0 <= self.beta_initial <= 1 or not 0 <= self.beta_decay <= 1:
            raise ValueError("DAgger probabilities must be in [0, 1]")
        if self.update_env_batch_size is not None and self.update_env_batch_size < 1:
            raise ValueError("update_env_batch_size must be positive")
        if self.optimizer_env_batch_size is not None and self.optimizer_env_batch_size < 1:
            raise ValueError("optimizer_env_batch_size must be positive")
        if self.teacher_quaternion_sign not in ("raw", "positive_world_w"):
            raise ValueError("Unsupported teacher quaternion sign convention")

    def beta(self, completed_iterations: int) -> float:
        return self.beta_initial * self.beta_decay**completed_iterations


@dataclass
class PerceptionNoiseCfg:
    enabled: bool = True
    # One encoder reading is shared by teacher and student; clean state stays separate.
    # Magnitudes follow NVIDIA DeXtreme ManualDR; bias lasts one episode here.
    joint_enabled: bool = True
    joint_noise_std: float = 0.005
    joint_bias_std: float = 0.01
    joint_delay_max_steps: int = 1  # Engineering default; paper does not specify proprio delay range.
    max_period: int = 3
    jitter_probability: float = 0.1
    failure_probability: float = 0.3
    position_noise: float = 0.012
    position_bias: float = 0.012
    rotation_noise_deg: float = 1.0
    rotation_bias_deg: float = 0.1
    failure_position_extent: float = 0.15  # hand-frame cube; paper does not specify replacement distribution

    def __post_init__(self):
        if self.max_period < 1:
            raise ValueError("max_period must be positive")
        if not isinstance(self.joint_delay_max_steps, int) or self.joint_delay_max_steps < 0:
            raise ValueError("joint_delay_max_steps must be a nonnegative integer")
        if not 0 <= self.jitter_probability <= 1 or not 0 <= self.failure_probability <= 1:
            raise ValueError("Perception probabilities must be in [0, 1]")
        if (
            min(
                self.position_noise,
                self.position_bias,
                self.rotation_noise_deg,
                self.rotation_bias_deg,
                self.failure_position_extent,
                self.joint_noise_std,
                self.joint_bias_std,
            )
            < 0
        ):
            raise ValueError("Noise magnitudes must be nonnegative")
