import gymnasium as gym
import numpy as np
from simul_env.simulator import Simulator


class SigmabanEnv(gym.Env):
    """
    Simplified environment for standing control task.
    Based on the structure of StandupEnv but without the heavy options system.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(self, render_mode=None, maxSec: float = 15.0):
        super().__init__()

        # --- Simulation ---
        self.sim = Simulator()
        self.render_mode = render_mode
        self.agent_dt = 0.05
        self.episode_duration_s = maxSec
        self.physics_dt = self.sim.model.opt.timestep
        self.physics_steps_per_agent_step = int(self.agent_dt / self.physics_dt)
        self.max_steps = int(self.episode_duration_s / self.agent_dt)

        # --- Joints & Actuators ---
        self.dofs = ["elbow", "shoulder_pitch", "hip_pitch", "knee", "ankle_pitch"]
        self.left_actuators = [f"left_{dof}" for dof in self.dofs]
        self.right_actuators = [f"right_{dof}" for dof in self.dofs]
        self.ranges = [self.sim.model.actuator(a).ctrlrange for a in self.left_actuators]
        self.range_low = np.array([r[0] for r in self.ranges])
        self.range_high = np.array([r[1] for r in self.ranges])

        # --- Spaces ---
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0] * len(self.dofs), dtype=np.float32),
            high=np.array([1.0] * len(self.dofs), dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = len(self.dofs) * 2 + 3 + 1 + 8
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Standing pose target ---
        self.target_pose = np.deg2rad([-49.5, 19.5, -52, 79, -36.5])
        self.target_tilt = np.deg2rad(8.5)
        self.target_state = np.concatenate([self.target_pose, [self.target_tilt]])

        # --- Episode vars ---
        self.current_step = 0
        self.last_q = np.zeros(len(self.dofs))
        self.q_history = []
        self.tilt_history = []
        self.last_termination_info = {}  # <--- added for safe storage of diagnostic info

    # -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0

        # --- stabilize before start ---
        for _ in range(int(1.5 / self.physics_dt)):
            self.sim.step()

        self.q_history = [[self.sim.get_q(f"left_{dof}") for dof in self.dofs]]
        self.tilt_history = [self.get_tilt()]
        self.last_termination_info = {}

        obs = self._get_obs()
        return obs, {}

    # -----------------------------------------------------------
    def step(self, action):
        # Map normalized [-1,1] to actuator range
        action = np.clip(action, -1, 1)
        target_ctrl = self.range_low + (action + 1) * 0.5 * (self.range_high - self.range_low)

        # Apply symmetric control
        for i, dof in enumerate(self.dofs):
            self.sim.set_control(self.left_actuators[i], target_ctrl[i])
            self.sim.set_control(self.right_actuators[i], target_ctrl[i])

        for _ in range(self.physics_steps_per_agent_step):
            self.sim.step()

        # --- Observation & Reward ---
        obs = self._get_obs()
        q = np.array([self.sim.get_q(f"left_{dof}") for dof in self.dofs])
        tilt = self.get_tilt()
        current_state = np.concatenate([q, [tilt]])

        # Reward for standing close to target
        posture_error = np.linalg.norm(current_state - self.target_state)
        posture_reward = np.exp(-20 * posture_error**2)

        # Reward for smooth action
        variation_penalty = np.exp(-np.linalg.norm(action - self.last_q)) * 0.05
        self.last_q = action

        # Penalty for collisions
        collision_penalty = np.tanh(0.001 * self.sim.self_collisions())

        reward = posture_reward + variation_penalty - collision_penalty

        # Termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        self.current_step += 1

        if self.render_mode == "human":
            self.sim.render(True)

        return obs, reward, terminated, truncated, {}

    # -----------------------------------------------------------
    def _get_obs(self):
        q = [self.sim.get_q(f"left_{dof}") for dof in self.dofs]
        qdot = [self.sim.get_qdot(f"left_{dof}") for dof in self.dofs]
        gyro = self.sim.get_gyro()
        forces = [self.sim.centroidal_force()]
        pressure = self.sim.get_pressure_sensors()
        leftpressure = np.array(pressure["left"])
        rightpressure = np.array(pressure["right"])
        return np.concatenate([q, qdot, gyro, forces, leftpressure, rightpressure]).astype(np.float32)

    # -----------------------------------------------------------
    def get_tilt(self):
        R = self.sim.data.site("trunk").xmat
        return -np.arctan2(R[6], R[8])

    # -----------------------------------------------------------
    def _check_termination(self):
        """Check if the episode should terminate."""
        info = {
            "out_of_bounds": False,
            "qpos_nan": False,
            "qpos_inf": False,
            "invalid": False,
            "flipped": False
        }

        # Check for out of bounds (xy position)
        xy_pos = self.sim.data.body("torso_2023").xpos[:2]
        dist_from_origin = np.linalg.norm(xy_pos)
        if dist_from_origin > 2.0:  # If it crawls 2 meters away
            info["out_of_bounds"] = True

        # Check for invalid qpos (NaN or Inf)
        qpos = self.sim.data.qpos
        is_nan = np.isnan(qpos).any()
        is_inf = np.isinf(qpos).any()
        if is_nan or is_inf:
            info["qpos_nan"] = bool(is_nan)
            info["qpos_inf"] = bool(is_inf)
            info["invalid"] = is_nan or is_inf

        # Check flipped using torso_2023 orientation
        try:
            xmat = np.asarray(self.sim.data.body("torso_2023").xmat).reshape(3, 3)
            up_z = float(xmat[2, 2])
            info["root_up_z"] = up_z
            info["flipped"] = up_z < -0.1
        except Exception:
            info["root_up_z"] = None
            info["flipped"] = False

        self.last_termination_info = info

        # --- Grace period logic ---
        # Allow 5 steps to settle after reset before checking termination
        GRACE_STEPS = 5
        if getattr(self, "current_step", 0) < GRACE_STEPS:
            # During grace period, only enforce severe out_of_bounds
            return bool(info["out_of_bounds"])

        # After grace period, terminate if any critical flag is True
        return bool(info["out_of_bounds"] or info["invalid"] or info["flipped"])

    # -----------------------------------------------------------
    def render(self):
        self.sim.render(True)
