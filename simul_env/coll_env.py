import gymnasium as gym
import numpy as np
import mujoco
import time
from collections import deque 
from simul_env.simulator import Simulator


class SigmabanEnv(gym.Env):
    """
    Gymnasium environment wrapper for the Sigmaban humanoid robot.

    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}


    def __init__(
        self,
        model_dir: str = None,
        render_mode: str | None = None,
        maxSec: float = 15.0,
        randomization_noise_level: float = 0.0,
        curriculum_enabled: bool = True,
        reward_threshold: float = -8.0,
        noise_increment: float = 0.05,
        max_noise: float = 0.2,
    ):
        
        super().__init__()

        self.sim = Simulator(model_dir)
        self.render_mode = render_mode
        self.agent_dt = 0.05
        self.episode_duration_s = maxSec
        physics_dt = self.sim.model.opt.timestep
        if physics_dt <= 0:
            raise ValueError(
                "MuJoCo model physics timestep (opt.timestep) must be greater than 0."
            )
        self.physics_steps_per_agent_step = int(self.agent_dt / physics_dt)
        self.maxStep = int(self.episode_duration_s / self.agent_dt)

        # --- 5-DOF Symmetric Control ---
        self.dofs = [
            "elbow", "shoulder_pitch", "hip_pitch", "knee", "ankle_pitch"
        ]
        self.control_joints = [
            "left_elbow", "left_shoulder_pitch", "left_hip_pitch", "left_knee", "left_ankle_pitch"
        ]
        self.left_actuators = [f"left_{dof}" for dof in self.dofs]
        self.right_actuators = [f"right_{dof}" for dof in self.dofs]
        self.ranges = [
            self.sim.model.actuator(act).ctrlrange for act in self.control_joints
        ]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0] * len(self.dofs), dtype=np.float32),
            high=np.array([1.0] * len(self.dofs), dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = len(self.control_joints) * 2 + 3 + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.standing_pose_qpos = np.deg2rad([-49.5, 19.5, -52, 79, -36.5])
        self.target_height = 0.4
        
        # --- Episode variables ---
        self.current_step = 0
        self.last_q = np.zeros(len(self.control_joints), dtype=np.float32)
        self.last_termination_info = {}
        
        # --- MODIFIED: Internal Curriculum Variables ---
        self.randomization_noise_level = randomization_noise_level
        self.curriculum_enabled = curriculum_enabled
        self.reward_threshold = reward_threshold
        self.noise_increment = noise_increment
        self.max_noise = max_noise
        
        # Buffer to track mean reward over last 100 episodes
        self.episode_reward_buffer = deque(maxlen=100)
        self.current_episode_reward = 0.0

        # --- Collision Weights ---
        self.collision_weights = {}
        legs = 0.9
        foots = 0.5
        for i in range(1,8):  # torso
            self.collision_weights[i] = 3
        for i in range(8, 16):  # neck to head
            self.collision_weights[i] = 5
        for i in range(16, 32):  # shoulders to elbows
            self.collision_weights[i] = 0.95
        for i in range(32, 43):  # left leg
            self.collision_weights[i] = legs
        for i in range(43, 49):  # left foot
            self.collision_weights[i] = foots
        for i in range(49, 60):  # right leg
            self.collision_weights[i] = legs
        for i in range(60, 66):  # right foot
            self.collision_weights[i] = foots
        cleats=[45,46,47,48, 
                62,63,64,65]
        for i in cleats:
            self.collision_weights[i] = -0.2
    # -----------------------------------------------------------
    def set_randomization_level(self, level: float):
        """Public method to manually override noise level if needed."""
        self.randomization_noise_level = level
    
    # -----------------------------------------------------------
    def _update_curriculum(self):
        """
        --- NEW: Internal curriculum logic ---
        Checks mean reward and increases noise if threshold is met.
        """
        if not self.curriculum_enabled:
            return

        # Only check if the buffer is full (e.g., 20 episodes)
        if len(self.episode_reward_buffer) == 20:
            mean_reward = np.mean(self.episode_reward_buffer)
            
            if mean_reward > self.reward_threshold:
                new_noise = self.randomization_noise_level + self.noise_increment
                
                if new_noise <= self.max_noise:
                    # Success! Increase noise
                    self.randomization_noise_level = new_noise
                    print(f"--- CURRICULUM: Success! Mean reward {mean_reward:.2f} > {self.reward_threshold:.2f} ---")
                    print(f"--- CURRICULUM: Increasing noise to {self.randomization_noise_level:.3f} ---")
                    
                    # Clear buffer to wait for a new stable average
                    self.episode_reward_buffer.clear()
                
                elif self.randomization_noise_level < self.max_noise:
                    # We are at the max noise level
                    print(f"--- CURRICULUM: Reached Max Noise Level: {self.max_noise:.3f} ---")
                    self.randomization_noise_level = self.max_noise
                    self.curriculum_enabled = False # Turn off curriculum
                    
    # -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        
        # --- NEW: Curriculum logic runs on reset ---
        # (Only if it's not the very first step of the run)
        if self.current_step > 0:
            self.episode_reward_buffer.append(self.current_episode_reward)
            self._update_curriculum()
            
        self.current_episode_reward = 0.0
        # --- End of new logic ---

        super().reset(seed=seed)
        self.sim.reset()

        # --- 5-second stabilization period ---
        physics_dt = self.sim.model.opt.timestep
        stabilization_seconds = 5.0
        num_stabilization_steps = int(stabilization_seconds / physics_dt)
        for _ in range(num_stabilization_steps):
            self.sim.step()
            
        # --- Apply randomization *after* stabilization ---
        if self.randomization_noise_level > 0.0:
            qpos = self.sim.data.qpos.copy()
            qvel = self.sim.data.qvel.copy()
            nq = self.sim.model.nq
            nv = self.sim.model.nv

            pos_noise = self.np_random.uniform(
                low=-self.randomization_noise_level,
                high=self.randomization_noise_level,
                size=nq
            )
            pos_noise[0:7] = 0.0  # No noise on root
            
            vel_noise_scale = self.randomization_noise_level * 0.1 
            vel_noise = self.np_random.uniform(
                low=-vel_noise_scale,
                high=vel_noise_scale,
                size=nv
            )
            vel_noise[0:6] = 0.0 # No noise on root
            
            self.sim.data.qpos[:] = qpos + pos_noise
            self.sim.data.qvel[:] = qvel + vel_noise
            mujoco.mj_forward(self.sim.model, self.sim.data)

        self.current_step = 0
        self.last_q = np.array([self.sim.get_q(j) for j in self.control_joints])
        self.last_termination_info = {}

        return self._get_obs(), {}

    # -----------------------------------------------------------
    def step(self, action):
        target_q = np.zeros(len(self.control_joints), dtype=np.float32)
        for i, joint in enumerate(self.control_joints):
            jmin, jmax = self.ranges[i]
            target_q[i] = jmin + (action[i] + 1) * 0.5 * (jmax - jmin)
        for i in range(len(self.dofs)):
            self.sim.set_control(self.left_actuators[i], float(target_q[i]))
            self.sim.set_control(self.right_actuators[i], float(target_q[i]))

        for _ in range(self.physics_steps_per_agent_step):
            self.sim.step()

        obs = self._get_obs()
        current_q = np.array([self.sim.get_q(j) for j in self.control_joints])
        posture_error = np.sum(np.abs(current_q - self.standing_pose_qpos))
        posture_reward = np.exp(-2.0 * posture_error)
        collisions = self._get_collision_penalty()
        pose_change = np.mean(np.abs(current_q - self.last_q))
        variation_bonus = pose_change
        self.last_q = current_q

        reward = (
            posture_reward *0.6
            + variation_bonus *0.1
            - collisions *0.2
        )

        self.current_step += 1
        terminated = self._check_termination()
        truncated = self.current_step >= self.maxStep

        # --- NEW: Track reward for internal curriculum ---
        self.current_episode_reward += reward

        info = {
            "collisions": collisions,
            "posture_reward": posture_reward,
            "variation_bonus": variation_bonus,
        }
        info.update(self.last_termination_info)

        if self.render_mode == "human":
            self.sim.render(True)

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------
    def _get_obs(self):
        joint_pos = [self.sim.get_q(j) for j in self.control_joints]
        joint_vel = [self.sim.get_qdot(j) for j in self.control_joints]
        gyro = self.sim.get_gyro()
        forces = [self.sim.centroidal_force()]
        obs = np.concatenate([joint_pos, joint_vel, gyro, forces])
        return obs.astype(np.float32)

    # -----------------------------------------------------------
    def _get_collision_penalty(self):
        penalty = 0.0
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            n1 = mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            n2 = mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, g2)

            name1 = n1.lower() if n1 else ""
            name2 = n2.lower() if n2 else ""

            if "floor" in name1 or "ground" in name1:
                body_id = g2
            elif "floor" in name2 or "ground" in name2:
                body_id = g1
            else:
                continue # This is a self-collision, ignore for now
            
            penalty += self.collision_weights.get(body_id, 0.0)
        return penalty

    # -----------------------------------------------------------
    def _check_termination(self):
        # This function now has the KeyError fix
        info = {}
        try:
            x, y, z = self.sim.data.body("root").xpos
            info["root_pos"] = (float(x), float(y), float(z))
        except Exception as e:
            info["root_pos"] = None
            info["error_reading_root_pos"] = str(e) # Good to log the error

        if info["root_pos"] is not None:
            x, y, z = info["root_pos"]
            info["fell_through"] = z < 0.2
            info["out_of_bounds"] = abs(x) > 5 or abs(y) > 5 or z > 3
        
        qpos = getattr(self.sim.data, "qpos", None)
        if qpos is None:
            info["invalid"] = True
        else:
            is_nan = np.any(np.isnan(qpos))
            is_inf = np.any(np.isinf(qpos))
            info["qpos_nan"] = bool(is_nan)
            info["qpos_inf"] = bool(is_inf)
            info["invalid"] = is_nan or is_inf
        
        try:
            xmat = np.asarray(self.sim.data.body("root").xmat).reshape(3, 3)
            up_z = float(xmat[2, 2])
            info["root_up_z"] = up_z
            info["flipped"] = up_z < -0.1
        except Exception:
            info["root_up_z"] = None
            info["flipped"] = False

        self.last_termination_info = info

        GRACE_STEPS = 5 #to ignore some termination for 5 step
        if getattr(self, "current_step", 0) < GRACE_STEPS:
            return bool(info.get("out_of_bounds", False))

        return bool(
            info.get("fell_through", False)
            or info.get("out_of_bounds", False)
            or info.get("invalid", False)
            or info.get("flipped", False)
        )

