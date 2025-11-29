%%writefile simul_env/coll_env.py

import gymnasium as gym
import numpy as np
from simul_env.simulator import Simulator

# --- KODE BARU (VERSI STAND UP) ---
class SigmabanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, maxSec: float = 10.0):
        super().__init__()
        self.sim = Simulator()
        self.render_mode = render_mode
        self.agent_dt = 0.05
        self.episode_duration_s = maxSec
        self.max_steps = int(self.episode_duration_s / self.agent_dt)
        self.physics_dt = self.sim.model.opt.timestep
        self.physics_steps_per_agent_step = int(self.agent_dt / self.physics_dt)

        self.dofs = ["elbow", "shoulder_pitch", "hip_pitch", "knee", "ankle_pitch"]
        self.left_actuators = [f"left_{dof}" for dof in self.dofs]
        self.right_actuators = [f"right_{dof}" for dof in self.dofs]
        self.ranges = [self.sim.model.actuator(a).ctrlrange for a in self.left_actuators]
        self.range_low = np.array([r[0] for r in self.ranges])
        self.range_high = np.array([r[1] for r in self.ranges])

        # Action Space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.dofs) * 2,), dtype=np.float32
        )
        
        # Observation Space (Ditambah Z-Height)
        # qpos (20) + qvel (20) + gyro (3) + z-pos (1) + pressure (8) = 52
        obs_dim = len(self.dofs) * 2 * 2 + 3 + 1 + 8 
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0

        # --- PAKSA TIDUR TERLENTANG ---
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        qpos[2] = 0.35  # Jatuhkan ke lantai
        qpos[3:7] = [0.707, 0, 0.707, 0] # Rotasi 90 derajat
        
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.model.forward() # Update fisika

        # Stabilisasi
        for _ in range(20): self.sim.step()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        target_ctrl = np.concatenate([
            self.range_low + (action[:5] + 1) * 0.5 * (self.range_high - self.range_low),
            self.range_low + (action[5:] + 1) * 0.5 * (self.range_high - self.range_low)
        ])
        
        all_actuators = self.left_actuators + self.right_actuators
        for i, name in enumerate(all_actuators):
            self.sim.set_control(name, target_ctrl[i])

        for _ in range(self.physics_steps_per_agent_step):
            self.sim.step()

        # --- REWARD BARU (Fokus Tinggi Badan) ---
        torso_z = self.sim.data.body("torso_2023").xpos[2]
        
        r_height = 10.0 * torso_z 
        r_ctrl = -0.05 * np.sum(np.square(action))
        
        # Bonus jika punggung tegak
        torso_mat = self.sim.data.body("torso_2023").xmat.reshape(3, 3)
        r_upright = 2.0 * torso_mat[2, 2] if torso_mat[2, 2] > 0 else 0
        
        r_stand = 20.0 if torso_z > 0.45 else 0.0

        reward = r_height + r_upright + r_ctrl + r_stand

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = False
        if np.isnan(self.sim.data.qpos).any(): terminated = True

        if self.render_mode == "human": self.sim.render(True)

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        q = [self.sim.get_q(f"left_{dof}") for dof in self.dofs] + \
            [self.sim.get_q(f"right_{dof}") for dof in self.dofs]
        qdot = [self.sim.get_qdot(f"left_{dof}") for dof in self.dofs] + \
               [self.sim.get_qdot(f"right_{dof}") for dof in self.dofs]
        gyro = self.sim.get_gyro()
        torso_z = [self.sim.data.body("torso_2023").xpos[2]] # Z-Height Info
        pressure = self.sim.get_pressure_sensors()
        flat_pressure = np.concatenate([pressure["left"], pressure["right"]])
        
        return np.concatenate([q, qdot, gyro, torso_z, flat_pressure]).astype(np.float32)

    def render(self):
        self.sim.render(True)
