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
        obs_dim = len(self.dofs) * 2 + 3 + 1 + 8
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.standing_pose_qpos = np.deg2rad([-49.5, 19.5, -52, 79, -36.5])
        self.target_height = 0.4
        
        # --- Episode variables ---
        self.current_step = 0
        self.last_q = np.zeros(len(self.control_joints), dtype=np.float32)
        self.last_termination_info = {}
        

        # --- Collision Weights ---
        self.collision_weights = {}
        for i in range(7):  # torso
            self.collision_weights[i] = 5.0
        for i in range(7, 15):  # neck to head
            self.collision_weights[i] = 10.0
        for i in range(15, 31):  # shoulders to elbows
            self.collision_weights[i] = 0.8
        for i in range(31, 42):  # left leg
            self.collision_weights[i] = 0.5
        for i in range(42, 48):  # left foot
            self.collision_weights[i] = -0.05
        for i in range(48, 59):  # right leg
            self.collision_weights[i] = 0.5
        for i in range(59, 65):  # right foot
            self.collision_weights[i] = -0.05
        cleats=[45,46,47,48, 
                62,63,64,65]
        for i in cleats:
            self.collision_weights[i] = -0.25
    # -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.current_step = 0
        self.last_q[:] = 0
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
            posture_reward *0.01
            + variation_bonus *0.1
            - collisions
        )

        self.current_step += 1
        terminated = self._check_termination()
        truncated = self.current_step >= self.maxStep


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
        dictpressure = self.sim.get_pressure_sensors()
        leftpress= dictpressure["left"]
        rightpres= dictpressure["right"]
        obs = np.concatenate([joint_pos, joint_vel, gyro, forces, leftpress, rightpres])
        return obs.astype(np.float32)

    # -----------------------------------------------------------
    def _get_collision_penalty(self, show=False):
        if show:
            print("------")
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
            
            if show:
                print(body_id,self.collision_weights.get(body_id, 0.0))
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

