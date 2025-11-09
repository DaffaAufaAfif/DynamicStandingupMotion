import os
import re
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from simul_env.coll_env import SigmabanEnv

# -------------------------
# Project folder on Drive
# -------------------------
project_dir = "/content/drive/MyDrive/Intern_ichirov2/ProjectFall/"
log_dir = os.path.join(project_dir, "logs/sigmaban_ppo/")
save_path = os.path.join(project_dir, "checkpoints/")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

# -------------------------
# Parallel environments
# -------------------------
num_envs = 8  # adjust as needed

def make_env(rank: int):
    def _init():
        env = SigmabanEnv( render_mode=None)
        env = Monitor(env, filename=os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init

env_fns = [make_env(i) for i in range(num_envs)]
vec_env = DummyVecEnv(env_fns)

# -------------------------
# Load or create VecNormalize
# -------------------------
vecnorm_path = os.path.join(save_path, "vec_normalize.pkl")
if os.path.exists(vecnorm_path):
    print("🔄 Loading previous VecNormalize...")
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
else:
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# -------------------------
# Checkpoint callback & logger
# -------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=save_path,
    name_prefix="ppo_sigmaban"
)

logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# -------------------------
# Load last PPO checkpoint if exists
# -------------------------
def find_latest_checkpoint(path):
    checkpoints = [
        f for f in os.listdir(path)
        if f.startswith("ppo_sigmaban") and f.endswith(".zip")
    ]
    if not checkpoints:
        return None
    checkpoints.sort(
        key=lambda x: int(re.findall(r"(\d+)", x)[-1]) if re.findall(r"(\d+)", x) else 0
    )
    return os.path.join(path, checkpoints[-1])

last_ckpt = find_latest_checkpoint(save_path)
if last_ckpt:
    print(f"🔁 Resuming training from checkpoint: {last_ckpt}")
    model = PPO.load(last_ckpt, env=vec_env, device="cpu")
else:
    print("🚀 Starting new PPO training from scratch.")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048 // num_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        tensorboard_log=log_dir,
        device="cpu",
    )

model.set_logger(logger)

# -------------------------
# Train PPO
# -------------------------
total_timesteps = 10_000_000  # adjust as needed
print(f"\n🎯 Training for {total_timesteps:,} timesteps on CPU ({num_envs} envs)...\n")

model.learn(
    total_timesteps=total_timesteps,
    callback=checkpoint_callback,
    progress_bar=True,
)

# -------------------------
# Save final model & VecNormalize
# -------------------------
final_model_path = os.path.join(save_path, "ppo_sigmaban_final.zip")
model.save(final_model_path)
vec_env.save(vecnorm_path)

print(f"\n✅ Training complete. Final model saved to:\n{final_model_path}")
