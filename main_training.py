import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from simul_env.coll_env import SigmabanEnv

# -----------------------------
def write(msg):
    print(msg, flush=True)

# --- Config ---
saveName = "ppo_sigmaban_quick"
log_dir = "./logs"
save_dir = "./checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
write("=========================================")
write(f"Using device: {device}")
write("=========================================")

stats_path = os.path.join(save_dir, f"{saveName}_vec_normalize.pkl")

# -----------------------------
def make_env_fn():
    def _init():
        env = SigmabanEnv(render_mode=None, maxSec=15.0)
        return Monitor(env)
    return _init

# -----------------------------
def main():
    n_timesteps = int(100e6)
    n_envs = 16
    n_steps = 512
    batch_size = 64
    n_epochs = 2
    gamma = 0.998
    learning_rate = 5e-5
    gae_lambda = 0.95
    ent_coef = 0.0001
    use_sde = True
    sde_sample_freq = 3

    policy_kwargs = dict(
        log_std_init=-2.0,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    # --- Vectorized environments ---
    env = make_vec_env(
        SigmabanEnv,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
        env_kwargs=dict(render_mode=None, maxSec=15.0),
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=gamma)

    eval_env = make_vec_env(
        SigmabanEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs=dict(render_mode=None, maxSec=15.0),
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, gamma=gamma)
    eval_env.training = False
    eval_env.norm_reward = False

    # --- Logger ---
    log_path = os.path.join(log_dir, saveName)
    new_logger = configure(log_path, ["stdout", "log", "tensorboard"])

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,  # small quick save interval
        save_path=save_dir,
        name_prefix=saveName,
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=3, verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=stop_callback,
    )

    # --- PPO Model ---
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        learning_rate=learning_rate,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=0.2,
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
    )
    model.set_logger(new_logger)

    # --- Train ---
    write("Training Starts...")
    try:
        model.learn(
            total_timesteps=n_timesteps,
            callback=[checkpoint_callback, eval_callback],
            reset_num_timesteps=False,
        )
    except Exception as e:
        write(f"\n!!! Training interrupted by error: {e} !!!")
        import traceback
        traceback.print_exc()
    finally:
        final_path = os.path.join(save_dir, f"{saveName}_final.zip")
        model.save(final_path)
        env.save(stats_path)
        write("--- Training Finished ---")
        write(f"Final model saved as {final_path}")
        write(f"Normalization stats saved as {stats_path}")

# -----------------------------
if __name__ == "__main__":
    main()
