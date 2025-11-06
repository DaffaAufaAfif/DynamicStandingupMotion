import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from simul_env.coll_env import SigmabanEnv


def write(msg):
    print(msg, flush=True)


# --- Config ---
saveName = "ppo_sigmaban_fast_5nov"
log_dir = "./logs"
save_dir = "./checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
write("=========================================")
write(f"Using device: {device}")
write("=========================================")

# --- Vectorization ---
N_ENVS = 8
stats_path = os.path.join(save_dir, f"{saveName}_vec_normalize.pkl")


def make_env_fn():
    """Create one monitored environment instance."""
    def _init():
        env = SigmabanEnv(render_mode=None, maxSec=10.0)
        return Monitor(env)
    return _init


def main():
    # ---- Training budget ----
    total_timesteps = 8_000_000         # Fast run (~<10M steps)
    saveFrq_timesteps = 100_000
    saveFrq_steps_per_env = saveFrq_timesteps // N_ENVS

    # --- Vectorized training env ---
    env = make_vec_env(
        SigmabanEnv,
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv,
        env_kwargs=dict(render_mode=None, maxSec=10.0),
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

    # --- Vectorized eval env ---
    eval_env = make_vec_env(
        SigmabanEnv,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs=dict(render_mode=None, maxSec=10.0),
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, gamma=0.99)
    eval_env.training = False
    eval_env.norm_reward = False  # Don’t normalize reward during eval

    # --- Logger ---
    log_path = os.path.join(log_dir, saveName)
    new_logger = configure(log_path, ["stdout", "log", "tensorboard"])

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=saveFrq_steps_per_env,
        save_path=save_dir,
        name_prefix=saveName,
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,  # Stop early if plateau
        min_evals=3,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=100_000,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=stop_callback,
    )

    # --- Resume checkpoint if present ---
    latest_checkpoint = None
    checkpoint_prefix = saveName + "_"
    if os.path.exists(save_dir):
        ckpts = [
            f for f in os.listdir(save_dir)
            if f.startswith(checkpoint_prefix) and f.endswith(".zip")
        ]
        if ckpts:
            latest_checkpoint = max(
                ckpts, key=lambda x: os.path.getmtime(os.path.join(save_dir, x))
            )

    model = None
    if latest_checkpoint and os.path.exists(stats_path):
        write(f"Resuming from checkpoint: {latest_checkpoint}")
        env = VecNormalize.load(stats_path, env)
        env.training = True

        model_path = os.path.join(save_dir, latest_checkpoint)
        model = PPO.load(model_path, env=env, device=device)
        model.set_logger(new_logger)
    else:
        write(f"Starting new training session | {saveName}")

        # --- PPO policy/net architecture ---
        policy_kwargs = dict(
            log_std_init=-2.0,
            ortho_init=False,
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )

        # --- PPO hyperparams (optimized for quick training) ---
        lr_schedule = get_linear_fn(5e-4, 1e-5, 1.0)

        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=1,
            n_steps=256,         # faster updates
            batch_size=128,      # smaller batches
            n_epochs=4,          # fewer passes per batch
            learning_rate=lr_schedule,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,       # slightly more exploration
            vf_coef=0.5,
            clip_range=0.2,
            use_sde=True,
            sde_sample_freq=4,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
        )
        model.set_logger(new_logger)

    # --- Train ---
    write("Training Starts...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
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
        write("--- Training Finished or Interrupted ---")
        write(f"Final model saved as {final_path}")
        write(f"Normalization stats saved as {stats_path}")


if __name__ == "__main__":
    main()
