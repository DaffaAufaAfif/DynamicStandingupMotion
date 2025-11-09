import os
import time
import mujoco
from stable_baseline3 import PPO
#from simul_env.env_wrapper import SigmabanEnv
from simul_env.coll_env import SigmabanEnv
collision_weights = {}
for i in range(7):  # torso parts
    collision_weights[i] = "Torso"
for i in range(7, 15):  # neck & head
    collision_weights[i] = "Heads"
for i in range(15, 31):  # arms
    collision_weights[i] = "Arms"
for i in range(31, 42):  # left leg
    collision_weights[i] = "Left Leg"
for i in range(42, 48):  # left foot
    collision_weights[i] = "Left foot"
for i in range(48, 59):  # right leg
    collision_weights[i] = "Rigt leg"
for i in range(59, 65):  # right foot
    collision_weights[i] = "right foot"

def get_collision_penalty(sim):
    #penalty = 0.0
    #collisions=[]
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        n1 = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        n2 = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, g2)

        name1 = n1.lower() if n1 else ""
        name2 = n2.lower() if n2 else ""

        if "floor" in name1 or "ground" in name1:
            body_id = g2
        elif "floor" in name2 or "ground" in name2:
            body_id = g1
        else:
            continue
        print(body_id, collision_weights[body_id])
       # if body_id<7:
        #    print("HEYYYY")
         #   time.sleep(5)
        #penalty += collision_weights.get(body_id, 0.0)
        #collisions.append((body_id, penalty, n1, n2))
        

def find_latest_checkpoint(checkpoint_dir="./checkpoints"):
    """
    Finds the most recent PPO checkpoint file.
    """
    if not os.path.exists(checkpoint_dir):
        print(" No checkpoint directory found.")
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    if not checkpoints:
        print(" No checkpoints found in", checkpoint_dir)
        return None

    latest = max(
        checkpoints,
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )
    return os.path.join(checkpoint_dir, latest)


def main():
    # Initialize environment with render
    env = SigmabanEnv(render_mode="human")

    # Try to find latest checkpoint
    latest_model_path = find_latest_checkpoint()
    if latest_model_path is None:
        #print(" No PPO model found! Train the model first.")
        print("Manual mode")
        latest_model_path="ppo_sigmaban_co2.zip"
        #return

    print(f"Loading latest model: {latest_model_path}")
    model = PPO.load(latest_model_path, env=env)

    obs, _ = env.reset()
    step = 0

    print(" Running trained model in simulation... Press Ctrl+C to exit.")

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.sim.render(realtime=True)  # render in Mujoco viewer
            step += 1
            #get_collision_penalty(sim=env)
            try:
                current_z = env.sim.data.body("root").xpos[2]
            except Exception:
                current_z = 0.0
            print(current_z)
            env._get_collision_penalty()
            if terminated or truncated:
                print(f"Episode finished after {step} steps.")
                obs, _ = env.reset()
                step = 0

            # Slow down rendering a bit
            #time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    env.close()


if __name__ == "__main__":
    main()
