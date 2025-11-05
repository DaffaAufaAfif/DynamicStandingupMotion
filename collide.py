import time
import mujoco
import numpy as np
from simul_env.simulator import Simulator


def main():
    # -----------------------------------------------------------
    # 1. Initialize simulator
    # -----------------------------------------------------------
    sim = Simulator()  # 🟡 change path if needed

    # -----------------------------------------------------------
    # 2. Define penalty values by geom ID
    # -----------------------------------------------------------
    collision_weights = {}
    collision_weights = {}
    for i in range(1,7): #0 - 6 torso(badan)
        collision_weights[i] = 8.0 #don't do
    for i in range(8,16): #7 - 14 neck to head
        collision_weights[i] = 15.0 #fatal
    for i in range(16,32): #15 - 30 shoulder to elbow
        collision_weights[i] = 1.2 #low penalty
    for i in range(32,43): #31 - 30 LEFT hip to ankle
        collision_weights[i] = 4
    for i in range(43,49): # LEFT foot
        collision_weights[i] = -9# reward 
    for i in range(49,60): # RIGHT hip to ankle
        collision_weights[i] = 4
    for i in range(60,66): # RIGHT foot
        collision_weights[i] = -9
    # -----------------------------------------------------------
    # 3. Collision penalty function using IDs
    # -----------------------------------------------------------
    def get_collision_penalty(sim, weights):
        penalty = 0.0
        collisions=[]
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
            if body_id<7:
                print("HEYYYY")
                time.sleep(5)
            penalty = collision_weights.get(body_id, 0.0)
            collisions.append((body_id, penalty, n1, n2))
        
        return penalty, collisions

    # -----------------------------------------------------------
    # 4. Run a few simulation steps to see collisions
    # -----------------------------------------------------------
    control_joints = [
            "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
            "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
            "left_ankle_pitch", "left_ankle_roll",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
            "right_ankle_pitch", "right_ankle_roll"
        ]
    print("=== Starting collision test (Simulator-based) ===")
    print(sim.get_q(j) for j in control_joints)
    for step in range(100000):
        sim.step()
        total_penalty, collided = get_collision_penalty(sim, collision_weights)

        if collided:
            print(f"\nStep {step} | Total penalty: {total_penalty:.3f}")
            for (gid, pen, n1, n2) in collided:
                print(f"  Geom {gid:2d} ({n1} <-> {n2}) → penalty {pen}")
            print("----------------------")
            time.sleep(1)

        # render if you want
        sim.render(True)
        print(sim.get_q(j) for j in control_joints)
        time.sleep(0.001)

    print("\n=== Finished collision test ===")


if __name__ == "__main__":
    main()
