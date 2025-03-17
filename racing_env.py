from env import BaseRL
import pybullet as p
import numpy as np
import pygame
import time

gate_positions = [
    [5, 0, 1], [10, 0, 1]
    # [4, -2.5, 1], [6, 0, 1], [4, 2.5, 1],
    # [-4, -2.5, 1], [-6, 0, 1], [-4, 2.5, 1]
]

gate_orientations = [
    p.getQuaternionFromEuler([0, 0, 0]),
    p.getQuaternionFromEuler([0, 0, 0]),
    # p.getQuaternionFromEuler([0, 0, 0]),
    # p.getQuaternionFromEuler([0, 0, np.pi/2]),
    # p.getQuaternionFromEuler([0, 0, np.pi]),
    # p.getQuaternionFromEuler([0, 0, np.pi]),
    # p.getQuaternionFromEuler([0, 0, np.pi/2]),
    # p.getQuaternionFromEuler([0, 0, 0])
]

class RacingEnv(BaseRL):
    def __init__(
        self,
        URDF="./models/cf2/cf2x.urdf",
        pyb_substeps: int = 8, # Total 8*30 = 240Hz
        ctrl_freq: int = 30,
        gui=False,
    ):
        super(RacingEnv, self).__init__(URDF, pyb_substeps, ctrl_freq, gui)
        self.next_gate_id = None
        
    def reset(self, seed=None, options=None):
        self.INIT_XYZ = np.random.uniform([-1.0, -1.0, 1.0], [1.0, 1.0, 1.0])
        self.next_gate_id = self.GATE_IDS[0]
        return super(RacingEnv, self).reset(seed, options)

    def _addObstacles(self):
        super(RacingEnv, self)._addObstacles()
        self.GATE_IDS = []
        for position, orientation in zip(gate_positions, gate_orientations):
            self.GATE_IDS.append(p.loadURDF("./models/gate/gate.urdf", position, orientation, useFixedBase=True))
        # for gate_id in self.GATE_IDS:
        #     p.setCollisionFilterGroupMask(gate_id, -1, 0, 0)

    def _computeObs(self):
        self.rgb, self.dep, self.seg = self._getImages()
        self.rgb = self.rgb / 255.0
        self.seg = 255*np.where(self.seg < 2, 0, 1).astype(np.uint8)
        
        next_gate_rel_pos = np.array(p.getBasePositionAndOrientation(self.next_gate_id)[0]) - self.pos
        # self.seg = np.zeros((self.IMG_RES, self.IMG_RES))
        return self.seg.reshape(1, self.IMG_RES, self.IMG_RES)
        # return (self.seg, self.last_3_actions, np.concatenate([self.pos, self.rpy, self.vel, self.ang_v, next_gate_rel_pos]))

    def _computeReward(self):
        reward = 0.001
        reward -= 0.001*(self.pos[2] - 1.0)**2
        next_gate_pos = np.array(p.getBasePositionAndOrientation(self.next_gate_id)[0])
        reward -= 0.001*np.linalg.norm(next_gate_pos - self.pos)
        
        # Drone forward vetcor
        forward = np.array(p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.DRONE_ID)[1]))[:3]
        forward = forward / np.linalg.norm(forward)
        # Vector from drone to next gate
        next_gate_rel_pos = next_gate_pos - self.pos
        # Angle between forward and tnext_gate_rel_pos
        reward += 0.001*np.dot(forward, next_gate_rel_pos) / np.linalg.norm(next_gate_rel_pos)
        
        if np.linalg.norm(next_gate_rel_pos) < 0.5:
            reward += 1.0 - np.linalg.norm(next_gate_rel_pos)
            self.next_gate_id = self.GATE_IDS[(self.GATE_IDS.index(self.next_gate_id) + 1) % len(self.GATE_IDS)]
        
        if self._computeTerminated():
            reward -= 1.0
        
        return reward

    def _computeTerminated(self):
        if self.pos[2] < 0.100 or self.pos[2] > 3.0 or np.linalg.norm(self.pos[:2], ord=np.inf) > 10.0: # Out of bounds
            return True
        if np.abs(self.rpy[0]) > np.pi/2 or np.abs(self.rpy[1]) > np.pi/2: # Upside down
            return True
        if len(p.getContactPoints(self.DRONE_ID, -1)) > 0: # Collision with Gate
            return True
        return False

    def _computeTruncated(self):
        # Check if time limit is reached
        if self.step_counter >= 5 / self.PYB_TIMESTEP: # 5 seconds, 5*30*5 = 750 steps
            return True
        return False

    def _computeInfo(self):
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years


if __name__ == "__main__":
    env = RacingEnv(gui=False)
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    for i in range(10):
        obs, info = env.reset()
        time.sleep(0.5)
        terminated, truncated = False, False
        total_reward = 0
        
        while not terminated and not truncated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            thrust = (joystick.get_axis(2) + 1) / 2
            roll = joystick.get_axis(3)
            pitch = joystick.get_axis(1)
            yaw = -joystick.get_axis(0)

            action = np.array([thrust, roll, pitch, yaw])
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            print(f"Reward: {reward:.4f} Info: {info}")
            if joystick.get_axis(4)>0:
                terminated = True
                
        print(f"Episode {i} Total Reward: {total_reward:.4f}")
            # time.sleep(0.05)

    env.close()
    pygame.quit()
