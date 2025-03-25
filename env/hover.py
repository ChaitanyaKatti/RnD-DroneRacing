from env.base import BaseEnv
import pybullet as p
import numpy as np
import pygame
import time

gate_positions = [
    [1.5, -1.2, 1],
    [1.5, 1.2, 1],
]

gate_orientations = [
    p.getQuaternionFromEuler([0, 0, 0]),
    p.getQuaternionFromEuler([0, 0, 0]),
]

class HoverEnv(BaseEnv):
    def __init__(
        self,
        URDF="./assets/cf2/cf2x.urdf",
        pyb_substeps: int = 8,
        ctrl_freq: int = 30,
        gui=False,
    ):
        super(HoverEnv, self).__init__(URDF, pyb_substeps, ctrl_freq, gui)
        self.next_gate_id = None

    def reset(self, seed=None, options=None):
        # self.INIT_XYZ = np.random.uniform([-1.0, -1.0, 1.0], [1.0, 1.0, 1.0])
        self.next_gate_id = self.GATE_IDS[0]
        return super(HoverEnv, self).reset(seed, options)

    def _addObstacles(self):
        super(HoverEnv, self)._addObstacles()
        self.GATE_IDS = []
        for position, orientation in zip(gate_positions, gate_orientations):
            self.GATE_IDS.append(p.loadURDF("./assets/gate/gate.urdf", position, orientation, useFixedBase=True))

    def _computeObs(self):
        self.rgb, self.dep, self.seg = self._getImages()
        self.rgb = self.rgb / 255.0
        self.seg = 255*np.where(self.seg < 1, 0, 1).astype(np.uint8)

        next_gate_rel_pos = np.array(p.getBasePositionAndOrientation(self.next_gate_id)[0]) - self.pos
        # Transform next_gate_rel_pos to drone frame
        next_gate_rel_pos = -np.array(p.invertTransform(next_gate_rel_pos, p.getQuaternionFromEuler(self.rpy))[0])

        return {
            "img": np.reshape(self.seg, (1, 84, 84)),
            "last_3_actions": self.last_3_actions,
            "kinematics": np.stack([self.pos, self.rpy, self.vel, self.ang_v, next_gate_rel_pos])
        }

    def _computeReward(self):
        reward = 0.01
        reward -= 0.02*np.linalg.norm(self.pos - np.array([0, 0, 1]))
        return reward

    def _computeTerminated(self):
        if np.linalg.norm(self.pos - np.array([0, 0, 1])) > 1.0: # Out of bounds
            return True
        return False

    def _computeTruncated(self):
        # Check if time limit is reached
        if self.step_counter >= 5 / self.PYB_TIMESTEP: # 5 seconds, 5*30 = 150 steps
            return True
        return False

    def _computeInfo(self):
        return {}


if __name__ == "__main__":
    env = HoverEnv(gui=True)
    pygame.init()
    pygame.joystick.init()

    try:
        joystick = pygame.joystick.Joystick(0)
    except:
        print("No Joystick Connected")
        pygame.quit()
        exit()
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
            # print(f"Reward: {reward:.4f} Info: {info}")
            if joystick.get_axis(4)>0:
                terminated = True
            time.sleep(0.01)
        print(f"Episode {i} Total Reward: {total_reward:.4f}")

    env.close()
    pygame.quit()
