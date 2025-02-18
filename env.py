import numpy as np
import gymnasium as gym
from gymnasium import spaces
from renderer.renderer import DroneRenderer
from utils import quat_to_rot, quat_to_euler, dynamicsJIT
import time
from constants import *

class QuadrotorEnv(gym.Env):
    """Custom Quadrotor environment compatible with Gymnasium."""

    def __init__(self):
        super().__init__()
        self.mass = 0.027
        self.gravity = 9.81
        self.arm_length = 0.046
        self.maxF = 0.5886  # Maximum thrust 60grams * 9.81m/s^2 = 0.5886 N
        self.minF = 0.0 # Minimum thrust
        self.I = np.diag([2.3951*10e-5, # Ixx 
                          2.3951*10e-5,   # Iyy
                          3.2347*10e-5])  # Izz
        self.invI = np.linalg.inv(self.I)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)  # 3 for position, 3 for velocity, 9 for rotation, 3 for angular rates
        self.action_space = spaces.Box(low=np.array([0, -1, -1, -1]),
                                       high=np.array([1, 1, 1, 1]), dtype=np.float32)  # 1 for thrust, 3 for RPY rates
        self.state = None
        self.reset()
        self.step_count = 0
        self.renderer = None
        self._next_step_time = time.perf_counter()
    
    def trajectory(self, t):
        return np.array([2*np.cos(t * 2 * np.pi), 
                         2*np.sin(t * 2 * np.pi), 
                         0.0])
    def project_trajectory(self):
        '''project x, y, z onto the nearest point on the trajectory'''
        x, y = self.state[0], self.state[1]
        return 2 * np.array([x, y, 0]) / np.linalg.norm([x, y])
    def trajectory_derivative(self):
        '''Project x, y, z onto the nearest point on the trajectory and return the derivative'''
        x, y = self.state[0], self.state[1]
        return np.array([-y, x, 0]) / np.linalg.norm([x, y])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = np.zeros(13)
        self.state[:3] = self.trajectory(np.random.uniform(0, 1))  + np.random.uniform(-0.1, 0.1, 3)  # Initial position
        self.state[10:13] = np.random.uniform(-0.1, 0.1, 3)  # Initial angular rates
        self.state[6] = 1.0  # Initial quaternion (w=1, i.e., no rotation)
        obs = np.concatenate(
            [
                self.state[:3],  # Position
                self.state[3:6],  # Velocity
                quat_to_rot(self.state[6:10]).flatten(),  # Rotation matrix
                self.state[10:13],  # Angular rates
            ]
        )
        return obs, {}

    def rate_controller(self, state, target_rates):
        """
        Rate controller using a PD law with feed-forward compensation.
        target_rates: desired body rates (rad/s) provided in the body frame.
        state: current state vector.
        """
        omega = state[10:13]  # current body rates [p, q, r]
        # PD gains
        kp = np.diag([4.0, 4.0, 4.0])
        kd = np.diag([0.5, 0.5, 0.5])
        # Feed-forward compensation for the gyroscopic (cross product) term:
        M_ff = np.cross(omega, self.I @ omega)
        # PD term: the desired angular acceleration (here taken as 0) is implicit.
        M_pd = self.I @ (kp @ (target_rates - omega) - kd @ omega)
        return M_ff + M_pd

    def dynamics(self, state, thrust, target_rates):
        """
        Computes the derivative of the state vector.
        state: current state vector (13-D)
        thrust: scalar thrust command (in Newtons), assumed along the body z-axis.
        target_rates: desired body rates (rad/s) in the body frame.
        """
        # Unpack state
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        qw, qx, qy, qz = state[6:10]
        p, q, r = state[10:13]
        quat = np.array([qw, qx, qy, qz])

        # Compute the rotation matrix from body to world frame.
        # (quat_to_rot returns a rotation from world to body, so we take the transpose)
        R = quat_to_rot(quat).T

        # Compute linear acceleration.
        # Thrust is applied along the body z-axis.
        a = (1 / self.mass) * (R @ np.array([0, 0, thrust]) - np.array([0, 0, self.mass * self.gravity]))

        # Quaternion dynamics.
        # To help preserve unit norm, add a correction term.
        K_quat = 2.0
        quat_error = 1 - np.sum(quat ** 2)
        qdot = -0.5 * np.array([
            [0,  -p,  -q,  -r],
            [p,   0,  -r,   q],
            [q,   r,   0,  -p],
            [r,  -q,   p,   0]
        ]) @ quat + K_quat * quat_error * quat

        # Compute control moments using the improved rate controller.
        M = self.rate_controller(state, target_rates)

        # Angular acceleration using the rotational dynamics.
        omega = np.array([p, q, r])
        domega = self.invI @ (M - np.cross(omega, self.I @ omega))

        # Assemble the time derivative of the state vector.
        dstatedt = np.zeros_like(state)
        dstatedt[0:3] = np.array([vx, vy, vz])  # velocity
        dstatedt[3:6] = a                       # acceleration
        dstatedt[6:10] = qdot                   # quaternion derivative
        dstatedt[10:13] = domega                # angular acceleration

        return dstatedt
    
    def step(self, action, dt = CTRL_DT):
        """
        Advances the simulation by one time step using an RK4 integrator.
        action: a 4-element array with [normalized thrust, body rate commands...]
                Note: body rate commands (n_target_p, n_target_q, n_target_r) are normalized,
                and are scaled to rad/s inside the method.
        """
        if self.renderer is not None: # Control the simulation speed for rendering
            time_to_wait = self._next_step_time - time.perf_counter()
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            self._next_step_time = time.perf_counter() + dt
        
        # Unpack the action
        n_thrust, n_target_p, n_target_q, n_target_r = action # Normalized thrust and body rate commands
        thrust = (self.maxF - self.minF) * n_thrust + self.minF

        # Convert normalized rate commands into target body rates (rad/s).
        target_rates = 2.0*np.array([n_target_p, n_target_q, n_target_r])

        dt = 1.0/self.renderer.fps if self.renderer is not None else dt

        state = self.state.copy()

        # Runge–Kutta 4 (RK4) integration
        k1 = dynamicsJIT(state, thrust, target_rates)
        k2 = dynamicsJIT(state + 0.5 * dt * k1, thrust, target_rates)
        k3 = dynamicsJIT(state + 0.5 * dt * k2, thrust, target_rates)
        k4 = dynamicsJIT(state + dt * k3, thrust, target_rates)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize the quaternion to prevent drift.
        new_state[6:10] /= np.linalg.norm(new_state[6:10])
        self.state = new_state
        self.step_count += 1
        
        # Construct a 18-dimensional observation:
        # [position (3), velocity (3), rot matrix (9), angular rates (3)]
        obs = np.concatenate([self.state[:3], 
                              self.state[3:6],
                              quat_to_rot(self.state[6:10]).flatten(),
                              self.state[10:13]])
        reward, done, truncated, info = self.compute_reward()

        return obs, reward, done, truncated, info

    def compute_reward(self):
        projected = self.project_trajectory()
        reward  = 0.001
        reward -= 0.001 * np.linalg.norm(self.state[0:3] - projected)
        reward += 0.001 * np.dot(self.state[3:6], self.trajectory_derivative()) / np.linalg.norm(self.state[3:6])
        reward -= 0.001 * np.linalg.norm(self.state[3:6] - self.trajectory_derivative())
        done = False
        truncated = False
        info = {}

        if self.state[2] < -2.0:
            truncated = True
            info['done_reason'] = 'Crash'
            reward = -5.0
        if self.step_count > EPISODE_LENGTH:
            done = True
            info['done_reason'] = 'Episode length exceeded'

        return reward, done, truncated, info

    def render(self):
        if self.renderer is None:
            self.renderer = DroneRenderer()
        if not self.renderer.should_close():
            self.renderer.render(self.state[:3], quat_to_rot(self.state[6:10]).T)

    def close(self):
        if self.renderer is not None:
            self.renderer.cleanup()
