import numpy as np
from typing import Callable
from numba import njit

def quat_to_rot(quat):
    """Convert a quaternion to a rotation matrix."""
    qw, qx, qy, qz = quat
    # Compute the rotation matrix elements.
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def quat_to_euler(quat):
    """Convert a quaternion to Euler angles."""
    qw, qx, qy, qz = quat
    # Compute the Euler angles.
    phi = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    theta = np.arcsin(2*(qw*qy - qz*qx))
    psi = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    return np.array([phi, theta, psi]) # roll, pitch, yaw


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

@njit
def dynamicsJIT(state, thrust, target_rates):
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
    I = np.diag(np.array([2.3951 * 10e-5, # Ixx 
                          2.3951*10e-5,   # Iyy
                          3.2347*10e-5]))  # Izz
    invI = np.linalg.inv(I)
    mass = 0.027
    gravity = 9.81
    
    # Compute the rotation matrix from body to world frame.
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ]).T

    # Compute linear acceleration.
    # Thrust is applied along the body z-axis.
    a = (1 / mass) * (R @ np.array([0, 0, thrust]) - np.array([0, 0, mass*gravity]))

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
    omega = state[10:13]  # current body rates [p, q, r]
    # PD gains
    kp = np.array([20.0, 20.0, 20.0])
    # PD term: the desired angular acceleration (here taken as 0) is implicit.
    M_pd = I @ (kp * (target_rates - omega))
    M = M_pd

    # Angular acceleration using the rotational dynamics.
    omega = np.array([p, q, r])
    domega = invI @ (M - np.cross(omega, I @ omega))

    # Assemble the time derivative of the state vector.
    dstatedt = np.zeros_like(state)
    dstatedt[0:3] = np.array([vx, vy, vz])  # velocity
    dstatedt[3:6] = a                       # acceleration
    dstatedt[6:10] = qdot                   # quaternion derivative
    dstatedt[10:13] = domega                # angular acceleration

    return dstatedt
