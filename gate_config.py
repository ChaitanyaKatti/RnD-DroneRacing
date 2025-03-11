import pybullet as p
import numpy as np

# Define waypoints (same as your example)
gate_positions = [
    [4, -2.5, 1], [6, 0, 1], [4, 2.5, 1],
    [-4, -2.5, 1], [-6, 0, 1], [-4, 2.5, 1]
]

gate_orientations = [
    p.getQuaternionFromEuler([0, 0, 0]),
    p.getQuaternionFromEuler([0, 0, np.pi/2]),
    p.getQuaternionFromEuler([0, 0, np.pi]),
    p.getQuaternionFromEuler([0, 0, np.pi]),
    p.getQuaternionFromEuler([0, 0, np.pi/2]),
    p.getQuaternionFromEuler([0, 0, 0])
]

# Include start and end positions
setpoints_position = [
    [0, 0, 1], [4, -2.5, 1], [6, 0, 1], [4, 2.5, 1],
    [0, 0, 1],
    [-4, -2.5, 1], [-6, 0, 1], [-4, 2.5, 1], [0, 0, 1]
]
setpoints_orientation = [
    p.getQuaternionFromEuler([0,            np.pi/6,        0]),
    p.getQuaternionFromEuler([-np.pi/4,     np.pi/6,        0]),
    p.getQuaternionFromEuler([-np.pi/4,     0,              np.pi/2]),
    p.getQuaternionFromEuler([-np.pi/4,     np.pi/6,        np.pi]),
    p.getQuaternionFromEuler([0,            np.pi/6,        np.pi]),
    p.getQuaternionFromEuler([np.pi/4,      np.pi/6,        np.pi]),
    p.getQuaternionFromEuler([np.pi/4,      0,              np.pi/2]),
    p.getQuaternionFromEuler([np.pi/4,      np.pi/6,        0]),
    p.getQuaternionFromEuler([0,            np.pi/6,        0])
]

def bezier_curve(P0, P1, P2, P3, t):
    """Compute a point on a cubic Bézier curve at parameter t (0 to 1)."""
    return (1-t)**3 * np.array(P0) + 3*(1-t)**2*t * np.array(P1) + 3*(1-t)*t**2 * np.array(P2) + t**3 * np.array(P3)

def generate_bezier_trajectory(setpoints_position, setpoints_orientation, num_samples=10):
    trajectory = []
    
    for i in range(len(setpoints_position) - 1):
        P0 = np.array(setpoints_position[i])  # Start point
        P3 = np.array(setpoints_position[i+1])  # End point

        # Control points for smooth curve (adjust scaling factor for smoothness)
        P1 = P0 + 0.4 * (P3 - P0)  # First control point
        P2 = P3 - 0.4 * (P3 - P0)  # Second control point

        start_ori = setpoints_orientation[i]
        end_ori = setpoints_orientation[i+1]

        for t in np.linspace(0, 1, num_samples):
            pos = bezier_curve(P0, P1, P2, P3, t)
            ori = p.getQuaternionSlerp(start_ori, end_ori, t)
            trajectory.append((pos, ori))

    return trajectory

def compute_arc_length(trajectory):
    """Compute cumulative arc length along the trajectory."""
    distances = []
    total_distance = 0
    
    for i in range(1, len(trajectory)):
        pos_prev = trajectory[i-1][0]
        pos_curr = trajectory[i][0]
        distance = np.linalg.norm(pos_curr - pos_prev)
        total_distance += distance
        distances.append(total_distance)
    
    return np.array([0] + distances)

def reparametrize_trajectory(trajectory, desired_points=None):
    """
    Reparametrize trajectory for constant velocity using arc length parameterization.
    
    Args:
        trajectory: List of (position, orientation) tuples
        desired_points: Number of points in output trajectory (defaults to input length)
        
    Returns:
        New trajectory with approximately constant velocity
    """
    if desired_points is None:
        desired_points = len(trajectory)
    
    # Compute cumulative arc length
    arc_lengths = compute_arc_length(trajectory)
    total_length = arc_lengths[-1]
    
    # Create uniform spacing based on total arc length
    uniform_distances = np.linspace(0, total_length, desired_points)
    
    # Initialize new trajectory
    new_trajectory = []
    
    # Interpolate positions and orientations at uniform distances
    for target_distance in uniform_distances:
        # Find segment containing target distance
        idx = np.searchsorted(arc_lengths, target_distance)
        if idx == 0:
            new_trajectory.append(trajectory[0])
            continue
        elif idx == len(arc_lengths):
            new_trajectory.append(trajectory[-1])
            continue
            
        # Linear interpolation parameter
        segment_length = arc_lengths[idx] - arc_lengths[idx-1]
        alpha = (target_distance - arc_lengths[idx-1]) / segment_length
        
        # Interpolate position
        pos1, ori1 = trajectory[idx-1]
        pos2, ori2 = trajectory[idx]
        new_pos = pos1 + alpha * (pos2 - pos1)
        
        # Interpolate orientation using Slerp
        new_ori = p.getQuaternionSlerp(ori1, ori2, alpha)
        
        new_trajectory.append((new_pos, new_ori))
    
    return new_trajectory

# Generate smooth trajectory
trajectory = reparametrize_trajectory(generate_bezier_trajectory(setpoints_position, setpoints_orientation, num_samples=40))

# Print number of interpolated points
print(f"Generated trajectory points: {len(trajectory)}")
