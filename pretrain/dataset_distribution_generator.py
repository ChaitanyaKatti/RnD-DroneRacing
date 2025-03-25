import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from trajectory import trajectory

NUM_IMAGES = 10000

# Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load scene objects
plane_id = p.loadURDF("plane.urdf")
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
gate_ids = []
for position, orientation in zip(gate_positions, gate_orientations):
    gate_ids.append(p.loadURDF("./models/gate/gate.urdf", position, orientation, useFixedBase=True))


# Define camera parameters
width, height = 384, 384
fov = 110  # Field of view
aspect = width / height
near = 0.1
far = 1000
proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Create folder
os.makedirs("./dataset/rgb", exist_ok=True)
os.makedirs("./dataset/seg", exist_ok=True)
seg_means = np.zeros(NUM_IMAGES)

def get_random_matrix():
    i = np.random.randint(0, len(trajectory))
    pos, orn = trajectory[i] # Vector and Quaternion

    # Randomise position about the trajectory
    pos = pos + np.random.uniform(-1, 1, 3)
    # Randomise orientation about the trajectory
    euler = p.getEulerFromQuaternion(orn)
    euler_perturbed = [angle + np.random.uniform(-0.8, 0.8) for angle in euler]
    orn = p.getQuaternionFromEuler(euler_perturbed)

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    transform_matrix[3, :3] = -np.dot(pos, transform_matrix[:3, :3])
    iDontKnowHowIGotThisButItWorks = np.array([[0, 0, -1, 0],
                                               [-1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, 1]])
    transform_matrix = np.dot(transform_matrix, iDontKnowHowIGotThisButItWorks)
    return transform_matrix.flatten()


for i in tqdm(range(NUM_IMAGES)):
    view_matrix = get_random_matrix()

    # Render segmentation mask
    _, _, _, _, segmentation_mask = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    )

    # Match all the gate ids
    seg_img = np.array(segmentation_mask, dtype=np.uint8)
    gate_mask = np.zeros_like(seg_img)
    for gate_id in gate_ids:
        gate_mask += (seg_img == gate_id)

    seg_means[i] = np.mean(gate_mask)


p.disconnect()
cv2.destroyAllWindows()

# Save segmentation distribution
np.save(f"./dataset/seg_means_{NUM_IMAGES}.npy", seg_means)
# Find mean and std of segmentation masks
mean = np.mean(seg_means)
std = np.std(seg_means)
print(f"Mean: {mean}, Std: {std}")
# Plot bar chart of frequency of sums
plt.hist(seg_means, bins=50)
plt.xlabel("Sum of pixel values")
plt.ylabel("Frequency")
plt.title("Histogram of segmentation mask pixel values")
plt.show()