import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

NUM_IMAGES = 1000

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
fov = 115  # Field of view
aspect = width / height
near = 0.1
far = 1000
proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Create folder
os.makedirs("./dataset/rgb", exist_ok=True)
os.makedirs("./dataset/seg", exist_ok=True)
seg_means = np.zeros(NUM_IMAGES)

def get_random_matrix():
    pos = np.random.uniform(-1, 1, 3) # Random Position Vector
    pos = pos * np.array([10, 10, 1]) + np.array([0, 0, 1]) # Scale and shift
    rpy = np.random.uniform(-1, 1, 3) # Random Euler angles
    rpy = rpy * np.array([np.pi, np.pi/2, np.pi]) # Scale
    orn = p.getQuaternionFromEuler(rpy)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    transform_matrix[3, :3] = -np.dot(pos, transform_matrix[:3, :3])
    iDontKnowHowIGotThisButItWorks = np.array([[0, 0, -1, 0],
                                               [-1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, 1]])
    transform_matrix = np.dot(transform_matrix, iDontKnowHowIGotThisButItWorks)
    return transform_matrix.flatten()

i = 0
while i < NUM_IMAGES:
    view_matrix = get_random_matrix()

    # Render segmentation mask
    _, _, rgb_img, _, segmentation_mask = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    )
    rgb_img = cv2.cvtColor(np.array(rgb_img, dtype=np.uint8), cv2.COLOR_RGBA2BGR)

    # Match all the gate ids
    seg_img = np.array(segmentation_mask, dtype=np.uint8)
    gate_mask = np.zeros_like(seg_img)
    for gate_id in gate_ids:
        gate_mask += (seg_img == gate_id)

    # Rejection Sampling to improve distribution
    mean = np.mean(gate_mask)
    p_target = np.exp(-0.5*((mean-0.1)/0.05)**2)
    p_current = 70/(10000*mean**3 + 8000*mean**2 + 40*mean**0.5 + 1) # Previously calculated distribution
    if np.random.rand() < p_target/p_current:
        seg_means[i] = mean
        print(f"Image {i}/{NUM_IMAGES}")
        gate_mask = gate_mask * 255
        gate_mask = cv2.cvtColor(gate_mask, cv2.COLOR_GRAY2BGR)

        # Save images
        cv2.imwrite(f"./dataset/rgb/rgb_{i}.png", rgb_img)
        cv2.imwrite(f"./dataset/seg/seg_{i}.png", gate_mask)
        
        # Increment counter
        i += 1
    else:
        print(f"Rejected image {i}")


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