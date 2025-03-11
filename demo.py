import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
from gate_config import gate_positions, gate_orientations, trajectory

# Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

# Load scene objects
plane_id = p.loadURDF("plane.urdf")
# gate_id = p.loadURDF("./models/gate/gate.urdf", [0, 0, 1], useFixedBase=True)
gate_ids = []
for position, orientation in zip(gate_positions, gate_orientations):
    gate_ids.append(p.loadURDF("./models/gate/gate.urdf", position, orientation, useFixedBase=True))

drone_id = p.loadURDF("./models/cf2/cf2x.urdf", [-1, 0, 1], flags=p.URDF_USE_INERTIA_FROM_FILE)

# Define camera parameters
width, height = 384, 384
fov = 115  # Field of view
aspect = width / height
near = 0.1
far = 1000
p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[0,0,1])

def get_transform_matrix(id):
    """Returns the 4x4 transformation matrix in world coordinates."""
    pos, orn = p.getBasePositionAndOrientation(id)
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[3, :3] = -np.dot(pos, rot_matrix) + np.array([-0.2,0,0])
    iDontKnowHowIGotThisButItWorks = np.array([[0, 0, -1, 0],
                                               [-1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, 1]])
    transform_matrix = np.dot(transform_matrix, iDontKnowHowIGotThisButItWorks)
    return transform_matrix.flatten()

# Hide ui
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
i=0

# Render segmentation mask
while True:
    p.stepSimulation()
    time.sleep(1./240.)

    # Get the current viewer camera pose
    cam_info = p.getDebugVisualizerCamera()
    view_matrix = cam_info[2]  # View matrix
    proj_matrix = cam_info[3]  # Projection matrix
    
    
    
    
    
    # Move the drone along the trajectory
    pos, orn = trajectory[i]
    p.resetBasePositionAndOrientation(drone_id, pos, orn)
    i = (i + 1) % len(trajectory)    

    


    # Move the drone
    keys = p.getKeyboardEvents()
    if p.B3G_UP_ARROW in keys:
        p.applyExternalForce(drone_id, -1, [0.1, 0, 0], [0, 0, 0], p.LINK_FRAME)
    if p.B3G_DOWN_ARROW in keys:
        p.applyExternalForce(drone_id, -1, [-0.1, 0, 0], [0, 0, 0], p.LINK_FRAME)
    if p.B3G_LEFT_ARROW in keys:
        p.applyExternalForce(drone_id, -1, [0, 0.1, 0], [0, 0, 0], p.LINK_FRAME)
    if p.B3G_RIGHT_ARROW in keys:
        p.applyExternalForce(drone_id, -1, [0, -0.1, 0], [0, 0, 0], p.LINK_FRAME)
    if p.B3G_SPACE in keys:
        p.applyExternalForce(drone_id, -1, [0, 0, 0.1], [0, 0, 0], p.LINK_FRAME)
    if p.B3G_SHIFT in keys:
        p.applyExternalForce(drone_id, -1, [0, 0, -0.1], [0, 0, 0], p.LINK_FRAME)

    view_matrix = np.array(get_transform_matrix(drone_id))
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)


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
    gate_mask = gate_mask * 255
    gate_mask = cv2.cvtColor(gate_mask, cv2.COLOR_GRAY2BGR)

    combined_img = cv2.vconcat([rgb_img, gate_mask])

    cv2.imshow("RGB & Segmentation Mask", combined_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

p.disconnect()
cv2.destroyAllWindows()
