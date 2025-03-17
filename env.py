import time
import xml.etree.ElementTree as etxml
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium.spaces import Box, Tuple
import cv2
from constants import *

class BaseRL(gym.Env):
    def __init__(
        self,
        URDF="./models/cf2/cf2x.urdf",
        pyb_substeps: int = 1,
        ctrl_freq: int = 30,
        gui=False,
    ):
        #### Constants #############################################
        self.URDF = URDF
        self.G = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.IMG_RES = IMG_RES
        self.GUI = gui

        self.CTRL_FREQ = ctrl_freq
        self.PYB_SUBSTESPS = pyb_substeps
        self.CTRL_TIMESTEP = 1.0 / self.CTRL_FREQ
        self.PYB_TIMESTEP = self.CTRL_TIMESTEP

        #### Load the drone properties from the .urdf file #########
        (
            self.M,
            self.L,
            self.THRUST2WEIGHT_RATIO,
            self.J,
            self.J_INV,
            self.KF,
            self.KM,
            self.COLLISION_H,
            self.COLLISION_R,
            self.COLLISION_Z_OFFSET,
            self.MAX_SPEED_KMH,
            self.GND_EFF_COEFF,
            self.PROP_RADIUS,
            self.DRAG_COEFF,
            self.DW_COEFF_1,
            self.DW_COEFF_2,
            self.DW_COEFF_3,
        ) = self._parseURDFParameters()
        print("[INFO] RacingEnv.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))

        #### Compute constants #####################################
        self.WEIGHT = 9.8 * self.M
        self.HOVER_RPM = np.sqrt(self.WEIGHT / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.WEIGHT) / (4 * self.KF))
        self.MAX_THRUST = 4 * self.KF * self.MAX_RPM**2
        self.INIT_XYZ = [0.0, 0.0, 1.0]
        self.INIT_RPYS = [0.0, 0.0, 0.0]

        #### Camera parameters #####################################
        self.FPV_ANGLE = 25.0
        self.FOV = 110.0
        self.NEAR = 0.01
        self.FAR = 100.0
        self.rgb, self.dep, self.seg = None, None, None

        #### Control parameters ####################################
        self.RATE_SCALE = np.array([2.0, 2.0, 5.0])
        self.I_GAIN = 0.0
        self.P_GAIN = 7000

        #### Create action and observation spaces ##################
        # Action consists of thrust, roll rate, pitch rate, yaw rate
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        # Obs consists of img, past 3 actions, and kinematic info
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, self.IMG_RES, self.IMG_RES), dtype=np.uint8)
        # self.observation_space = gym.spaces.Tuple(
        #     (gym.spaces.Box(low=0, high=1, shape=(self.IMG_RES, self.IMG_RES), dtype=np.float32),
        #      gym.spaces.Box(low=-1, high=1, shape=(3, 4), dtype=np.float32),
        #      gym.spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32))
        # )
        self.last_3_actions = np.zeros((3, 4))

        self._initializePyBullet()
        self._housekeeping()
        self._addDrone()
        self._addObstacles()
        self._updateAndStoreKinematicInformation()

    def reset(self, seed=None, options=None):
        p.resetBasePositionAndOrientation(
            self.DRONE_ID,
            self.INIT_XYZ,
            p.getQuaternionFromEuler(self.INIT_RPYS),
            physicsClientId=self.CLIENT,
        )
        p.resetBaseVelocity(
            self.DRONE_ID, [0, 0, 0], [0, 0, 0], physicsClientId=self.CLIENT
        )
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info

    def step(self, action):
        rpm = self._rateController(action)

        self._PyBulletDynamics(rpm)
        p.stepSimulation(physicsClientId=self.CLIENT)

        self.last_3_actions = np.roll(self.last_3_actions, 1, axis=0)
        self.last_3_actions[0] = action
        self._updateAndStoreKinematicInformation()

        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        #### Advance the step counter ##############################
        self.step_counter += 1
        return obs, reward, terminated, truncated, info

    def render(self, mode="human", close=False):
        combined = cv2.vconcat([cv2.cvtColor((255*self.rgb).astype(np.uint8), cv2.COLOR_RGBA2BGR), 
                                cv2.cvtColor((self.seg).astype(np.uint8), cv2.COLOR_GRAY2BGR)])        
        cv2.imshow("DroneCam", combined)
        cv2.waitKey(1)

    def close(self):
        p.disconnect(physicsClientId=self.CLIENT)
        cv2.destroyAllWindows()

    def _initializePyBullet(self):
        if self.GUI:
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]: # Disable all debug visualizations
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(
                cameraDistance=3,
                cameraYaw=-30,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.CLIENT,
            )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
        else:
            self.CLIENT = p.connect(p.DIRECT)
            # p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)

        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.CLIENT
        )
        p.setPhysicsEngineParameter(fixedTimeStep=self.PYB_TIMESTEP, numSubSteps=self.PYB_SUBSTESPS, physicsClientId=self.CLIENT)

    def _addDrone(self):
        self.DRONE_ID = p.loadURDF(
            self.URDF,
            self.INIT_XYZ,
            p.getQuaternionFromEuler(self.INIT_RPYS),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.CLIENT,
        )
        if self.GUI:
            self._showDroneLocalAxes()
        #### Remove default damping #################################
        # p.changeDynamics(self.DRONE_ID, -1, linearDamping=0, angularDamping=0)

    def _addObstacles(self):
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

    def _housekeeping(self):
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1
        self.Y_AX = -1
        self.Z_AX = -1
        self.last_input_switch = 0
        self.last_3_actions = np.zeros((3, 4))

        #### Initialize the drones kinemaatic information ##########
        self.pos = np.array(self.INIT_XYZ)
        self.quat = np.array(p.getQuaternionFromEuler(self.INIT_RPYS))
        self.rpy = np.array(self.INIT_RPYS)
        self.vel = np.zeros(3)
        self.ang_v = np.zeros(3)
        self.rpy_rates = np.zeros(3)

        #### Initialize the rate controller variables ##############
        self.rate_error_sum = np.zeros(3)

    def _updateAndStoreKinematicInformation(self):
        self.pos, self.quat = p.getBasePositionAndOrientation(self.DRONE_ID, physicsClientId=self.CLIENT)
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(self.DRONE_ID, physicsClientId=self.CLIENT)
        self.rpy_rates = np.dot(np.array(p.getMatrixFromQuaternion(self.quat)).reshape(3, 3).T, self.ang_v)

    def _getImages(self):
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        cameraPos = self.pos + np.dot(rot_mat, np.array([0.03, 0.0, 0.01]))
        target = cameraPos + np.dot(rot_mat, np.array([1000, 0, 1000*np.tan(self.FPV_ANGLE)]))
        droneUpVector = np.dot(rot_mat, np.array([0, 0, 1]))
        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=cameraPos,
                                             cameraTargetPosition=target,
                                             cameraUpVector=droneUpVector,
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=self.FOV,
                                                      aspect=1.0,
                                                      nearVal=self.NEAR,
                                                      farVal=self.FAR,
                                                      )
        w, h, rgb, dep, seg = p.getCameraImage(width=self.IMG_RES,
                                                 height=self.IMG_RES,
                                                 shadow=0,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=p.ER_NO_SEGMENTATION_MASK,
                                                 physicsClientId=self.CLIENT
                                                 )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    def _PyBulletDynamics(self, rpm):
        forces = np.array(rpm**2) * self.KF
        torques = np.array(rpm**2) * self.KM
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
        for i in range(4):
            p.applyExternalForce(self.DRONE_ID,
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_ID,
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot.T, drag_factors * np.array(self.vel))

        p.applyExternalForce(self.DRONE_ID,
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )

        # # Fix drones position at origin
        # p.resetBasePositionAndOrientation(self.DRONE_ID,
        #                                     self.INIT_XYZ,
        #                                     self.quat,
        #                                     physicsClientId=self.CLIENT
        #                                     )
        # p.resetBaseVelocity(self.DRONE_ID,
        #                     [0, 0, 0],
        #                     np.dot(base_rot, self.rpy_rates),
        #                     physicsClientId=self.CLIENT
        #                     )

    def _showDroneLocalAxes(self):
        AXIS_LENGTH = 2*self.L
        self.X_AX = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                    lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                    lineColorRGB=[1, 0, 0],
                                                    parentObjectUniqueId=self.DRONE_ID,
                                                    parentLinkIndex=-1,
                                                    replaceItemUniqueId=int(self.X_AX),
                                                    physicsClientId=self.CLIENT
                                                    )
        self.Y_AX = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                    lineToXYZ=[0, AXIS_LENGTH, 0],
                                                    lineColorRGB=[0, 1, 0],
                                                    parentObjectUniqueId=self.DRONE_ID,
                                                    parentLinkIndex=-1,
                                                    replaceItemUniqueId=int(self.Y_AX),
                                                    physicsClientId=self.CLIENT
                                                    )
        self.Z_AX = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                    lineToXYZ=[0, 0, AXIS_LENGTH],
                                                    lineColorRGB=[0, 0, 1],
                                                    parentObjectUniqueId=self.DRONE_ID,
                                                    parentLinkIndex=-1,
                                                    replaceItemUniqueId=int(self.Z_AX),
                                                    physicsClientId=self.CLIENT
                                                    )

    def _parseURDFParameters(self):
        URDF_TREE = etxml.parse((self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3

    def _rateController(self, action):
        thrust = action[0] * self.MAX_THRUST

        if thrust < 0.1:
            self.rate_error_sum = np.zeros(3) # Reset the integral term

        desired_rate = self.RATE_SCALE * np.array([action[1], action[2], action[3]])  # roll, pitch, yaw rate
        rate_error = desired_rate - self.rpy_rates
        self.rate_error_sum += rate_error*self.CTRL_TIMESTEP
        torque = np.dot(self.J, self.P_GAIN * rate_error + self.I_GAIN * self.rate_error_sum)

        MixerMatrix = np.array([
            np.array([ 1,  1,  1,  1], dtype=np.float32),                       # Total thrust
            np.array([-1, -1,  1,  1], dtype=np.float32) * self.L/np.sqrt(2),   # Roll torque
            np.array([-1,  1,  1, -1], dtype=np.float32) * self.L/np.sqrt(2),  # Pitch torque
            0.0005*np.array([-1,  1, -1,  1], dtype=np.float32) * self.KF/self.KM     # Yaw torque
        ])/4

        forces = MixerMatrix.T @ np.array([thrust, torque[0], torque[1], torque[2]])
        forces = np.clip(forces, 0, self.MAX_THRUST / 4)  # Thrust Saturation
        rpm = np.sqrt(forces / self.KF)  # Convert to RPM

        # print("Action:", action)
        # print("Thrust:", thrust)
        # print("Error", rate_error)
        # print("Torque:", torque)
        # print('Forces:', forces)
        # print('RPM:', rpm)
        # print()

        return rpm

    def _computeObs(self):
        raise NotImplementedError
    def _computeReward(self):
        raise NotImplementedError
    def _computeTerminated(self):
        raise NotImplementedError
    def _computeTruncated(self):
        raise NotImplementedError
    def _computeInfo(self):
        raise NotImplementedError
