import time
import pkg_resources
import xml.etree.ElementTree as etxml
from datetime import datetime
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gate_config import gate_positions, gate_orientations

class RacingEnv(gym.Env):
    def __init__(self,
                 urdf,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 output_folder='results'
                 ):
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        self.IMG_RES = [64, 64]
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError(
                "[ERROR] pyb_freq is not divisible by ctrl_freq."
            )
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1.0 / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1.0 / self.PYB_FREQ
        self.GUI = gui
        self.URDF = urdf
        self.OUTPUT_FOLDER = output_folder

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
        self.WEIGHT = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.WEIGHT / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.WEIGHT) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        self.INIT_XYZ = [0.0, 0.0, 1.0]
        self.INIT_RPYS = [0.0, 0.0, 0.0]
        

        #### Connect to PyBullet ###################################
        if self.GUI:
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
        else:
            self.CLIENT = p.connect(p.DIRECT)
            # p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)

        #### Create action and observation spaces ##################
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.IMG_RES[1], self.IMG_RES[0]),
            dtype=np.uint8,
        )

        self._housekeeping()
        self._updateAndStoreKinematicInformation()

    def reset(self, seed=None, options=None):
        p.resetSimulation(physicsClientId=self.CLIENT)
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info

    def step(self, action):
        for _ in range(self.PYB_STEPS_PER_CTRL):
            if self.PYB_STEPS_PER_CTRL > 1:
                self._updateAndStoreKinematicInformation()
            self._PyBulletDynamics(action)
            p.stepSimulation(physicsClientId=self.CLIENT)

        self.last_action = action
        self._updateAndStoreKinematicInformation()

        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + self.PYB_STEPS_PER_CTRL
        return obs, reward, terminated, truncated, info

    def render(self,
               mode='human',
               close=False
               ):
        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
        print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
              "——— wall-clock time {:.1f}s,".format(time.time()-self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter*self.PYB_TIMESTEP, self.PYB_FREQ, (self.step_counter*self.PYB_TIMESTEP)/(time.time()-self.RESET_TIME)))
        for i in range (self.NUM_DRONES):
            print("[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                  "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                  "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                  "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0]*self.RAD2DEG, self.rpy[i, 1]*self.RAD2DEG, self.rpy[i, 2]*self.RAD2DEG),
                  "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i, 2]))

    def close(self):
        p.disconnect(physicsClientId=self.CLIENT)

    def _housekeeping(self):
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1
        self.Y_AX = -1
        self.Z_AX = -1
        self.last_input_switch = 0
        self.last_action = np.zeros(4)

        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros(3)
        self.quat = np.zeros(4)
        self.rpy = np.zeros(3)
        self.vel = np.zeros(3)
        self.ang_v = np.zeros(3)
        self.rpy_rates = np.zeros(3)

        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)

        #### Load ground plane, drone and obstacles models #########
        self.GATE_IDS = []
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        self.DRONE_ID = p.loadURDF(
            self.URDF,
            self.INIT_XYZ,
            p.getQuaternionFromEuler(self.INIT_RPYS),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.CLIENT,
        )
        for position, orientation in zip(gate_positions, gate_orientations):
            self.GATE_IDS.append(p.loadURDF("./models/gate/gate.urdf", position, orientation, useFixedBase=True))

        #### Remove default damping #################################
        # p.changeDynamics(self.DRONE_ID, -1, linearDamping=0, angularDamping=0)
        if self.GUI:
            self._showDroneLocalAxes()

    def _updateAndStoreKinematicInformation(self):
        self.pos, self.quat = p.getBasePositionAndOrientation(self.DRONE_ID, physicsClientId=self.CLIENT)
        self.rpy = p.getEulerFromQuaternion(self.quat)
        self.vel, self.ang_v = p.getBaseVelocity(self.DRONE_ID, physicsClientId=self.CLIENT)

    def _getImages(self):
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat)).reshape(3, 3)
        target = np.dot(rot_mat,np.array([1000, 0, 0])) + np.array(self.pos)
        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos+np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[0, 0, 1],
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=self.L,
                                                      farVal=1000.0
                                                      )
        [w, h, rgb, dep, seg] = p.getCameraImage(width=self.IMG_RES[0],
                                                 height=self.IMG_RES[1],
                                                 shadow=0,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                 physicsClientId=self.CLIENT
                                                 )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    def _PyBulletDynamics(self, rpm):
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
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
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2*np.pi*rpm/60))
        drag = np.dot(base_rot.T, drag_factors*np.array(self.vel))

        p.applyExternalForce(self.DRONE_ID,
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )

    def _PythonDynamics(self, rpm):
        #### Current state #########################################
        pos = self.pos
        quat = self.quat
        vel = self.vel
        rpy_rates = self.rpy_rates
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        #### Compute forces and torques ############################
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.WEIGHT])
        z_torques = np.array(rpm**2)*self.KM
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L/np.sqrt(2))
        y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L/np.sqrt(2))
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M

        #### Update state ##########################################
        vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv
        pos = pos + self.PYB_TIMESTEP * vel
        quat = self._integrateQ(quat, rpy_rates, self.PYB_TIMESTEP)
        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(self.DRONE_ID,
                                          pos,
                                          quat,
                                          physicsClientId=self.CLIENT
                                          )
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(self.DRONE_ID,
                            vel,
                            np.dot(rotation, rpy_rates),
                            physicsClientId=self.CLIENT
                            )
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates = rpy_rates

    def _integrateQ(self, quat, omega, dt):
        omega_norm = np.linalg.norm(omega)
        p, q, r = omega
        if np.isclose(omega_norm, 0):
            return quat
        lambda_ = np.array([
            [ 0,  r, -q, p],
            [-r,  0,  p, q],
            [ q, -p,  0, r],
            [-p, -q, -r, 0]
        ]) * .5
        theta = omega_norm * dt / 2
        quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
        return quat

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

    def _computeObs(self):
        self.rgb, self.dep, self.seg = self._getImages()
        gate_mask = np.zeros_like(self.seg)
        for gate_id in self.GATE_IDS:
            gate_mask += (self.seg == gate_id)
        return gate_mask.astype('float32')

    def _preprocessAction(self, action):
        return action

    def _computeReward(self):
        return 0.0

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

if __name__ == "__main__":
    env = RacingEnv("./models/cf2/cf2x.urdf", gui=True)
    obs, info = env.reset()
    for i in range(1000):
        print(i)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    env.close()
