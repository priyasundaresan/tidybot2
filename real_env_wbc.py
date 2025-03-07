# Author: Jimmy Wu
# Date: October 2024

from cameras import KinovaCamera, LogitechCamera
from constants import BASE_RPC_HOST, BASE_RPC_PORT, ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import BASE_CAMERA_SERIAL
from arm_server_wbc import ArmManager
from base_server import BaseManager
from wbc_ik_solver import IKSolver
from scipy.spatial.transform import Rotation as R
import numpy as np

class RealEnv:
    def __init__(self):
        # RPC server connection for base
        base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            base_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to base RPC server, is base_server.py running?') from e

        # RPC server connection for arm
        arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            arm_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e

        # RPC proxy objects
        self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
        self.arm = arm_manager.Arm()

        # Cameras
        self.base_camera = LogitechCamera(BASE_CAMERA_SERIAL)
        self.wrist_camera = KinovaCamera()

        self.wbc_ik_solver = IKSolver()
        self.RESET_QPOS = np.array([0., 0., 0., 0., -0.34906585, 3.14159265, -2.54818071, 0., -0.87266463, 1.57079633, 0., 0., 0., 0., 0., 0., 0., 0.])

        self.arm_base_offset = [0.1199, 0, 0.3948] # arm is forward (0.1199m) and raised by base height (0.3948m)

    def get_obs(self):
        obs = {}
        obs.update(self.base.get_state())
        obs.update(self.arm.get_state())
        obs['base_image'] = self.base_camera.get_image()
        obs['wrist_image'] = self.wrist_camera.get_image()
        return obs

    def reset(self):
        print('Resetting base...')
        self.base.reset()

        print('Resetting arm...')
        self.arm.reset()

        self.wbc_ik_solver.configuration.update(self.RESET_QPOS)
        print('Robot has been reset')

    def step(self, action):
        qpos_base = self.base.get_state()['base_pose']
        qpos_arm = self.arm.get_qpos()

        #if action['teleop_mode'] == 'arm':
        if 'arm_pos' in action:
            T_action = np.eye(4)
            T_action[:3, :3] = R.from_euler('z', qpos_base[2]).as_matrix()
            T_action[:3, 3] = np.array([qpos_base[0], qpos_base[1], 0]) + self.arm_base_offset
            arm_pos_adjusted = T_action@np.array([action['arm_pos'][0], action['arm_pos'][1], action['arm_pos'][2], 1.0])
            arm_pos_adjusted = arm_pos_adjusted[:3]
            action['arm_pos'] = arm_pos_adjusted

            action_qpos = self.wbc_ik_solver.solve(action['arm_pos'], \
                                                   action['arm_quat'], \
                                                   np.hstack([qpos_base, qpos_arm, np.zeros(8)]))
            action_base_pose = action_qpos[:3]
            action_arm_qpos = action_qpos[3:10]
            action_arm_qpos = action_arm_qpos % (2 * np.pi) # Unwrapping
            action['base_pose'] = action_base_pose
            action['arm_qpos'] = action_arm_qpos

        print(action['base_pose'].round(2), action['arm_qpos'].round(2))

        self.base.execute_action(action)  # Non-blocking
        self.arm.execute_action(action)   # Non-blocking

    def close(self):
        self.base.close()
        self.arm.close()
        self.base_camera.close()
        self.wrist_camera.close()

if __name__ == '__main__':
    import time
    import numpy as np
    from constants import POLICY_CONTROL_PERIOD
    env = RealEnv()
