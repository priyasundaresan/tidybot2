# Author: Jimmy Wu
# Date: October 2024
#
# Note: This is a basic simulation environment for sanity checking the
# real-world pipeline for teleop and imitation learning. Performance metrics,
# reward signals, and termination signals are not implemented.

import math
import multiprocessing as mp
from multiprocessing import shared_memory
import random
import time
from threading import Thread
import cv2 as cv
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
import mink
from constants import POLICY_CONTROL_PERIOD

class ShmState:
    def __init__(self, existing_instance=None):
        arr = np.empty(3 + 3 + 4 + 1 + 1)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        self.base_pose = self.data[:3]
        self.arm_pos = self.data[3:6]
        self.arm_quat = self.data[6:10]
        self.gripper_pos = self.data[10:11]
        self.initialized = self.data[11:12]
        self.initialized[:] = 0.0

    def close(self):
        self.shm.close()

class ShmImage:
    def __init__(self, camera_name=None, width=None, height=None, existing_instance=None):
        if existing_instance is None:
            self.camera_name = camera_name
            arr = np.empty((height, width, 3), dtype=np.uint8)
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.camera_name = existing_instance.camera_name
            arr = existing_instance.data
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.data.fill(0)

    def close(self):
        self.shm.close()

# Adapted from https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/renderer.py
class Renderer:
    def __init__(self, model, data, shm_image):
        self.model = model
        self.data = data
        self.image = np.empty_like(shm_image.data)

        # Attach to existing shared memory image
        self.shm_image = ShmImage(existing_instance=shm_image)

        # Set up camera
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA.value, shm_image.camera_name)
        width, height = model.cam_resolution[camera_id]
        self.camera = mujoco.MjvCamera()
        self.camera.fixedcamid = camera_id
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # Set up context
        self.rect = mujoco.MjrRect(0, 0, width, height)
        self.gl_context = mujoco.gl_context.GLContext(width, height)
        self.gl_context.make_current()
        self.mjr_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mjr_context)

        # Set up scene
        self.scene_option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, 10000)

    def render(self):
        self.gl_context.make_current()
        mujoco.mjv_updateScene(self.model, self.data, self.scene_option, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(self.rect, self.scene, self.mjr_context)
        mujoco.mjr_readPixels(self.image, None, self.rect, self.mjr_context)
        self.shm_image.data[:] = np.flipud(self.image)

    def close(self):
        self.gl_context.free()
        self.gl_context = None
        self.mjr_context.free()
        self.mjr_context = None

class BaseController:
    def __init__(self, qpos, qvel, ctrl):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl

    def reset(self):
        # Initialize base at origin
        self.qpos[:] = np.zeros(3)
        self.ctrl[:] = self.qpos

    def control_callback(self, base_pose):
        self.ctrl[:] = base_pose

class ArmController:
    def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper

    def reset(self):
        # Initialize arm in "retract" configuration
        self.qpos[:] = np.array([0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633]) # retract
        #self.qpos[:] = np.array([0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633]) # home
        self.ctrl[:] = self.qpos
        self.ctrl_gripper[:] = 0.0

    def control_callback(self, qpos, gripper_pos):
        self.ctrl_gripper[:] = 255.0 * gripper_pos  # fingers_actuator, ctrlrange [0, 255]
        self.ctrl[:] = qpos

class MujocoSim:
    def __init__(self, mjcf_path, command_queue, shm_state, show_viewer=True):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.command_queue = command_queue
        self.show_viewer = show_viewer

        self.model.body_gravcomp[:] = 1.0
        body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        for object_name in ['cube']:
            if object_name in body_names:
                self.model.body_gravcomp[self.model.body(object_name).id] = 0.0

        # Cache references to array slices
        self.base_dofs = base_dofs = self.model.body('base_link').jntnum.item()
        self.arm_dofs = arm_dofs = 7
        self.qpos_base = self.data.qpos[:base_dofs]
        qvel_base = self.data.qvel[:base_dofs]
        ctrl_base = self.data.ctrl[:base_dofs]
        qpos_arm = self.data.qpos[base_dofs:(base_dofs + arm_dofs)]
        qvel_arm = self.data.qvel[base_dofs:(base_dofs + arm_dofs)]
        ctrl_arm = self.data.ctrl[base_dofs:(base_dofs + arm_dofs)]
        self.qpos_gripper = self.data.qpos[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]
        ctrl_gripper = self.data.ctrl[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]
        self.qpos_cube = self.data.qpos[(base_dofs + arm_dofs + 8):(base_dofs + arm_dofs + 8 + 7)]  # 8 for gripper qpos, 7 for cube qpos

        # Controllers
        self.base_controller = BaseController(self.qpos_base, qvel_base, ctrl_base)
        self.arm_controller = ArmController(qpos_arm, qvel_arm, ctrl_arm, self.qpos_gripper, ctrl_gripper)

        # Shared memory state for observations
        self.shm_state = ShmState(existing_instance=shm_state)

        # Variables for calculating arm pos and quat
        site_id = self.model.site('pinch_site').id
        self.site_xpos = self.data.site(site_id).xpos
        self.site_xmat = self.data.site(site_id).xmat
        self.site_quat = np.empty(4)
        self.base_height = self.model.body('gen3/base_link').pos[2]
        self.base_rot_axis = np.array([0.0, 0.0, 1.0])
        self.base_quat_inv = np.empty(4)

        # Tasks and solver setup
        self.configuration = mink.Configuration(self.model)
        self.end_effector_task = mink.FrameTask(
            frame_name="pinch_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        # Base and arm velocity limits
        self.max_base_velocity = np.array([0.5, 0.5, np.pi/2])  # (x, y, yaw)
        self.max_arm_velocity = np.array([math.radians(80)] * 4 + [math.radians(140)] * 3)
        
        # Create a dictionary mapping joint names to velocity limits
        joint_names = [
            "joint_x",
            "joint_y",
            "joint_th",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]
        velocity_limits = {name: limit for name, limit in zip(joint_names, np.concatenate([self.max_base_velocity, self.max_arm_velocity]))}
        self.velocity_limit = mink.VelocityLimit(self.model, velocity_limits)
        self.position_limit = mink.ConfigurationLimit(self.model)

        self.limits = [self.velocity_limit, self.position_limit]

        self.posture_cost = np.zeros((self.model.nv,))
        self.posture_cost[3:] = 1e-3
        self.posture_task = mink.PostureTask(self.model, cost=self.posture_cost)

        self.tasks = [self.end_effector_task, self.posture_task]
        self.solver = "quadprog"
        self.pos_threshold = 1e-4
        self.ori_threshold = 1e-4

        self.max_iters = 20

        self.frequency = 100.0 
        self.rate_limiter = RateLimiter(frequency=self.frequency, warn=False)

        # Reset the environment
        self.reset()

        # Set control callback
        mujoco.set_mjcb_control(self.control_callback)

    def reset(self):
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Reset cube
        self.qpos_cube[:2] += np.random.uniform(-0.1, 0.1, 2)
        theta = np.random.uniform(-math.pi, math.pi)
        self.qpos_cube[3:7] = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
        mujoco.mj_forward(self.model, self.data)

        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()

        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

    def control_callback(self, *_):
        # Check for new command
        command = None if self.command_queue.empty() else self.command_queue.get()

        if command == 'reset':
            self.reset()

        elif command is not None:
            command['base_pose'] = np.zeros(3) # For WBC
           
            T_wt = np.eye(4)
            T_wt[:3,:3] = R.from_quat(command["arm_quat"]).as_matrix()
            T_wt[:3, 3] = command["arm_pos"]
            T_wt = mink.SE3.from_matrix(T_wt)

            self.end_effector_task.set_target(T_wt)

            # IK solving
            for _ in range(self.max_iters):
                vel = mink.solve_ik(
                    self.configuration,
                    self.tasks,
                    1 / self.frequency,
                    self.solver,
                    1e-3,
                    limits=self.limits 
                )
                self.configuration.integrate_inplace(vel, 1 / self.frequency)
                err = self.end_effector_task.compute_error(self.configuration)
                if (
                    np.linalg.norm(err[:3]) <= self.pos_threshold
                    and np.linalg.norm(err[3:]) <= self.ori_threshold
                ):
                    break

            # Apply controls
            arm_action = self.configuration.q[self.base_dofs:(self.base_dofs + self.arm_dofs)]
            base_action = self.configuration.q[:self.base_dofs]
            gripper_action = 1 if (command is None) else command["gripper_pos"]

            # Control callbacks
            self.base_controller.control_callback(base_action)
            self.arm_controller.control_callback(arm_action, gripper_action)

        self.rate_limiter.sleep()

        # Update base pose
        self.shm_state.base_pose[:] = self.qpos_base

        ## Update arm pos
        # self.shm_state.arm_pos[:] = self.site_xpos
        site_xpos = self.site_xpos.copy()
        site_xpos[2] -= self.base_height  # Base height offset
        site_xpos[:2] -= self.qpos_base[:2]  # Base position inverse
        mujoco.mju_axisAngle2Quat(self.base_quat_inv, self.base_rot_axis, -self.qpos_base[2])  # Base orientation inverse
        mujoco.mju_rotVecQuat(self.shm_state.arm_pos, site_xpos, self.base_quat_inv)  # Arm pos in local frame

        # Update arm quat
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)
        # self.shm_state.arm_quat[:] = self.site_quat
        mujoco.mju_mulQuat(self.shm_state.arm_quat, self.base_quat_inv, self.site_quat)  # Arm quat in local frame

        # Update gripper pos
        self.shm_state.gripper_pos[:] = self.qpos_gripper / 0.8  # right_driver_joint, joint range [0, 0.8]

        # Notify reset() function that state has been initialized
        self.shm_state.initialized[:] = 1.0

    def launch(self):
        if self.show_viewer:
            mujoco.viewer.launch(self.model, self.data, show_left_ui=False, show_right_ui=False)

        else:
            # Run headless simulation at real-time speed
            last_step_time = 0
            while True:
                while time.time() - last_step_time < self.model.opt.timestep:
                    time.sleep(0.0001)
                last_step_time = time.time()
                mujoco.mj_step(self.model, self.data)

class MujocoEnv:
    def __init__(self, render_images=True, show_viewer=True, show_images=False):
        self.mjcf_path = 'models/stanford_tidybot/scene_wbc.xml'
        self.render_images = render_images
        self.show_viewer = show_viewer
        self.show_images = show_images
        self.command_queue = mp.Queue(1)

        # Shared memory for state observations
        self.shm_state = ShmState()

        # Shared memory for image observations
        if self.render_images:
            self.shm_images = []
            model = mujoco.MjModel.from_xml_path(self.mjcf_path)
            for camera_id in range(model.ncam):
                camera_name = model.camera(camera_id).name
                width, height = model.cam_resolution[camera_id]
                self.shm_images.append(ShmImage(camera_name, width, height))

        # Start physics loop
        mp.Process(target=self.physics_loop, daemon=True).start()

        if self.render_images and self.show_images:
            # Start visualizer loop
            mp.Process(target=self.visualizer_loop, daemon=True).start()

    def physics_loop(self):
        # Create sim
        sim = MujocoSim(self.mjcf_path, self.command_queue, self.shm_state, show_viewer=self.show_viewer)

        # Start render loop
        if self.render_images:
            Thread(target=self.render_loop, args=(sim.model, sim.data), daemon=True).start()

        # Launch sim
        sim.launch()  # Launch in same thread as creation to avoid segfault

    def render_loop(self, model, data):
        # Set up renderers
        renderers = [Renderer(model, data, shm_image) for shm_image in self.shm_images]

        # Render camera images continuously
        while True:
            start_time = time.time()
            for renderer in renderers:
                renderer.render()
            render_time = time.time() - start_time
            if render_time > 0.1:  # 10 fps
                print(f'Warning: Offscreen rendering took {1000 * render_time:.1f} ms, try making the Mujoco viewer window smaller to speed up offscreen rendering')

    def visualizer_loop(self):
        shm_images = [ShmImage(existing_instance=shm_image) for shm_image in self.shm_images]
        last_imshow_time = time.time()
        while True:
            while time.time() - last_imshow_time < 0.1:  # 10 fps
                time.sleep(0.01)
            last_imshow_time = time.time()
            for i, shm_image in enumerate(shm_images):
                cv.imshow(shm_image.camera_name, cv.cvtColor(shm_image.data, cv.COLOR_RGB2BGR))
                cv.moveWindow(shm_image.camera_name, 640 * i, -100)
            cv.waitKey(1)

    def reset(self):
        self.shm_state.initialized[:] = 0.0
        self.command_queue.put('reset')

        # Wait for state publishing to initialize
        while self.shm_state.initialized == 0.0:
            time.sleep(0.01)

        # Wait for image rendering to initialize (Note: Assumes all zeros is not a valid image)
        if self.render_images:
            while any(np.all(shm_image.data == 0) for shm_image in self.shm_images):
                time.sleep(0.01)

    def get_obs(self):
        arm_quat = self.shm_state.arm_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)

        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        obs = {
            'base_pose': self.shm_state.base_pose.copy(),
            'arm_pos': self.shm_state.arm_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.shm_state.gripper_pos.copy(),
        }
        if self.render_images:
            for shm_image in self.shm_images:
                obs[f'{shm_image.camera_name}_image'] = shm_image.data.copy()
        return obs

    def step(self, action):
        # Note: We intentionally do not return obs here to prevent the policy from using outdated data
        self.command_queue.put(action)

    def close(self):
        self.shm_state.close()
        self.shm_state.shm.unlink()
        if self.render_images:
            for shm_image in self.shm_images:
                shm_image.close()
                shm_image.shm.unlink()

if __name__ == '__main__':
    env = MujocoEnv()
    # env = MujocoEnv(show_images=True)
    # env = MujocoEnv(render_images=False)
    try:
        while True:
            env.reset()
            obs = env.get_obs()

            random_pos = 0.1 * np.random.rand(3) + np.array([0.55, 0.0, 0.4])
            random_quat = np.random.rand(4)
            random_gripper_pos = np.random.rand(1)
            for _ in range(100):
                action = {
                    'base_pose': np.zeros(3), # No base pos, handled by WBC
                    'arm_pos': random_pos + np.random.uniform(-0.05, 0.05, 3), 
                    'arm_quat': random_quat,
                    'gripper_pos': random_gripper_pos,
                }
                env.step(action)
                obs = env.get_obs()
                #print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
                time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        env.close()
