import sys
# TODO: hardcode the path to the flexivrdk
sys.path.insert(0, '/home/ubuntu/my_code/flexiv_rdk/lib_py') # flexiv_rdk 0.9.1
import flexivrdk
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .ik import RobotModel


class FlexivRobot:
    """
    Flexiv Robot Control Class.
    """

    logger_name = "FlexivRobot"

    def __init__(self, addr=["192.168.2.100", "192.168.2.34"], urdf_path="peel_flexiv.urdf"):
        self.mode = flexivrdk.Mode
        self.robot_states = flexivrdk.RobotStates()
        self.robot_addr = addr
        self.log = flexivrdk.Log()

        robot_ip, local_ip = self.robot_addr
        self.robot = flexivrdk.Robot(robot_ip, local_ip)
        # Clear fault on robot server if any
        if self.robot.isFault():
            self.log.info("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            self.robot.clearFault()
            time.sleep(2)
            # Check again
            if self.robot.isFault():
                self.log.info("Fault cannot be cleared, exiting ...")
                return
            self.log.info("Fault on robot server is cleared")
        # Enable the robot, make sure the E-stop is released before enabling
        self.log.info("Enabling ...")
        self.robot.enable()
        # Wait for the robot to become operational
        seconds_waited = 0
        while not self.robot.isOperational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == 10:
                self.log.info("Still waiting for robot to become operational, please check that the robot 1) has no fault, 2) is in [Auto (remote)] mode.")
        self.log.info("Robot is now operational")
        self.robot.setMode(self.mode.NRT_JOINT_POSITION)
        self.joint_limits_low = np.array([-2.7925, -2.2689, -2.9671, -1.8675, -2.9671, -1.3963, -2.9671]) + 0.1
        self.joint_limits_high = np.array([2.7925, 2.2689, 2.9671, 2.6878, 2.9671, 4.5379, 2.9671]) - 0.1
        # TODO: hardcode the init joint angles
        self.init_joints = np.array([1.095040, -0.8206, -0.7206045, 2.2341, 0.1138227, 1.83865, 0.91781])
        
        self.controller = "impedance"

        self.robot_model = RobotModel(urdf_path, link_id=7)

    def home(self):
        self.move_joints([self.init_joints.tolist()], block=True)
    
    def set_zero_ft(self):
        self.robot.setMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.executePrimitive("ZeroFTSensor()")
        while self.robot.isBusy():
            time.sleep(1)
        self.log.info("Zero FT sensor")
        self.robot.setMode(self.mode.NRT_JOINT_POSITION)
    
    def set_mode(self, name='impedance'):
        assert name in ['impedance','hybrid_force']
        if name == "impedance":
            self.robot.setMode(self.mode.NRT_JOINT_POSITION)
            self.controller = 'impedance'
        elif name == "hybrid_force":
            self.robot.stop()
            frame_str = "BASE"
            self.robot.setForceControlFrame(frame_str)
            self.robot.setForceControlAxis([False, False, True, False, False, False])
            self.robot.setMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
            self.robot.resetMaxContactWrench()
            self.controller = 'hybrid_force'
        else:
            raise NotImplementedError


    def move_joints(self, target_pos_list, target_vel=[0.0]*7, target_acc=[0.0]*7, max_vel=[0.1]*7, max_acc=[0.3]*7, block=True):
        if self.controller != "impedance":
            self.set_mode("impedance")
        
        for target_pos in target_pos_list:
            target_pos_clip = np.clip(np.array(target_pos), self.joint_limits_low, self.joint_limits_high).tolist()
            self.robot.sendJointPosition(target_pos_clip, target_vel, target_acc, max_vel, max_acc)
            if block:
                while self.get_delta_q(target_pos) > 0.01:
                    time.sleep(0.01)
    
    def move_hybrid(self, target_list, max_linear_vel=0.1, max_angular_vel=0.2, wait_time=0.2):
        if self.controller != "hybrid_force":
            self.set_mode("hybrid_force")
        
        for i, target in enumerate(target_list):
            target_pose = target[:7]
            target_wrench = target[-6:]
            self.robot.sendCartesianMotionForce(target_pose, target_wrench, maxLinearVel=max_linear_vel, maxAngularVel=max_angular_vel)
            time.sleep(wait_time)

    @staticmethod
    def cvt_hybrid_parameters(act_list):
        act = np.array(act_list)
        ft_xyz = act[:, :3]
        ft_quat = act[:, 3:7]
        force = act[:, 7:10]
        ft_rot_matrix = Rot.from_quat(ft_quat).as_matrix()

        if np.linalg.norm(ft_xyz[-1] - ft_xyz[0]) < 0.005:
            raise RuntimeError("cvt controller wrong")
        motion_direction = (ft_xyz[-1] - ft_xyz[0]) / np.linalg.norm(ft_xyz[-1] - ft_xyz[0])

        force_incam = (ft_rot_matrix @ force[:, :, None])[:, :, 0]
        force_cmd_incam = force_incam - np.dot(force_incam, motion_direction)[:, None] * motion_direction
        force_cmd = (ft_rot_matrix.transpose((0, 2, 1)) @ force_cmd_incam[:, :, None])[:, :, 0]
        act[:, 7:10] = force_cmd
        return act.tolist()
    
    def cvt_action(self, act, c2b, hybrid_force=False):
        xyz, quat = act[:3], act[3:7]
        pose = np.identity(4)
        pose[:3, 3] = xyz
        pose[:3,:3] = Rot.from_quat(quat).as_matrix()

        pose_inbase = c2b @ pose
        xyz_inbase = pose_inbase[:3, 3]
        quat_inbase = Rot.from_matrix(pose_inbase[:3, :3]).as_quat()

        if not hybrid_force:
            if not hasattr(self, 'rest_joints'):
                self.rest_joints = self.get_joint_pos()
            joints = self.robot_model.inverse_kinematics(xyz_inbase.astype(np.float64), 
                                          np.array([quat_inbase[3], quat_inbase[0], quat_inbase[1], quat_inbase[2]]).astype(np.float64), 
                                          rest_pose=self.rest_joints.astype(np.float64))
            self.rest_joints = joints.copy()
            return joints.tolist()
        else:
            wrench = act[7:]
            wrench[:3] = pose_inbase[:3,:3] @ wrench[:3]
            wrench[3:] = pose_inbase[:3,:3] @ wrench[3:]
            wrench_cmd = wrench.tolist()
            pose_cmd = xyz_inbase.tolist() + [quat_inbase[-1]] + quat_inbase[:3].tolist()
            return pose_cmd + wrench_cmd


    def get_delta_q(self, target_q):
        current_q = self.get_q()
        delta_q = np.max(np.abs(np.array(current_q) - np.array(target_q)))
        return delta_q

    def get_q(self):
        return self.get_joint_pos()

    def _get_robot_states(self):
        self.robot.getRobotStates(self.robot_states)
        return self.robot_states
    
    def get_joint_pos(self):
        return np.array(self._get_robot_states().q)
    
    def get_joint_vel(self):
        return np.array(self._get_robot_states().dq)
    
    def get_tcp_pose(self, matrix=False):
        tcppose = np.array(self._get_robot_states().tcpPose)
        if matrix:
            pose = np.identity(4)
            pose[:3,:3] = Rot.from_quat(np.array([tcppose[4],tcppose[5],tcppose[6],tcppose[3]])).as_matrix()
            pose[:3,3] = np.array(tcppose[:3])
            return pose
        else:
            return tcppose
    
    def get_tcp_vel(self):
        return np.array(self._get_robot_states().tcpVel)

    def get_ext_wrench(self, base=True):
        if base:
            return np.array(self._get_robot_states().extWrenchInBase)
        else:
            return np.array(self._get_robot_states().extWrenchInTcp)
