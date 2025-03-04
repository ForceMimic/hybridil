import argparse
from collections import deque
import json
import time
from pynput import keyboard
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2
import zmq

from devices.flexiv import FlexivRobot
from devices.l515 import L515Camera
from configs.pose import *


def main(cfg):
    name = cfg['name']
    robot_addr = [cfg['robot_addr1'], cfg['robot_addr2']]
    urdf_path = cfg['urdf_path']
    l515_serial = cfg['l515_serial']
    port = cfg['port']
    pc_num = cfg['pc_num']
    voxel_size = cfg['voxel_size']
    clip_object = cfg['clip_object']
    clip_pyft = cfg['clip_pyft']
    clip_base = cfg['clip_base']
    obs_horizon = cfg['obs_horizon']
    act_horizon = cfg['act_horizon']
    force_threshold = cfg['force_threshold']
    force_num_threshold = cfg['force_num_threshold']

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{port}")

    robot = FlexivRobot(addr=robot_addr, urdf_path=urdf_path)
    cam = L515Camera(serial=l515_serial)
    robot.home()
    robot.set_zero_ft()

    finish = False
    def _on_prese(key):
        nonlocal finish
        if hasattr(key, 'char') and key.char == 'q':
            finish = True
            print("finish")
    def _on_release(key):
        pass
    keyboard.Listener(on_press=_on_prese, on_release=_on_release).start()    

    flexiv_act = None
    pointclouds = deque(maxlen=obs_horizon)
    robot0_eef_pos = deque(maxlen=obs_horizon)
    robot0_eef_quat =deque(maxlen=obs_horizon)
    robot0_eef_wrench = deque(maxlen=obs_horizon)
    action_deque = deque(maxlen=act_horizon)
    hybrid_force = False
    init_force_detection = True
    while True:
        if len(action_deque) == 0:
            color_image, depth_image = cam.get_data()
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) / 255.
            depth_image = depth_image * cam.depth_scale
            l515_pc_xyz_l515, l515_pc_rgb = L515Camera.get_point_cloud(depth_image, cam.intrinsics, color_image)

            pose_inbase = robot.get_tcp_pose(matrix=True)
            pose_incam = np.linalg.inv(np.linalg.inv(BASEr_2_BASE) @ L515_2_BASE) @ pose_inbase
            eef_pos = pose_incam[:3, 3]
            eef_quat = Rot.from_matrix(pose_incam[:3, :3]).as_quat()
            wrench_intcp = robot.get_ext_wrench(base=False)

            if clip_object:
                l515_pc_xyz_base = L515Camera.transform_pc(l515_pc_xyz_l515, L515_2_BASE)
                clip_object_mask = (l515_pc_xyz_base[:, 0] > OBJECT_SPACE[0][0]) & (l515_pc_xyz_base[:, 0] < OBJECT_SPACE[0][1]) & \
                                    (l515_pc_xyz_base[:, 1] > OBJECT_SPACE[1][0]) & (l515_pc_xyz_base[:, 1] < OBJECT_SPACE[1][1]) & \
                                    (l515_pc_xyz_base[:, 2] > OBJECT_SPACE[2][0]) & (l515_pc_xyz_base[:, 2] < OBJECT_SPACE[2][1])
            else:
                clip_object_mask = np.zeros((l515_pc_xyz_l515.shape[0],), dtype=bool)
            if clip_pyft:
                l515_pc_xyz_pyft = L515Camera.transform_pc(l515_pc_xyz_l515, np.linalg.inv(pose_incam))
                clip_pyft_mask = (l515_pc_xyz_pyft[:, 0] > PYFT_SPACE[0][0]) & (l515_pc_xyz_pyft[:, 0] < PYFT_SPACE[0][1]) & \
                                    (l515_pc_xyz_pyft[:, 1] > PYFT_SPACE[1][0]) & (l515_pc_xyz_pyft[:, 1] < PYFT_SPACE[1][1]) & \
                                    (l515_pc_xyz_pyft[:, 2] > PYFT_SPACE[2][0]) & (l515_pc_xyz_pyft[:, 2] < PYFT_SPACE[2][1])
            else:
                clip_pyft_mask = np.zeros((l515_pc_xyz_l515.shape[0],), dtype=bool)
            if clip_base:
                l515_pc_xyz_base = L515Camera.transform_pc(l515_pc_xyz_l515, L515_2_BASE)
                clip_base_mask = (l515_pc_xyz_base[:, 0] > BASE_SPACE[0][0]) & (l515_pc_xyz_base[:, 0] < BASE_SPACE[0][1]) & \
                                    (l515_pc_xyz_base[:, 1] > BASE_SPACE[1][0]) & (l515_pc_xyz_base[:, 1] < BASE_SPACE[1][1]) & \
                                    (l515_pc_xyz_base[:, 2] > BASE_SPACE[2][0]) & (l515_pc_xyz_base[:, 2] < BASE_SPACE[2][1])
            else:
                clip_base_mask = np.ones((l515_pc_xyz_l515.shape[0],), dtype=bool)
            valid_mask = np.logical_and(clip_base_mask, np.logical_or(clip_object_mask, clip_pyft_mask))
            # TODO: hardcode to throw out hands
            l515_pc_xyz_pyft = L515Camera.transform_pc(l515_pc_xyz_l515, np.linalg.inv(pose_incam))
            valid_mask = np.logical_and(valid_mask, l515_pc_xyz_pyft[:, 2] > 0.)
            valid_mask = np.where(valid_mask)[0]
            l515_pc_xyz_l515 = l515_pc_xyz_l515[valid_mask]
            l515_pc_rgb = l515_pc_rgb[valid_mask]
            if voxel_size != 0:
                l515_pc_xyz_l515, l515_pc_rgb = L515Camera.voxelize(l515_pc_xyz_l515, l515_pc_rgb, voxel_size)
            if pc_num != -1:
                if l515_pc_xyz_l515.shape[0] > pc_num:
                    valid_mask = np.random.choice(l515_pc_xyz_l515.shape[0], pc_num, replace=False)
                elif l515_pc_xyz_l515.shape[0] < pc_num:
                    print(f"Warning: {l515_pc_xyz_l515.shape[0] = }")
                    valid_mask = np.concatenate([np.arange(l515_pc_xyz_l515.shape[0]), np.random.choice(l515_pc_xyz_l515.shape[0], pc_num - l515_pc_xyz_l515.shape[0], replace=False)], axis=0)
                l515_pc_xyz_l515 = l515_pc_xyz_l515[valid_mask]
                l515_pc_rgb = l515_pc_rgb[valid_mask]
            color_pcd = np.concatenate([l515_pc_xyz_l515, l515_pc_rgb], axis=-1)
            
            pointclouds.append(color_pcd)
            robot0_eef_pos.append(eef_pos.tolist())
            robot0_eef_quat.append(eef_quat.tolist())
            robot0_eef_wrench.append(wrench_intcp.tolist())

            if len(pointclouds) < obs_horizon:
                continue

            t0 = time.time()
            print("========================  send data...   ========================")
            obs = dict()
            obs["name"] = name
            obs['pointcloud'] = np.array(pointclouds).tolist()
            obs['robot0_eef_pos'] = np.array(robot0_eef_pos).tolist()
            obs['robot0_eef_quat'] = np.array(robot0_eef_quat).tolist()
            obs['robot0_eef_wrench'] = np.array(robot0_eef_wrench).tolist()
            data_json = json.dumps(obs)
            socket.send_string(data_json)
            print("========================  send data time: {}, wating for return...   ========================".format(time.time() - t0))
            received_data_json = socket.recv_string()
            received_data = json.loads(received_data_json)
            print("======================== received data... ========================")
            pred_action = np.array(received_data["action"])
            Na = pred_action.shape[0]
            # TODO: hardcode index here to extract forces
            forces = np.linalg.norm(pred_action[:, 7:10], axis=1)

            if init_force_detection and np.sum(forces > force_threshold) > Na * force_num_threshold:
                init_force_detection = False
                first_force_index = np.argmax(forces > force_threshold)
                if first_force_index == 0:
                    hybrid_force = True
                    action_deque.extend(FlexivRobot.cvt_hybrid_parameters(pred_action))
                else:
                    hybrid_force = False
                    action_deque.extend(pred_action[:first_force_index])
            elif init_force_detection and np.sum(forces > force_threshold) <= Na * force_num_threshold:
                hybrid_force = False
                action_deque.extend(pred_action)
            elif not init_force_detection and np.sum(forces > force_threshold) == Na:
                hybrid_force = True
                action_deque.extend(FlexivRobot.cvt_hybrid_parameters(pred_action))
            elif not init_force_detection and np.sum(forces > force_threshold) > Na * force_num_threshold:
                first_force_index = np.argmax(forces <= force_threshold)
                if first_force_index == 0:
                    hybrid_force = False
                    action_deque.extend(pred_action[:np.argmax(forces > force_threshold)])
                else:
                    hybrid_force = True
                    action_deque.extend(FlexivRobot.cvt_hybrid_parameters(pred_action[:first_force_index]))
            else:
                hybrid_force = False
                action_deque.extend(pred_action)
            flexiv_act = []
        
        act = np.array(action_deque.popleft())
        if hybrid_force:
            pose_wrench = robot.cvt_action(act, c2b=np.linalg.inv(BASEr_2_BASE) @ L515_2_BASE, hybrid_force=True)
            flexiv_act.append(pose_wrench)
            if len(action_deque) == 0:
                robot.move_hybrid(flexiv_act)
        else:
            joints = robot.cvt_action(act, c2b=np.linalg.inv(BASEr_2_BASE) @ L515_2_BASE, hybrid_force=False)
            flexiv_act.append(joints)
            if len(action_deque) == 0:
                robot.move_joints(flexiv_act)
        
        if finish:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="(optional) name. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--robot_addr1",
        type=str,
        help="(optional) flexiv robot ip address 1. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--robot_addr2",
        type=str,
        help="(optional) flexiv robot ip address 2. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        help="(optional) path to urdf of robot. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--l515_serial",
        type=str,
        help="(optional) serial number of l515 camera. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="(optional) port number to bind the server. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--pc_num",
        type=int,
        help="(optional) number of points. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        help="(optional) voxel size. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--clip_object",
        action='store_true',
        help="(optional) whether to clip object. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--clip_pyft",
        action='store_true',
        help="(optional) whether to clip pyft. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--clip_base",
        action='store_true',
        help="(optional) whether to clip base. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--obs_horizon",
        type=int,
        help="(optional) observation horizon. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--act_horizon",
        type=int,
        help="(optional) action horizon. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--force_threshold",
        type=float,
        help="(optional) force threshold. Only needs to be provided if --config is not provided",
    )
    parser.add_argument(
        "--force_num_threshold",
        type=float,
        help="(optional) force number threshold. Only needs to be provided if --config is not provided",
    )

    args = parser.parse_args()
    if args.config is not None:
        cfg = json.load(open(args.config, 'r'))
    else:
        cfg = {
            "name": args.name,
            "robot_addr1": args.robot_addr1,
            "robot_addr2": args.robot_addr2,
            "urdf_path": args.urdf_path,
            "l515_serial": args.l515_serial,
            "port": args.port,
            "pc_num": args.pc_num,
            "voxel_size": args.voxel_size,
            "clip_object": args.clip_object,
            "clip_pyft": args.clip_pyft,
            "clip_base": args.clip_base,
            "obs_horizon": args.obs_horizon,
            "act_horizon": args.act_horizon,
            "force_threshold": args.force_threshold,
            "force_num_threshold": args.force_num_threshold,
        }
    main(cfg)
