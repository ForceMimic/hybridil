import numpy as np
import open3d as o3d
import pyrealsense2 as rs


class L515Camera:
    def __init__(self, serial:str):
        self.serial = serial

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial)
        self.config.enable_stream(*(rs.stream.depth, 1024, 768, rs.format.z16, 30))
        self.config.enable_stream(*(rs.stream.color, 1280, 720, rs.format.bgr8, 30))
        self.align = rs.align(rs.stream.color)
        self.hole_filling = rs.hole_filling_filter()

        pipe_profile = self.pipeline.start(self.config)
        depth_sensor = pipe_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        initial_frames = self.pipeline.wait_for_frames()
        initial_aligned_frames = self.align.process(initial_frames)
        initial_color_frame = initial_aligned_frames.get_color_frame()
        color_intrinsics = initial_color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.intrinsics = [color_intrinsics.ppx, color_intrinsics.ppy, color_intrinsics.fx, color_intrinsics.fy]
        self.mtx = np.array([[self.intrinsics[2], 0, self.intrinsics[0]], [0, self.intrinsics[3], self.intrinsics[1]], [0, 0, 1]])

        i = 20
        while i > 0:
            self.get_data()
            i -= 1
    
    def __del__(self):
        self.pipeline.stop()
    
    def get_data(self, hole_filling=False):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if hole_filling:
                depth_frame = self.hole_filling.process(depth_frame)
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            break
        return color_image, depth_image
    
    @staticmethod
    def get_point_cloud(depth, intrinsics, color=None):
        height, weight = depth.shape
        [pixX, pixY] = np.meshgrid(np.arange(weight), np.arange(height))
        x = (pixX - intrinsics[0]) * depth / intrinsics[2]
        y = (pixY - intrinsics[1]) * depth / intrinsics[3]
        z = depth
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        if color is None:
            rgb = None
        else:
            rgb = color.reshape(-1, 3)
        return xyz, rgb
    
    @staticmethod
    def transform_pc(pc_camera:np.ndarray, c2w:np.ndarray):
        # pc_camera: (N, 3), c2w: (4, 4)
        pc_camera_hm = np.concatenate([pc_camera, np.ones((pc_camera.shape[0], 1), dtype=pc_camera.dtype)], axis=-1)        # (N, 4)
        pc_world_hm = pc_camera_hm @ c2w.T                                                          # (N, 4)
        pc_world = pc_world_hm[:, :3]                                                               # (N, 3)
        return pc_world
    
    @staticmethod
    def voxelize(pc_xyz:np.ndarray, pc_rgb:np.ndarray, voxel_size:float):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downsampled_pc_xyz = np.asarray(downsampled_pcd.points)
        downsampled_pc_rgb = np.asarray(downsampled_pcd.colors)
        return (downsampled_pc_xyz, downsampled_pc_rgb)
