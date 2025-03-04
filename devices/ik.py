import os
import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class RobotModel():
    # modified from pinocchio, Faster execution can be modified to c++
    def __init__(self, urdf, link_id=7, eps = 1e-4, IT_MAX = 1000, DT = 1e-1, damp = 1e-12):
        
        self.model = pin.buildModelsFromUrdf(urdf,package_dirs=[os.path.dirname(os.path.abspath(urdf))])[0]
        self.data = self.model.createData()
        
        self.eps = eps
        self.IT_MAX = IT_MAX
        self.DT = DT
        self.damp = damp
        self.link_id = link_id
        
        self.joint_limits_high = self.model.upperPositionLimit
        self.joint_limits_low = self.model.lowerPositionLimit
    
    def inverse_kinematics(self, link_pos, link_quat, rest_pose=None,verbose=False):
        rot = Rot.from_quat(link_quat).as_matrix()
        oMdes = pin.SE3(rot, link_pos)
        q = rest_pose if rest_pose is not None else pin.neutral(self.model)
        i = 0
        JOINT_ID = self.link_id
        while True:
            pin.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[JOINT_ID].actInv(oMdes)
            err = pin.log(iMd).vector  # in joint frame
            if np.linalg.norm(err) < self.eps:
                success = True
                break
            if i >= self.IT_MAX:
                success = False
                break
            J = pin.computeJointJacobian(self.model, self.data, q, JOINT_ID)  # in joint frame
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + self.damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * self.DT)
            if verbose and not i % 10:
                print(f"{i}: error = {err.T}")
            i += 1
        return q.flatten()


if __name__ == "__main__":
    import pybullet as pb
    import pybullet_data
    import time
    
    urdf = 'peel_flexiv.urdf'
    model = RobotModel(urdf)
    
    uid = pb.connect(pb.GUI)
    pb.resetDebugVisualizerCamera(cameraDistance=0.5,cameraYaw=-45,cameraPitch=-30,cameraTargetPosition=[0.2,-0.3,0.70])
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane = pb.loadURDF('plane.urdf')
    panda = pb.loadURDF(urdf, [0,0,0], useFixedBase=True)
    pb.setGravity(0,0,0)
    
    rest_pose = np.array([-0.0, -0.0, -0.0, -np.pi/2, 0.0, np.pi/2, np.pi/4])
    for i in range(len(rest_pose)):
        pb.resetJointState(panda, i, rest_pose[i])
    
    
    link_postion = np.array([0.7681307 , -0.12141033,  0.46617095]).astype(np.float64)
    link_quat = np.array([ 0.4254, -0.5076,  0.4588, -0.5924]).astype(np.float64)
    
    joint_pinocchio = model.inverse_kinematics(link_postion, link_quat)
    for i in range(len(joint_pinocchio)):   
        pb.resetJointState(panda, i, joint_pinocchio[i])
    
    init_joints = np.random.rand(10000,7) * (model.joint_limits_high - model.joint_limits_low) + model.joint_limits_low
    init_joints = init_joints.astype(np.float64)
    t0 = time.time()
    joint_pinocchios = []
    for i in range(100):
        # joint_pinocchio_time = robotmodel.inverse_kinematics(link_postion, link_quat, rest_pose=rest_pose)
        joint_pinocchio_one = model.inverse_kinematics(link_postion, link_quat,rest_pose=init_joints[i])
        joint_pinocchios.append(joint_pinocchio_one)
    print('pinocchio time: {} s'.format((time.time()-t0)))
    
    for i in range(10000):
        joint_pinocchio_one = joint_pinocchios[i]
        for j in range(len(joint_pinocchio_one)):
            pb.resetJointState(panda, j, joint_pinocchio_one[j])
        ii = 0
        while ii < 5:
            time.sleep(0.1)
            ii += 1
            pb.stepSimulation()
