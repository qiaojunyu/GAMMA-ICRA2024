import glob
import pdb
import time
from enum import Enum
import pybullet as pb
import numpy as np
import argparse
import json
class JointType(Enum):
    prismatic = 1
    revolute = 0
    fixed = 4
    continuous = -1
    floating = -1
    planar = -1


def get_object_ids(data_type):
    object_ids = []
    if data_type == "train":
        all_cat_list = ['StorageFurniture', 'Microwave', 'Refrigerator', 'Door']
        with open("object_ids/train_data_list.txt", 'r') as fin:
            for l in fin.readlines():
                shape_id, cat = l.rstrip().split()
                if cat not in all_cat_list:
                    continue
                object_ids.append(shape_id)
    elif data_type == "test":
        all_cat_list = ['StorageFurniture', 'Microwave', 'Refrigerator', 'Door', 'Safe', 'WashingMachine', 'Table']
        with open("object_ids/train_data_list.txt", 'r') as fin:
            for l in fin.readlines():
                shape_id, cat = l.rstrip().split()
                if cat not in all_cat_list:
                    continue
                object_ids.append(shape_id)
    else:
        raise NotImplementedError
    return object_ids

def get_joint_pos_vat(body_id, link_id):
    if link_id == -1:
        pos, ori = pb.getBasePositionAndOrientation(body_id)
    else:
        pos, ori = pb.getLinkState(body_id, link_id)[4:6]
    AABB_data = pb.getAABB(body_id, link_id)
    R_Mat = np.array(pb.getMatrixFromQuaternion(ori)).reshape(3, 3)
    pose = np.identity(4)
    pose[:3, :3] = R_Mat
    pose[:3, 3] = np.array(pos)
    return pose, AABB_data
def get_controlable_joints(object_id):
    control_joints = []
    total_joints = {}
    max_angles = []
    min_angles = []
    joint_name_to_index = dict()
    link_name_to_index = {pb.getBodyInfo(object_id)[0].decode('UTF-8'): -1, }
    joint_num = pb.getNumJoints(object_id)
    for joint_index in range(joint_num):
        info_tuple = pb.getJointInfo(object_id, joint_index)
        link_name = pb.getJointInfo(object_id, joint_index)[12].decode('UTF-8')
        link_name_to_index[link_name] = joint_index
        joint_name_to_index[info_tuple[1].decode('UTF-8')] = joint_index
        joint_info = {"name": info_tuple[1].decode('UTF-8'), "joint_id": joint_index,
                      "joint_type": info_tuple[2], "joint_axis": info_tuple[13],
                      "link_name": info_tuple[12].decode('UTF-8')}
        total_joints[joint_info["link_name"]] = joint_info
        if info_tuple[2] != 4:
            joint_info = {"name": info_tuple[1].decode('UTF-8'), "joint_id": joint_index,
                          "joint_type": info_tuple[2], "joint_axis": info_tuple[13], "link_name": info_tuple[12].decode('UTF-8')}
            control_joints.append(joint_info)
            min_angles.append(info_tuple[8])
            max_angles.append(info_tuple[9])
    return control_joints, total_joints, min_angles, max_angles, link_name_to_index, joint_name_to_index
def draw_axis(base_postition, dicretion, length=0.9, color=[1,0,0]):
    end_p = base_postition + dicretion*length
    pb.addUserDebugLine(base_postition, end_p,
                        lineColorRGB=color,
                        lineWidth=0.3)

def change_part_color(object, link_id):
    pb.changeVisualShape(object, link_id, rgbaColor=[1.0, 0.0, 0.0, 1])

def anno_joint_type(urdf_path):
    urdf_file = "{}/mobility_vhacd.urdf".format(urdf_path)
    try:
        object = pb.loadURDF(urdf_file, useFixedBase=True)
    except Exception:
        print("error urdf: ", urdf_path)
        pb.resetSimulation()
        return

    link_data_annos = {}

    control_joints, total_joints, min_angles, max_angles, link_name_to_index, joint_name_to_index = get_controlable_joints(object)
    select_joints = []
    for joint_info in control_joints:
        print("joint_type: ", joint_info["joint_type"])
        if joint_info["joint_type"] == JointType.revolute.value or joint_info["joint_type"] == JointType.prismatic.value:
            select_joints.append(joint_info)

    if len(select_joints) == 0:
        print("no part id: ", urdf_path)
        pb.resetSimulation()
        return
    else:
        print("target part num: ", len(select_joints))

    for joint_info in select_joints:
        change_part_color(object, link_name_to_index[joint_info["link_name"]])
        selected = input("selected the red part: 0 or 1:")
        assert int(selected) == 0 or int(selected) == 1
        if int(selected) == 0:
            continue
        joint_pose, _ = get_joint_pos_vat(object, joint_info["joint_id"])
        joint_axis = np.array(joint_info["joint_axis"])
        direction = abs(joint_pose[:3, abs(joint_axis).argmax()])
        draw_axis(joint_pose[:3, 3], direction)
        link_data_annos[joint_info["link_name"]] = {"joint_base": joint_pose[:3, 3].flatten(order='F').tolist(),
                                                    "joint_axis": direction.flatten(order='F').tolist(),
                                                    "joint_type": joint_info["joint_type"],
                                                    "joint_id": joint_info["joint_id"],
                                                    "joint_name": joint_info["name"]}
    for link_name, joint_info in link_data_annos.items():
        info_tuple = pb.getJointInfo(object, joint_info["joint_id"])
        joint_low = info_tuple[8]
        joint_high = info_tuple[9]
        pb.resetJointState(object, joint_info["joint_id"], np.random.uniform(joint_low, joint_high))
    for _ in range(100):
        pb.stepSimulation()
        time.sleep(0.01)
    with open(urdf_path + "/target_part_info.json", "w") as f:
        json.dump(link_data_annos, f)
    pb.resetSimulation()
    print("finish: ", urdf_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='/data/where2act_original_sapien_dataset/', help='name of the dataset we use')
    parser.add_argument('--data_type', default='train', help='name of the dataset we use')
    args = parser.parse_args()
    pb.connect(pb.GUI)
    pb.setRealTimeSimulation(1)  # does not work with p.DIRECT
    object_ids = get_object_ids(args.data_type)
    for object_id in object_ids:
        anno_joint_type(args.file_path + object_id)