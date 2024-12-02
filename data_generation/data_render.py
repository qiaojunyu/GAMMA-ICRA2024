import math
import os
import pdb
import json
import numpy as np
import sapien.core as sapien
import argparse
from matplotlib import pyplot as plt
import copy

def get_cam_pos_fix(theta, phi, distance):
    x = math.sin(math.pi / 180 * theta) * math.cos(math.pi / 180 * phi) * distance
    y = math.sin(math.pi / 180 * theta) * math.sin(math.pi / 180 * phi) * distance
    z = math.cos(math.pi / 180 * theta) * distance
    return np.array([x, y, z])

def get_camera_pos_mat(camera):
    Rtilt = camera.get_model_matrix()
    Rtilt_rot = Rtilt[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Rtilt_trl = Rtilt[:3, 3]
    cam2_wolrd = np.eye(4)
    cam2_wolrd[:3, :3] = Rtilt_rot
    cam2_wolrd[:3, 3] = Rtilt_trl
    return cam2_wolrd


def set_scene(cam_pos, width=800, height=800, joint_q_range=0.6, urdf_file=None):
    sim = sapien.Engine()
    renderer = sapien.SapienRenderer()
    sim.set_renderer(renderer)
    renderer = sapien.SapienRenderer()
    sim.set_renderer(renderer)
    scene_config = sapien.SceneConfig()
    scene = sim.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    # scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
    # urdf load
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf_file)
    assert robot, "URDF not loaded."
    joints = robot.get_joints()
    qpos = []
    for joint in joints:
        if joint.get_parent_link() is None:
            continue
        joint_type = joint.type
        if joint_type == 'revolute' or joint_type == 'prismatic' or joint_type == 'continuous':
            joint_limits = joint.get_limits()[0]
            joint_range = joint_limits[1] - joint_limits[0]
            qpos.append(np.random.uniform(joint_limits[0], joint_range*joint_q_range + joint_limits[0]))
    qpos = np.array(qpos)
    assert qpos.shape[0] == robot.get_qpos().shape[0], 'qpos shape not match.'
    robot.set_qpos(qpos=qpos)
    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera = scene.add_mounted_camera(
        name="camera",
        actor=camera_mount_actor,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=width,
        height=height,
        fovx=np.deg2rad(35.0),
        fovy=np.deg2rad(35.0),
        near=0.1,
        far=100.0,
    )
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    camera_mount_actor.set_pose(sapien.Pose(mat44))
    scene.step()
    scene.update_render()
    camera.take_picture()
    return scene, camera, robot, mat44

def get_seg_mask(camera):
    seg_labels = camera.get_uint32_texture("Segmentation")  # [H, W, 4]
    part_mask_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    return part_mask_image

def get_link_name_and_mask_id(cabinet):
    articulation_information = {}
    for link, joint in zip(cabinet.get_links(), cabinet.get_joints()):
        if len(link.get_visual_bodies()) == 0:
            continue
        # check_mesh_bbox(base_part_mesh)
        articulation_information[link.get_name()] = {"name": link.get_name(),
                                                     "mask_id": link.get_id(),
                                                     "joint_type": joint.type,
                                                     "joint_name": joint.name,
                                                     "parent_link": joint.get_parent_link().get_name(),
                                                     }
    return articulation_information

def change_image_background(object_mask_ids, mask_map, rgb_image):
    mask_ids = np.unique(mask_map)
    invalid_masks = []
    for mask_id in mask_ids:
        if mask_id not in object_mask_ids:
            invalid_masks.append(mask_id)
    valid_mask = np.ones_like(mask_map)
    for in_valid_mask in invalid_masks:
        valid_mask = valid_mask & (mask_map != in_valid_mask)
    valid_mask = valid_mask.astype(bool)
    no_background_image = copy.deepcopy(rgb_image)
    no_background_image[~valid_mask] = np.array([216, 206, 189], dtype=np.uint8)
    return no_background_image

def render_data(urdf_path, object_id, save_path, render_view_id=0, view=0):
    phi = np.random.uniform(150, 210)
    theta = np.random.uniform(50, 80)
    distance = np.random.uniform(4.5, 5.0)
    cam_pos = get_cam_pos_fix(theta, phi, distance)
    urdf_file = "{}/{}/mobility_vhacd.urdf".format(urdf_path, object_id)
    scene, camera, robot, mat44 = set_scene(cam_pos, urdf_file=urdf_file)
    intrinsic_matrix = camera.get_intrinsic_matrix()
    result = {"camera_intrinsic": intrinsic_matrix}
    rgba = camera.get_float_texture('Color')
    rgb = rgba[:, :, :3]
    rgb_img = (rgb * 255).clip(0, 255).astype("uint8")
    result["rgb_image"] = rgb_img
    position = camera.get_float_texture('Position')
    depth_map = -position[..., 2]
    camera2world_matrix = get_camera_pos_mat(camera)
    result["depth_map"] = depth_map
    result["camera2world_matrix"] = camera2world_matrix
    part_mask_image = get_seg_mask(camera)
    result["part_mask_map"] = part_mask_image.astype(np.uint8)
    save_path = "{}/{}/{}/".format(save_path, object_id, render_view_id)
    os.makedirs(save_path, exist_ok=True)
    np.savez(save_path + 'render.npz', **result)
    articulation_information = get_link_name_and_mask_id(robot)
    with open(save_path + 'articulation.json', 'w') as f:
        json.dump(articulation_information, f)
    if view:
        object_mask_ids = []
        for link_name, link_data in articulation_information.items():
            object_mask_ids.append(link_data["mask_id"])
        image = change_image_background(object_mask_ids, result["part_mask_map"], result["rgb_image"])
        plt.imshow(image)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf_path', default='./where2act_original_sapien_dataset/', help='name of the dataset we use')
    parser.add_argument('--save_path', default='./train/', help='name of the dataset we use')
    parser.add_argument('--object_id', default='10036', help='name of the dataset we use')
    parser.add_argument('--render_view_id', default=1, help='name of the dataset we use')
    parser.add_argument('--view_render', default=0, help='name of the dataset we use')
    args = parser.parse_args()
    render_data(args.urdf_path, args.object_id, args.save_path, args.render_view_id, args.view_render)







