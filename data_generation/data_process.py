import copy
import os
import pdb
import json
import numpy as np
import argparse
import open3d as o3d

def pc_camera_to_world(pc, extrinsic):
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    pc = (R @ pc.T).T + T
    return pc


def view_point_clouds(point_cloud, rgbs=None,  joint_bases=None, joint_axises=None):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud)
    if rgbs is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgbs)
    if joint_bases is not None and joint_axises is not None:
        lines_pcds = []
        for joint_base, joint_axis in zip(joint_bases, joint_axises):
            polygon_points = np.array([joint_base.tolist(), (joint_base + joint_axis*0.1).tolist()])
            lines = [[0, 1]]
            lines_pcd = o3d.geometry.LineSet()
            lines_pcd.lines = o3d.utility.Vector2iVector(lines)
            lines_pcd.colors = o3d.utility.Vector3dVector([np.array([1, 0, 0])])
            lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
            lines_pcds.append(lines_pcd)
        o3d.visualization.draw_geometries([cloud, *lines_pcds,axis_pcd])
    else:
        o3d.visualization.draw_geometries([cloud, axis_pcd])

def view_sem_labels(point_cloud, sem_masks):
    clouds = []
    for mask_id in np.unique(sem_masks):
        part_point_cloud = point_cloud[sem_masks == mask_id]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        if mask_id == 0:
            cloud.paint_uniform_color([1, 0, 0])
        elif mask_id == 1:
            cloud.paint_uniform_color([0, 1, 0])
        elif mask_id == 2:
            cloud.paint_uniform_color([0, 0., 0.])
        clouds.append(cloud)
    o3d.visualization.draw_geometries([*clouds])


def view_part_labels(point_cloud, part_masks):
    clouds = []
    colors = np.random.randint(1, 255, size=(300, 3)) / 255.0
    for mask_id in np.unique(part_masks):
        part_point_cloud = point_cloud[part_masks == mask_id]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color(colors[mask_id])
        clouds.append(cloud)
    o3d.visualization.draw_geometries([*clouds])


def get_point_cloud(files, object_mask_ids):
    rgb_image = files['rgb_image']
    depth_map = files['depth_map']
    mask_map = files["part_mask_map"]
    camera2world_matrix = files['camera2world_matrix']
    width, height = rgb_image.shape[0], rgb_image.shape[1]
    K = np.array(files['camera_intrinsic']).reshape(3, 3)
    y_coords, x_coords = np.indices((height, width))
    z_new = depth_map.astype(float)
    mask_ids = np.unique(mask_map)
    invalid_masks = []
    for mask_id in mask_ids:
        if mask_id not in object_mask_ids:
            invalid_masks.append(mask_id)
    valid_mask = np.ones_like(mask_map)
    for in_valid_mask in invalid_masks:
        valid_mask = valid_mask & (mask_map != in_valid_mask)
    valid_mask = valid_mask.astype(bool)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    z_new = z_new[valid_mask]
    x_new = (x_coords - K[0, 2]) * z_new / K[0, 0]
    y_new = (y_coords - K[1, 2]) * z_new / K[1, 1]
    point_cloud = np.stack((x_new, y_new, z_new), axis=-1)
    per_point_rgb = rgb_image[y_coords, x_coords] / 255.0
    camera_point_clouds = np.array(point_cloud)
    world_point_clouds = pc_camera_to_world(camera_point_clouds, camera2world_matrix)
    per_point_rgbs = np.array(per_point_rgb)
    return world_point_clouds, per_point_rgbs, camera2world_matrix, mask_map[valid_mask]

def get_sem_and_instance_mask(articulations, anno_links, object_part_masks):
    merge_parts = {}
    sem_labels = {}
    joint_infos = {}
    for link_name, link_data in articulations.items():
        if link_data["name"] not in anno_links:
            if link_data["parent_link"] == "base":
                merge_parts[link_data["mask_id"]] = []
                sem_labels[link_data["mask_id"]] = 2
            else:
                parent_link_data = articulations[link_data["parent_link"]]
                if parent_link_data["name"] not in merge_parts:
                    merge_parts[parent_link_data["mask_id"]] = [link_data["mask_id"]]
                    sem_labels[parent_link_data["mask_id"]] = anno_links[parent_link_data["name"]]["joint_type"]
                    joint_infos[parent_link_data["mask_id"]] = {"joint_base": anno_links[parent_link_data["name"]]["joint_base"],
                                                                "joint_axis": anno_links[parent_link_data["name"]]["joint_axis"]}
                else:
                    merge_parts[parent_link_data["mask_id"]].append(link_data["mask_id"])
        else:
            merge_parts[link_data["mask_id"]] = []
            sem_labels[link_data["mask_id"]] = anno_links[link_name]["joint_type"]
            joint_infos[link_data["mask_id"]] = {"joint_base": anno_links[link_name]["joint_base"],
                                                 "joint_axis": anno_links[link_name]["joint_axis"]}

    part_ids = np.unique(object_part_masks)
    new_part_masks = copy.deepcopy(object_part_masks)
    sem_masks = np.zeros_like(object_part_masks)
    for part_id in part_ids:
        if part_id not in merge_parts:
            for parent_part_id, merged_part_ids in merge_parts.items():
                if part_id in merged_part_ids:
                    new_part_masks[object_part_masks == part_id] = parent_part_id
                    sem_masks[object_part_masks == part_id] = sem_labels[parent_part_id]
        else:
            new_part_masks[object_part_masks == part_id] = part_id
            sem_masks[object_part_masks == part_id] = sem_labels[part_id]

    assert np.unique(new_part_masks).shape[0] == len(joint_infos) + 1
    sort_instance_masks = np.zeros_like(new_part_masks)
    joint_bases = []
    joint_axises = []
    instance_id = 0
    for part_id in np.unique(new_part_masks):
        sort_instance_masks[new_part_masks == part_id] = instance_id
        if part_id in joint_infos:
            joint_bases.append(np.array(joint_infos[part_id]["joint_base"]))
            joint_axises.append(np.array(joint_infos[part_id]["joint_axis"]))
        instance_id += 1
    joint_bases = np.array(joint_bases)
    joint_axises = np.array(joint_axises)
    return new_part_masks, sem_masks, joint_bases, joint_axises

def process_data(urdf_path, render_file_path, save_path, view=0):
    object_id = render_file_path.split('/')[-2]
    view_id = render_file_path.split('/')[-1]
    anno_links = "{}/{}/target_part_info.json".format(urdf_path, object_id)
    with open(anno_links, 'r') as f:
        anno_links = json.load(f)
    render_data = np.load(render_file_path + '/render.npz', allow_pickle=True)
    with open(render_file_path + '/articulation.json', 'r') as f:
        articulations = json.load(f)
        object_mask_ids = []
        for link_name, link_data in articulations.items():
            object_mask_ids.append(link_data["mask_id"])
    world_point_clouds, per_point_rgbs, camera2world_matrix, object_part_masks = get_point_cloud(render_data,
                                                                                                  object_mask_ids)
    new_part_masks, sem_masks, joint_bases, joint_axises = get_sem_and_instance_mask(articulations, anno_links, object_part_masks)
    os.makedirs(save_path, exist_ok=True)
    if view:
        view_point_clouds(world_point_clouds, per_point_rgbs, joint_bases, joint_axises)
        view_sem_labels(world_point_clouds, sem_masks)
        view_part_labels(world_point_clouds, new_part_masks)
    np.savez(save_path + '/{}_{}.npz'.format(object_id, view_id),
             pcd_world=world_point_clouds,
             rgbs=per_point_rgbs,
             instance_mask=new_part_masks,
             function_mask=sem_masks,
             extrinsic=camera2world_matrix,
             joint_bases=joint_bases,
             joint_axises=joint_axises)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf_path', default='/data/where2act_original_sapien_dataset/', help='name of the dataset we use')
    parser.add_argument('--render_file_path', default='./train/10143/0', help='name of the dataset we use')
    parser.add_argument('--save_path', default='./train_processed/', help='name of the dataset we use')
    parser.add_argument('--view_render', default=0, help='name of the dataset we use')
    args = parser.parse_args()
    process_data(args.urdf_path, args.render_file_path, args.save_path, args.view_render)

