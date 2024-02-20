import pdb

import open3d as o3d
import numpy as np
import torch
from collections.abc import Sequence, Mapping

class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box


def plot_gripper_pro_max(grasp_pose, color=None):
    '''
    Author: chenxi-wang

    **Input:**

    - center: numpy array of (3,), target point as gripper center

    - R: numpy array of (3,3), rotation matrix of gripper

    - width: float, gripper width

    - score: float, grasp quality score

    **Output:**

    - open3d.geometry.TriangleMesh
    '''
    center = grasp_pose[4:7]
    R = grasp_pose[7:].reshape(3, 3)
    score = grasp_pose[0]
    width = grasp_pose[1]
    depth = grasp_pose[3]
    height = grasp_pose[2]
    x, y, z = center
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score  # red for high score
        color_g = 0
        color_b = 1 - score  # blue for low score

    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper

def view_object_part_points(point_cloud, linkpart):
    linkpart_min = int(linkpart.min())
    linkpart_max = int(linkpart.max())
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    part_point_cloud = []
    for part in range(linkpart_min, linkpart_max+1):
        selected = (linkpart == part).reshape(-1)
        selected_pcd = point_cloud[selected, :]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(selected_pcd)
        cloud.paint_uniform_color([0.3+0.2*part, 0.5, 0.5])
        part_point_cloud.append(cloud)
    o3d.visualization.draw_geometries([*part_point_cloud, axis_pcd])


def view_object_joint(point_cloud, mask, joint_translation, vec):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    part_num = int(mask.max()) + 1
    colors = np.random.randint(1, 255, size=(part_num + 5, 3)) / 255.0
    clouds = []
    for part_index in range(part_num):
        if (mask == part_index).sum() == 0:
            continue
        part_point_cloud = point_cloud[mask == part_index]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color(colors[part_index])
        clouds.append(cloud)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(joint_translation)
    cloud.paint_uniform_color([1.0, 0.5, 0.5])
    clouds.append(cloud)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(joint_translation+vec)
    cloud.paint_uniform_color([1.0, 0.2, 0.3])
    clouds.append(cloud)
    o3d.visualization.draw_geometries([*clouds, axis_pcd])

def change_link_mask(link_mask, link_id):
    selected = (link_mask[:, 0] != link_id).reshape(-1)
    link_mask[selected] = 0
    selected = (link_mask[:, 0] == link_id).reshape(-1)
    link_mask[selected] = 1
    return link_mask

def view_object_part(point_cloud, mask):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    part_num = int(mask.max()) + 1
    colors = np.random.randint(1, 255, size=(part_num + 5, 3)) / 255.0
    clouds = []
    for part_index in range(part_num):
        if (mask == part_index).sum() == 0:
            continue
        part_point_cloud = point_cloud[mask == part_index]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color(colors[part_index])
        clouds.append(cloud)
    o3d.visualization.draw_geometries([*clouds, axis_pcd])

def view_object_function_points(point_cloud, linkpart, link_id):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    part_point_cloud = []
    selected = (linkpart[:, 0] == link_id).reshape(-1)
    function_pcds = point_cloud[selected, :]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(function_pcds)
    cloud.paint_uniform_color([0.1, 0.2, 0.3])
    part_point_cloud.append(cloud)
    selected = (linkpart[:, 0] != link_id).reshape(-1)
    base_pcds = point_cloud[selected, :]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(base_pcds)
    cloud.paint_uniform_color([0.5, 0.6, 0.7])
    part_point_cloud.append(cloud)
    o3d.visualization.draw_geometries([*part_point_cloud, axis_pcd])


def sample_point(points, point_number):
    voxel_points, voxel_index = voxel_sample_points(points, point_number=int(point_number*1.5))
    point, fps_index = farthest_point_sample(voxel_points, npoint=point_number)
    return point, voxel_index, fps_index


def voxel_sample_points(points, method='voxel', point_number=4096, voxel_size=0.005):
    ''' points: numpy.ndarray, [N,3]
        method: 'voxel'/'random'
        num_points: output point number
        voxel_size: grid size used in voxel_down_sample
    '''
    assert (method in ['voxel', 'random'])
    if method == 'voxel':
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud, trace, _ = cloud.voxel_down_sample_and_trace(voxel_size=voxel_size, min_bound=cloud.get_min_bound() + 1, max_bound=cloud.get_max_bound() +1)
        to_index_org = np.max(trace, 1)
        points = np.array(cloud.points)
    if len(points) >= point_number:
        idxs = np.random.choice(len(points), point_number, replace=False)
    else:
        idxs1 = np.arange(len(points))
        idxs2 = np.random.choice(len(points), point_number - len(points), replace=True)
        idxs = np.concatenate([idxs1, idxs2])
    points = points[idxs]
    index_org = to_index_org[idxs]
    return points, index_org

def FindMaxDis(point_cloud):
    max_xyz = point_cloud.max(0)
    min_xyz = point_cloud.min(0)
    center = (max_xyz + min_xyz) / 2.0
    max_radius = ((((point_cloud - center)**2).sum(1))**0.5).max()
    return max_radius, center

def point_cloud_center_and_scale(point_could, random_scale=0.01):
    bound_max = point_could.max(0)
    bound_min = point_could.min(0)
    center = (bound_max + bound_min)/2.0
    scale = (bound_max - bound_min).max()
    scale = scale*(1 + random_scale)
    point_could = point_could - center
    point_could = point_could/scale
    return point_could, center, scale

def radius_based_denoising_numpy(numpy_point_cloud, nb_points=50, radius=0.05):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(numpy_point_cloud)
    cl, index = cloud.remove_statistical_outlier(nb_neighbors=nb_points, std_ratio=1.5)
    # cl, index = cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    numpy_point_cloud = numpy_point_cloud[index]
    return numpy_point_cloud


def view_point_real_and_sim(real_pcd, sim_point_cloud):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    clouds = []
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(real_pcd)
    cloud.paint_uniform_color([0, 1, 0.0])
    clouds.append(cloud)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(sim_point_cloud)
    cloud.paint_uniform_color([0.6, 0, 0.0])
    clouds.append(cloud)
    o3d.visualization.draw_geometries([*clouds, axis_pcd])


def add_random_noise_to_random_points(numpy_point_cloud, max_noise_std=0.03):
    noisy_point_cloud = numpy_point_cloud.copy()
    noise = np.random.normal(0, max_noise_std, size=numpy_point_cloud.shape)
    selected_indices = np.random.choice(noise.shape[0], int(noise.shape[0]*np.random.uniform(0.05, 0.3)), replace=False)
    noisy_point_cloud[selected_indices] = noisy_point_cloud[selected_indices] + noise[selected_indices]
    return noisy_point_cloud



def farthest_point_sample(point, npoint=4096):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        point: sampled pointcloud, [npoint, D]
        centroids: sampled pointcloud index
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    centroids = centroids.astype(np.int32)
    point = point[centroids]
    return point, centroids

def translate_pc_world_to_camera(point_cloud, extrinsic):
    extr_inv = np.linalg.inv(extrinsic)  # 求逆
    R = extr_inv[:3, :3]
    T = extr_inv[:3, 3]
    pc = (R @ point_cloud.T).T + T
    return pc


def translate_joint_base_world_to_camera(joint_base, extrinsic):
    extr_inv = np.linalg.inv(extrinsic)  # 求逆
    R = extr_inv[:3, :3]
    T = extr_inv[:3, 3]
    pc = (R @ joint_base.T).T + T
    return pc

def translate_joint_direc_world_to_camera(joint_direc, extrinsic):
    extr_inv = np.linalg.inv(extrinsic)  # 求逆
    R = extr_inv[:3, :3]
    T = extr_inv[:3, 3]
    pc = (R @ joint_direc.T).T + T
    return pc


def translate_pose_world_to_camera(pose, extrinsic):
    extr_inv = np.linalg.inv(extrinsic)
    pose = extr_inv @ pose
    return pose


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def real_pcd_filter(pcd):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcd)
    cl, index = cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    display_inlier_outlier(cloud, index)
    # cl, index = cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    pcd = pcd[index]
    return pcd

def view_point(point_cloud):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud)
    cloud.paint_uniform_color([0.1, 0, 0.1])
    o3d.visualization.draw_geometries([cloud, axis_pcd])


def view_row_point(point_cloud):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud)
    cloud.paint_uniform_color([204/255, 204/255, 204/255])
    o3d.visualization.draw_geometries([cloud])


def view_point_and_fps(point_cloud, mask, fps_ids):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    part_num = mask.max() + 1
    clouds = []
    for part_index in range(part_num):
        part_point_cloud = point_cloud[mask == part_index]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color([0.1 * part_index, 0, 0.1])
        clouds.append(cloud)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud[fps_ids])
    cloud.paint_uniform_color([1, 0, 0.1])
    clouds.append(cloud)
    o3d.visualization.draw_geometries([*clouds, axis_pcd])



def view_point_cloud_parts_and_center(point_cloud, mask, offset):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    part_num = int(mask.max()) + 1
    colors = np.random.randint(1, 255, size=(part_num + 5, 3)) / 255.0
    clouds = []
    for part_index in range(part_num):
        part_point_cloud = point_cloud[mask==part_index]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color(colors[part_index])
        clouds.append(cloud)
    lines_pcds = []
    for i in range(point_cloud.shape[0]):
        polygon_points = np.array([point_cloud[i].tolist(), (point_cloud[i]+offset[i]).tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0.1]])  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)
    o3d.visualization.draw_geometries([*clouds, *lines_pcds, axis_pcd])


def view_point_cloud_joint_and_center(point_cloud, function_mask, offset, offet_dir):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    function_parts = np.unique(function_mask)
    colors = np.random.randint(1, 255, size=(3, 3)) / 255.0
    clouds = []
    for part_index in function_parts:
        if part_index == 2:
            continue
        part_point_cloud = point_cloud[function_mask == part_index]
        part_point_cloud = part_point_cloud + offet_dir[function_mask == part_index]*offset[function_mask == part_index]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color(colors[part_index])
        clouds.append(cloud)
    o3d.visualization.draw_geometries([*clouds, axis_pcd])


def view_naocs_point_cloud_parts_and_joints(point_cloud, mask, joints):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    part_num = mask.max() + 1
    colors = np.random.randint(1, 255, size=(part_num + 5, 3)) / 255.0
    clouds = []
    for part_index in range(part_num):
        part_point_cloud = point_cloud[mask==part_index]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color(colors[part_index])
        clouds.append(cloud)
    lines_pcds = []
    for part_index, joint in enumerate(joints):
        if not joint:
            continue
        if joint["type"] == -1:
            continue
        joint_position = joint['abs_position']
        joint_axis = joint['axis']
        polygon_points = np.array([joint_position.tolist(), (joint_position+joint_axis*0.2).tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([[0.2*part_index, 0, 0.1]])  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)
    o3d.visualization.draw_geometries([*clouds, *lines_pcds, axis_pcd])

def view_point_cloud_parts_and_joints(point_cloud, mask, joint_directions, joint_proj_vecs, point_to_center, part_num=50):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    mask_ids = np.unique(mask)
    colors = np.random.randint(1, 255, size=(255, 3)) / 255.0
    clouds = []
    for mask_id in mask_ids:
        part_point_cloud = point_cloud[mask == mask_id]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color(colors[mask_id])
        clouds.append(cloud)
    lines_pcds = []
    for mask_id in mask_ids:
        part_point_cloud = point_cloud[mask == mask_id]
        part_joint_proj_vecs = joint_proj_vecs[mask == mask_id]
        part_joint_directions = joint_directions[mask == mask_id]
        part_point_to_center = point_to_center[mask == mask_id]
        for i in range(part_point_cloud.shape[0]):
            if i > part_num:
                break
            polygon_points = np.array([part_point_cloud[i].tolist(), (part_point_cloud[i]+part_joint_proj_vecs[i]).tolist()])
            lines = [[0, 1]]
            lines_pcd = o3d.geometry.LineSet()
            lines_pcd.lines = o3d.utility.Vector2iVector(lines)
            lines_pcd.colors = o3d.utility.Vector3dVector([colors[mask_id]])
            lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
            lines_pcds.append(lines_pcd)
            polygon_points = np.array([part_point_cloud[i].tolist(), (part_point_cloud[i] + part_joint_directions[i]*0.3).tolist()])
            lines = [[0, 1]]
            lines_pcd = o3d.geometry.LineSet()
            lines_pcd.lines = o3d.utility.Vector2iVector(lines)
            lines_pcd.colors = o3d.utility.Vector3dVector([colors[mask_id]])
            lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
            lines_pcds.append(lines_pcd)
            polygon_points = np.array([part_point_cloud[i].tolist(), (part_point_cloud[i] + part_point_to_center[i]).tolist()])
            lines = [[0, 1]]
            lines_pcd = o3d.geometry.LineSet()
            lines_pcd.lines = o3d.utility.Vector2iVector(lines)
            lines_pcd.colors = o3d.utility.Vector3dVector([colors[mask_id]])
            lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
            lines_pcds.append(lines_pcd)
    o3d.visualization.draw_geometries([*clouds, *lines_pcds, axis_pcd])

def dist_between_3d_line(p1, e1, p2, e2):
    orth_vect = np.cross(e1, e2)
    p = p1 - p2
    if np.linalg.norm(orth_vect) == 0:
        dist = np.linalg.norm(np.cross(p, e1)) / np.linalg.norm(e1)
    else:
        dist = np.linalg.norm(np.dot(orth_vect, p)) / np.linalg.norm(orth_vect)
    return dist

def view_point_cloud_parts(point_cloud, mask):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    part_num = int(mask.max()) + 1
    colors = np.random.randint(1, 255, size=(part_num+5, 3)) / 255.0
    clouds = []
    for part_index in range(part_num):
        # pdb.set_trace()
        part_point_cloud = point_cloud[mask == part_index]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color(colors[part_index])
        clouds.append(cloud)
    o3d.visualization.draw_geometries([*clouds, axis_pcd])