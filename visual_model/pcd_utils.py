import pdb

import open3d as o3d
import numpy as np

def draw_bbox_in_3D_pose_color(bbox):
    points = []
    for i in range(bbox.shape[0]):
        points.append(bbox[i].reshape(-1).tolist())
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

def visu_point_cloud_with_bbox_pose_color(pcds, pre_bbox, gt_bbox):
    pcds = pcds.astype(np.float)
    pre_bbox = pre_bbox.astype(np.float)
    gt_bbox = gt_bbox.astype(np.float)
    vis_list = []
    bbox_t = draw_bbox_in_3D_pose_color(pre_bbox)
    vis_list.append(bbox_t)
    bbox_t = draw_bbox_in_3D_pose_color(gt_bbox)
    vis_list.append(bbox_t)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcds)
    vis_list.append(pcd)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    vis_list.append(coord_frame)
    o3d.visualization.draw_geometries(vis_list)



def view_point(point_cloud):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud)
    cloud.paint_uniform_color([0.1, 0, 0.1])
    o3d.visualization.draw_geometries([cloud, axis_pcd])


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


def pc_camera_to_world(pc, extrinsic):
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    pc = (R @ pc.T).T + T
    return pc


def voxel_sample_points(points, method='voxel', point_number=20000, voxel_size=0.005):
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


def view_point_cloud_parts_and_joint(point_cloud, mask, result):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    colors = np.random.randint(1, 255, size=(30, 3)) / 255.0
    part_num = mask.max() + 1
    clouds = []
    for part_index in range(part_num):
        part_point_cloud = point_cloud[mask==part_index]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color(colors[part_index])
        clouds.append(cloud)
    lines_pcds = []
    for i in range(len(result)):
        polygon_points = np.array([result[i]["joint_translation"].tolist(), (result[i]["joint_translation"]+result[i]["joint_direction"]).tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0.1]])  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)
    o3d.visualization.draw_geometries([*clouds, *lines_pcds, axis_pcd])

def view_point_cloud_parts(point_cloud, mask):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    colors = np.random.randint(1, 255, size=(30, 3)) / 255.0
    clouds = []
    for part_index in np.unique(mask):
        part_point_cloud = point_cloud[mask==part_index]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(part_point_cloud)
        cloud.paint_uniform_color(colors[part_index])
        clouds.append(cloud)
    o3d.visualization.draw_geometries([*clouds, axis_pcd])


def point_cloud_center_and_scale(point_could):
    bound_max = point_could.max(0)
    bound_min = point_could.min(0)
    center = (bound_max + bound_min)/2.0
    scale = (bound_max - bound_min).max()
    scale = scale*(1 + 0.01)
    point_could = point_could - center
    point_could = point_could/scale
    return point_could, center, scale


def radius_based_denoising_numpy(numpy_point_cloud, nb_points=100, radius=0.05):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(numpy_point_cloud)
    cl, index = cloud.remove_statistical_outlier(nb_neighbors=nb_points, std_ratio=1.5)
    # cl, index = cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    numpy_point_cloud = numpy_point_cloud[index]
    return numpy_point_cloud

def translate_joint_direc_world_to_camera(joint_direc, extrinsic):
    extr_inv = np.linalg.inv(extrinsic)  # 求逆
    R = extr_inv[:3, :3]
    T = extr_inv[:3, 3]
    pc = (R @ joint_direc.T).T + T
    return pc


def view_point_cloud_parts_and_joints(point_cloud, mask, joint_directions, offset, offet_dir):
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
    for i in range(point_cloud.shape[0]):
        polygon_points = np.array([point_cloud[i].tolist(), (point_cloud[i]+joint_directions[i]*0.2).tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0.1]])  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)
    for i in range(point_cloud.shape[0]):
        polygon_points = np.array([point_cloud[i].tolist(), (point_cloud[i]+offet_dir[i]*offset[i]).tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([[1.0*offset[i], 0, 0.0]])  # 线条颜色
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)
    o3d.visualization.draw_geometries([*clouds, *lines_pcds, axis_pcd])