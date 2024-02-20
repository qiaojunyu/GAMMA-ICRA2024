import copy
import pdb
import time
from typing import Union, List
from pathlib import Path
import glob
import torch.utils.data as data
import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
from .data_utilts import point_cloud_center_and_scale, translate_joint_base_world_to_camera, translate_joint_direc_world_to_camera, sample_point, \
    translate_pc_world_to_camera, view_point_cloud_parts_and_joints, view_point_cloud_parts_and_center, voxel_sample_points

class GammaDataset(data.Dataset):
    """
    objcet link dataset
    """
    def __init__(self,
                 root: Union[Path, str, List[Path], List[str]],
                 noise: bool = True,
                 point_num: int = 10000,
                 ):
        self._root: List[Path] = [Path(r).expanduser() for r in root]
        self._noise = noise
        self._point_num = point_num
        self._files = []
        self._file_num = 0
        self.color_jitter = 0.3
        for root_path in self._root:
            objects_dir = str(root_path) + "/*"
            object_file_dirs = glob.glob(objects_dir)
            for object_file_dir in object_file_dirs:
                object_pcd_files = glob.glob(object_file_dir + "/*npz")
                self._files = self._files + object_pcd_files
                self._file_num = self._file_num + len(object_pcd_files)
        # self.data_check()

    def __len__(self):
        return self._file_num

    def data_check(self):
        print("start data check: ", self._file_num)
        for file_path in tqdm(self._files):
            file = np.load(file_path, allow_pickle=True)
            pcd_world = file["pcd_world"]
            joint_bases = file["joint_bases"]
            move_mask = file["move_mask"]
            instance_mask = file["instance_mask"]
            function_mask = file["function_mask"]
            pcd_world, voxel_index = voxel_sample_points(pcd_world, point_number=self._point_num)
            move_mask = move_mask[voxel_index]
            function_mask = function_mask[voxel_index]
            instance_mask = instance_mask[voxel_index]
            num_instances = np.unique(instance_mask).shape[0]
            if (num_instances - 1) != joint_bases.shape[0]:
                print("error data joint and part: ", file_path)
                os.remove(file_path)
                os.remove(file_path.replace("npz", "png"))
                continue
            if (move_mask == 1).sum() < 200:
                print("error data: ", file_path, " move part < 200: ", (move_mask == 1).sum())
                os.remove(file_path)
                os.remove(file_path.replace("npz", "png"))
                continue
            assert (move_mask == 1).sum() == (function_mask < 2).sum()
            for i in np.unique(instance_mask):
                indices = np.where(instance_mask == i)[0]
                part_function_mask = function_mask[indices]
                if np.unique(part_function_mask).shape[0] > 1:
                    print("error data, function error: ", file_path)
                    # view_object_part_points(pcd_world, instance_mask)
                if indices.shape[0] < 300:
                    print("no instance: {}, num: {}".format(i, indices.shape[0]))
                    print("error data: ", file_path)
                    os.remove(file_path)
                    os.remove(file_path.replace("npz", "png"))
                    break

    def cal_joint_to_part_offset(self, pcd, joint_base, joint_direction):
        joint_axis = joint_direction.reshape((3, 1))
        vec1 = pcd - joint_base
        # project to joint axis
        proj_len = np.dot(vec1, joint_axis)
        # np.clip(proj_len, a_min=self.epsilon, a_max=None, out=proj_len)
        proj_vec = proj_len * joint_axis.transpose()
        orthogonal_vec = - vec1 + proj_vec
        heatmap = np.linalg.norm(orthogonal_vec, axis=1).reshape(-1, 1)
        unitvec = orthogonal_vec / heatmap
        heatmap = 1.0 - heatmap
        heatmap[heatmap < 0] = 0
        proj_vec = orthogonal_vec
        return heatmap, unitvec, proj_vec


    def add_random_noise_to_random_points(self, numpy_point_cloud, max_noise_std=0.03):
        noisy_point_cloud = numpy_point_cloud.copy()
        noise = np.random.normal(0, max_noise_std, size=numpy_point_cloud.shape)
        selected_indices = np.random.choice(noise.shape[0], int(noise.shape[0]*np.random.uniform(0.05, 0.3)), replace=False)
        noisy_point_cloud[selected_indices] = noisy_point_cloud[selected_indices] + noise[selected_indices]
        return noisy_point_cloud

    def simulate_point_cloud_missing_points(self, numpy_point_cloud, missing_probability=0.1):
        missing_mask = np.random.rand(len(numpy_point_cloud)) < missing_probability
        indexs = ~missing_mask
        missing_point_cloud = numpy_point_cloud[indexs]
        return missing_point_cloud, indexs

    def radius_based_denoising_numpy(self, numpy_point_cloud, nb_points=30, radius=0.05):
        if numpy_point_cloud.shape[0] > self._point_num:
            numpy_point_cloud, index_voxel = voxel_sample_points(numpy_point_cloud, point_number=self._point_num)
        else:
            index_voxel = None
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(numpy_point_cloud)
        cl, index_denoise = cloud.remove_statistical_outlier(nb_neighbors=nb_points, std_ratio=1.5)
        numpy_point_cloud = numpy_point_cloud[index_denoise]
        return numpy_point_cloud, index_voxel, index_denoise

    def __getitem__(self, file_id: int, down_sample=True):
        file_id %= self._file_num
        file_path = self._files[file_id]
        file = np.load(file_path, allow_pickle=True)
        coord_world = file["pcd_world"]
        joint_bases = file["joint_bases"]
        joint_axises = file["joint_directions"]
        segment_mask = file["function_mask"]
        instance_mask = file["instance_mask"]
        extrinsic = file["extrinsic"]
        num_instances = np.unique(instance_mask).shape[0]
        assert (num_instances - 1) == joint_bases.shape[0]
        if self._noise:
            coord_world = self.add_random_noise_to_random_points(coord_world)
            coord_world, indexs = self.simulate_point_cloud_missing_points(coord_world)
            segment_mask = segment_mask[indexs]
            instance_mask = instance_mask[indexs]
            coord_world, index_org = voxel_sample_points(coord_world, point_number=int(self._point_num*1.1))
            segment_mask = segment_mask[index_org]
            instance_mask = instance_mask[index_org]
        else:
            coord_world, index_org = voxel_sample_points(coord_world, point_number=int(self._point_num*1.1))
            segment_mask = segment_mask[index_org]
            instance_mask = instance_mask[index_org]
        # view_object_joint(pcd_world, instance_mask, joint_bases, joint_directions)
        joint_ends = joint_axises + joint_bases
        pcd_camera = translate_pc_world_to_camera(coord_world, extrinsic)
        joint_bases = translate_joint_base_world_to_camera(joint_bases, extrinsic)
        joint_ends = translate_joint_direc_world_to_camera(joint_ends, extrinsic)
        if self._noise:
            random_scale = np.random.uniform(-0.05, 0.05)
        else:
            random_scale = 0.01
        point_could_center, center, scale = point_cloud_center_and_scale(pcd_camera, random_scale=random_scale)
        joint_bases = (joint_bases - center) / scale
        joint_ends = (joint_ends - center) / scale
        joint_directions = joint_ends - joint_bases
        joint_directions = joint_directions / np.linalg.norm(joint_directions, axis=1, keepdims=True)
        num_instances = int(instance_mask.max()) + 1
        joint_trans = np.zeros((pcd_camera.shape[0], 3), dtype=np.float32)
        joint_dirs = np.zeros((pcd_camera.shape[0], 3), dtype=np.float32)
        joint_offset_unitvecs = np.zeros((pcd_camera.shape[0], 3), dtype=np.float32)
        joint_offset_heatmaps = np.zeros((pcd_camera.shape[0], 1), dtype=np.float32)
        joint_proj_vecs = np.zeros((pcd_camera.shape[0], 3), dtype=np.float32)
        instance_regions = np.zeros((pcd_camera.shape[0], 9), dtype=np.float32)
        for instance_id in range(num_instances):
            indices = np.where(instance_mask == instance_id)[0]
            if indices.shape[0] == 0:
                print("no instance: {}".format(instance_id))
                continue
            if instance_id == 0:
                joint_trans[indices] = np.array([0, 0, 0])
                joint_dirs[indices] = np.array([0, 0, 0])
                joint_offset_unitvecs[indices] = np.array([0, 0, 0])
                joint_offset_heatmaps[indices] = 0
                joint_proj_vecs[indices] = np.array([0, 0, 0])
            else:
                joint_trans[indices] = joint_bases[instance_id - 1]
                joint_dirs[indices] = joint_directions[instance_id - 1]
                part_pcd = point_could_center[indices, :3]
                heatmap, unitvec, proj_vec = self.cal_joint_to_part_offset(part_pcd, joint_bases[instance_id - 1], joint_directions[instance_id - 1])
                joint_offset_unitvecs[indices] = unitvec
                joint_offset_heatmaps[indices] = heatmap
                joint_proj_vecs[indices] = proj_vec
            xyz_i = point_could_center[indices, :3]
            min_i = xyz_i.min(0)
            max_i = xyz_i.max(0)
            mean_i = xyz_i.mean(0)
            instance_regions[indices, 0:3] = mean_i
            instance_regions[indices, 3:6] = min_i
            instance_regions[indices, 6:9] = max_i
        point_center_offset = instance_regions[:, :3] - point_could_center
        # view_point_cloud_parts_and_joints(point_could_center, instance_mask, joint_dirs, joint_proj_vecs, point_center_offset)
        point_cloud_dim_min = point_could_center.min(axis=0)
        point_cloud_dim_max = point_could_center.max(axis=0)
        feat = copy.deepcopy(point_could_center)
        return {
            "coords": point_could_center,
            "scale": scale,
            "center": center,
            "point_center_offsets": point_center_offset,
            "feats": feat,
            "point_num": point_center_offset.shape[0],
            "sem_labels": segment_mask,
            "instance_labels": instance_mask,
            "joint_directions": joint_dirs,
            "joint_proj_vecs": joint_proj_vecs,
            "file_id": file_path,
            "point_cloud_dim_min": point_cloud_dim_min,
            "point_cloud_dim_max": point_cloud_dim_max,
        }

if __name__ == '__main__':
    root = ["/hdd/gamma/val"]
    # root = ["/aidata/qiaojun/train_data/gamma/train"]
    dataset = GammaDataset(root=root)
    print("data total: ", dataset.__len__())
    for i in range(dataset.__len__()):
        print("data num: ", i)
        data = dataset.__getitem__(i, down_sample=True)
        # print("heat_map_max: ", data["joint_offset_heatmaps"].max(),  data["joint_offset_heatmaps"].min(), data["scale"])


