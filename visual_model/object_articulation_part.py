from .pointnet2 import Point2SemSegMSGNetfeat
import torch.nn as nn
import torch
from .losses import focal_loss, dice_loss
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from .pointnet2_utils import index_points, index_masks
from .pcd_utils import point_cloud_center_and_scale, radius_based_denoising_numpy, view_point_cloud_parts_and_joint, voxel_sample_points

class gamma_model_net(nn.Module):
    def __init__(self, in_channel=3, num_classes=3, num_point=10000, device="cuda"):
        super(gamma_model_net, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.num_point = num_point
        self.device = device
        self.point_feat = Point2SemSegMSGNetfeat(in_channel=self.in_channel)
        self.sem_head = nn.Sequential(nn.Conv1d(128, self.num_classes, kernel_size=1, padding=0, bias=False))
        self.offset_feature = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(True)
        )
        self.offset_head = nn.Linear(128, 3)
        self.joint_feature = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(True),
        )
        self.joint_direction = nn.Linear(128, 3)
        self.joint_proj_vec = nn.Linear(128, 3)

    def forward(self, point_cloud):
        pc_feature = self.point_feat(point_cloud)
        sem_logits = self.sem_head(pc_feature).transpose(1, 2)
        point_center_offsets_feature = self.offset_feature(pc_feature).transpose(1, 2)
        point_center_offsets = torch.tanh(self.offset_head(point_center_offsets_feature))
        joint_feature = self.joint_feature(pc_feature).transpose(1, 2)
        joint_proj_vecs = torch.tanh(self.joint_proj_vec(joint_feature))
        joint_directions = torch.tanh(self.joint_direction(joint_feature))
        pre_data = dict()
        pre_data["sem_logits"] = sem_logits
        pre_data["point_center_offsets"] = point_center_offsets
        pre_data["joint_directions"] = joint_directions
        pre_data["joint_proj_vecs"] = joint_proj_vecs
        pre_data["point_center_offsets_feature"] = point_center_offsets_feature
        pre_data["joint_features"] = joint_feature
        pre_data["pc_features"] = pc_feature.transpose(1, 2)
        return pre_data

    def offset_loss(self, pre_offset, gt_offsets, valid_mask):
        gt_offsets = gt_offsets.reshape(-1, 3)
        pre_offset = pre_offset.reshape(-1, 3)
        pt_diff = pre_offset - gt_offsets
        pt_dist = torch.sum(pt_diff.abs(), dim=-1)
        loss_pt_offset_dist = pt_dist[valid_mask].mean()
        gt_offsets_norm = torch.norm(gt_offsets, dim=1).reshape(-1, 1)
        gt_offsets = gt_offsets/(gt_offsets_norm + 1e-8)
        pre_offsets_norm = torch.norm(pre_offset, dim=1).reshape(-1, 1)
        pre_offset = pre_offset/(pre_offsets_norm + 1e-8)
        dir_diff = -(gt_offsets * pre_offset).sum(dim=-1)
        loss_offset_dir = dir_diff[valid_mask].mean()
        loss_offset = loss_offset_dir + loss_pt_offset_dist
        return loss_offset

    def offset_and_direction_orthogonal_loss(self, pre_pro_offset, pre_direction, valid_mask):
        pre_pro_offset = pre_pro_offset.reshape(-1, 3)
        pre_direction = pre_direction.reshape(-1, 3)
        dot_product = torch.sum(pre_pro_offset * pre_direction, dim=1)
        dot_product_abs = torch.abs(dot_product)
        orthogonal_loss = dot_product_abs[valid_mask].mean()
        norms_product = torch.norm(pre_pro_offset, dim=1) * torch.norm(pre_direction, dim=1)
        cosine_of_angle = dot_product / norms_product
        angles_rad = torch.acos(cosine_of_angle[valid_mask])
        angles_deg = torch.rad2deg(angles_rad)
        return orthogonal_loss, angles_deg

    def loss_sem_seg(self, sem_logits: torch.Tensor, sem_labels: torch.Tensor,) -> torch.Tensor:
        loss = focal_loss(sem_logits, sem_labels, alpha=None, gamma=2.0, reduction="mean")
        loss += dice_loss(sem_logits[:, :, None, None], sem_labels[:, None, None],)
        return loss

    def get_loss(self, data_dict, sem_only=False, ignore_label=0):
        coords = data_dict["coords"].to(self.device, dtype=torch.float)
        if coords.shape[1] > self.num_point:
            # FPS sample
            fps_pcs_idx = furthest_point_sample(coords, self.num_point).cpu().numpy()
            feat_per_point = index_points(data_dict["feats"], fps_pcs_idx).transpose(1, 2).to(self.device, dtype=torch.float)
            data_dict["sem_labels"] = index_masks(data_dict["sem_labels"], fps_pcs_idx)
            data_dict["point_center_offsets"] = index_points(data_dict["point_center_offsets"], fps_pcs_idx)
            data_dict["joint_directions"] = index_points(data_dict["joint_directions"], fps_pcs_idx)
            data_dict["joint_proj_vecs"] = index_points(data_dict["joint_proj_vecs"], fps_pcs_idx)
        else:
            feat_per_point = data_dict["feats"].transpose(1, 2).to(self.device, dtype=torch.float)
        pred = self.forward(feat_per_point)
        valid_mask = data_dict["sem_labels"] != ignore_label
        valid_mask = valid_mask.reshape(-1)
        # part sem
        sem_mask_loss = self.loss_sem_seg(pred["sem_logits"].reshape(-1, self.num_classes), data_dict["sem_labels"].reshape(-1).to(self.device, dtype=torch.long))
        point_center_offset_loss = self.offset_loss(pred["point_center_offsets"], data_dict["point_center_offsets"].to(self.device, dtype=torch.float), valid_mask=valid_mask)
        # joint pose
        joint_direction_loss = self.offset_loss(pred["joint_directions"], data_dict["joint_directions"].to(self.device, dtype=torch.float), valid_mask=valid_mask)
        joint_proj_vec_loss = self.offset_loss(pred["joint_proj_vecs"], data_dict["joint_proj_vecs"].to(self.device, dtype=torch.float), valid_mask=valid_mask)
        orthogonal_loss, angles_deg = self.offset_and_direction_orthogonal_loss(pred["joint_proj_vecs"], pred["joint_directions"], valid_mask=valid_mask)
        # error
        if sem_only:
            total_loss = sem_mask_loss
        else:
            total_loss = sem_mask_loss + point_center_offset_loss + joint_direction_loss + joint_proj_vec_loss
        loss_dict = dict()
        result_dict = dict()
        sem_preds = torch.argmax(pred["sem_logits"].reshape(-1, self.num_classes).detach(), dim=-1)
        sem_labels = data_dict["sem_labels"].reshape(-1).to(self.device, dtype=torch.long)
        sem_all_accu = (sem_preds == sem_labels).sum().float()/(sem_preds.shape[0])
        pixel_accu = (sem_preds[valid_mask] == sem_labels[valid_mask]).sum().float()/(sem_preds[valid_mask].shape[0])
        loss_dict["function_loss"] = sem_mask_loss
        loss_dict["total_loss"] = total_loss
        loss_dict["point_center_offset_loss"] = point_center_offset_loss
        loss_dict["joint_direction_loss"] = joint_direction_loss
        loss_dict["joint_proj_vec_loss"] = joint_proj_vec_loss
        loss_dict["orthogonal_loss"] = orthogonal_loss
        loss_dict["orthogonal_angles_deg"] = (90 - angles_deg).mean()
        result_dict["sem_all_accu"] = sem_all_accu
        result_dict["pixel_accu"] = pixel_accu
        return loss_dict, result_dict


    @torch.no_grad()
    def evaluate(self, data_dict, cluster="joint_and_center", ignore_label=0, eps=0.1, min_samples=100):
        coords = data_dict["coords"].to(self.device, dtype=torch.float)
        assert coords.shape[0] == 1
        if coords.shape[1] > self.num_point:
            fps_pcs_idx = furthest_point_sample(coords, self.num_point).cpu().numpy()
            feat_per_point = index_points(data_dict["feats"], fps_pcs_idx).transpose(1, 2).to(self.device,dtype=torch.float)
            data_dict["sem_labels"] = index_masks(data_dict["sem_labels"], fps_pcs_idx)
            data_dict["instance_labels"] = index_masks(data_dict["instance_labels"], fps_pcs_idx)
            data_dict["point_center_offsets"] = index_points(data_dict["point_center_offsets"], fps_pcs_idx)
            data_dict["joint_directions"] = index_points(data_dict["joint_directions"], fps_pcs_idx)
            data_dict["joint_proj_vecs"] = index_points(data_dict["joint_proj_vecs"], fps_pcs_idx)
            data_dict["coords"] = index_points(data_dict["coords"], fps_pcs_idx)
        else:
            feat_per_point = data_dict["feats"].transpose(1, 2).to(self.device, dtype=torch.float)
        coord_per_point = data_dict["coords"]
        pred = self.forward(feat_per_point)
        valid_mask = data_dict["sem_labels"] != ignore_label
        valid_mask = valid_mask.reshape(-1)
        # finish sample
        object_cat_id = data_dict["file_id"][0].split("/")[-2]
        scale = data_dict["scale"][0].cpu().numpy()
        center = data_dict["center"][0].cpu().numpy()
        sem_preds = torch.argmax(pred["sem_logits"].reshape(-1, self.num_classes).detach(), dim=-1)
        sem_preds = sem_preds.detach().cpu().numpy()
        pre_function_mask = torch.argmax(pred["sem_logits"].reshape(-1, self.num_classes).detach(), dim=-1)
        pre_function_mask = pre_function_mask.detach().cpu().numpy()
        pre_joint_proj_vecs = pred["joint_proj_vecs"].detach().cpu().numpy()[0]
        pre_joint_directions = pred["joint_directions"].detach().cpu().numpy()[0]
        pre_point_center_offsets = pred["point_center_offsets"].detach().cpu().numpy()[0]
        gt_instance_labels = data_dict["instance_labels"].detach().cpu().numpy()[0]
        gt_function_masks = data_dict["sem_labels"].detach().cpu().numpy()[0]
        gt_joint_proj_vecs = data_dict["joint_proj_vecs"].detach().cpu().numpy()[0]
        gt_joint_directions = data_dict["joint_directions"].detach().cpu().numpy()[0]
        pcd = coord_per_point.detach().cpu().numpy()[0]

        # cal sem accus
        sem_accu = (sem_preds == gt_function_masks).sum() / gt_function_masks.shape[0]
        pixel_accu = (sem_preds[valid_mask] == gt_function_masks[valid_mask]).sum()/gt_function_masks[valid_mask].shape[0]

        result = []
        instance_errors = dict()
        pre_instance_id = 0
        pre_instance_labels = np.zeros_like(pre_function_mask)
        function_mask_ids = np.unique(pre_function_mask)
        for function_mask_id in function_mask_ids:
            if function_mask_id == ignore_label:
                continue
            if (pre_function_mask == function_mask_id).sum() < 30:
                continue
            part_indexs = np.where(pre_function_mask == function_mask_id)[0]
            part_pcd = pcd[pre_function_mask == function_mask_id]
            gt_instance_parts = gt_instance_labels[pre_function_mask == function_mask_id]
            pre_part_point_center_offsets = pre_point_center_offsets[pre_function_mask == function_mask_id]
            pre_part_joint_proj_vecs = pre_joint_proj_vecs[pre_function_mask == function_mask_id]
            pre_part_joint_directions = pre_joint_directions[pre_function_mask == function_mask_id]
            pre_pcd_to_center = part_pcd + pre_part_point_center_offsets
            pre_pcd_to_joint = part_pcd + pre_part_joint_proj_vecs
            if cluster == "joint_and_center":
                pcd_feature = np.hstack((pre_pcd_to_joint, pre_pcd_to_center))
            if cluster == "center":
                pcd_feature = pre_pcd_to_center
            if cluster == "joint":
                pcd_feature = pre_pcd_to_joint
            part_labels = self.sklearn_cluster(pcd_feature, eps=eps, min_samples=min_samples)
            part_instance_mask = np.zeros_like(part_labels)
            for part_id in range(part_labels.max() + 1):
                pre_instance_id = pre_instance_id + 1
                pre_joint_axis = np.median(pre_part_joint_directions[part_labels == part_id], axis=0)
                pre_joint_axis = pre_joint_axis / np.linalg.norm(pre_joint_axis)
                pred_joint_pts = pre_pcd_to_joint[part_labels == part_id]
                pred_joint_pt = np.median(pred_joint_pts, axis=0)
                result.append({"joint_type": function_mask_id, "joint_translation": pred_joint_pt, "joint_direction": pre_joint_axis})

                gt_instance_part = gt_instance_parts[part_labels == part_id]
                instance_id = np.argmax(np.bincount(gt_instance_part))
                gt_part_types = gt_function_masks[gt_instance_labels == instance_id]
                gt_part_type = np.argmax(np.bincount(gt_part_types))

                if gt_part_type == ignore_label:
                    continue
                elif gt_part_type == function_mask_id:
                    joint_type_acc = 1
                else:
                    joint_type_acc = 0

                part_instance_mask[part_labels == part_id] = pre_instance_id
                gt_part_joint_direction = gt_joint_directions[gt_instance_labels == instance_id]
                gt_joint_axis = np.median(gt_part_joint_direction, axis=0)
                if np.linalg.norm(gt_joint_axis) == 0:
                    continue

                gt_joint_axis = gt_joint_axis / np.linalg.norm(gt_joint_axis)
                gt_part_joint_proj_vecs = gt_joint_proj_vecs[gt_instance_labels == instance_id]
                part_pcd = pcd[gt_instance_labels == instance_id]
                gt_joint_pts = gt_part_joint_proj_vecs + part_pcd
                gt_joint_pt = np.median(gt_joint_pts, axis=0)

                # gt
                joint_end = gt_joint_pt + gt_joint_axis
                gt_joint_pt = gt_joint_pt * scale + center
                joint_end = joint_end * scale + center
                gt_joint_axis = joint_end - gt_joint_pt
                gt_joint_axis = gt_joint_axis / np.linalg.norm(gt_joint_axis)
                # pre
                joint_end = pred_joint_pt + pre_joint_axis
                pred_joint_pt = pred_joint_pt * scale + center
                joint_end = joint_end * scale + center
                pre_joint_axis = joint_end - pred_joint_pt
                pre_joint_axis = pre_joint_axis / np.linalg.norm(pre_joint_axis)
                angle_error = self.direction_error(gt_joint_axis, pre_joint_axis)
                if angle_error > 90:
                    angle_error = 180 - angle_error
                trans_error = self.dist_between_3d_lines(pred_joint_pt, gt_joint_pt, pre_joint_axis, gt_joint_axis)
                instance_errors[instance_id] = {"pre_pcd_num": (part_labels == part_id).sum(),
                                                "joint_type_acc": joint_type_acc,
                                                "real_pcd_num": (gt_instance_labels == instance_id).sum(),
                                                "gt_instance": instance_id, "angle_error": angle_error,
                                                "trans_error": trans_error, "pre_instance_id": pre_instance_id,
                                                "pre_joint_type": function_mask_id, "gt_joint_type": gt_part_type}
            pre_instance_labels[part_indexs] = part_instance_mask
        # view_point_cloud_parts(coord_per_point.reshape(-1, 3), pre_instance_labels)
        mean_iou, part_ious, gt_part_types, pre_part_types, gt_instances, pre_instances = self.calculate_point_cloud_part_iou(pre_instance_labels,
                                                                                          gt_instance_labels,
                                                                                          gt_function_masks,
                                                                                          pre_function_mask,
                                                                                          ignore_label)
        angle_errors = {}
        trans_errors = {}
        ious = {}
        joint_type_accus = {}
        if part_ious is not None:
            for part_iou, gt_part_type, pre_part_type in zip(part_ious, gt_part_types, pre_part_types):
                if gt_part_type == ignore_label:
                    continue
                if gt_part_type not in ious.keys():
                    ious[gt_part_type] = []
                if pre_part_type == gt_part_type:
                    ious[gt_part_type].append(part_iou)
                else:
                    ious[gt_part_type].append(0)
            for gt_instance_label in gt_instances:
                if gt_instance_label in instance_errors.keys():
                    instance_error = instance_errors[gt_instance_label]
                    if instance_error["gt_joint_type"] not in trans_errors.keys():
                        trans_errors[instance_error["gt_joint_type"]] = []
                        angle_errors[instance_error["gt_joint_type"]] = []
                        joint_type_accus[instance_error["gt_joint_type"]] = []
                    trans_errors[instance_error["gt_joint_type"]].append(instance_error["trans_error"])
                    angle_errors[instance_error["gt_joint_type"]].append(instance_error["angle_error"])
                    joint_type_accus[instance_error["gt_joint_type"]].append(instance_error["joint_type_acc"])

        object_result = {"angle_errors": angle_errors, "trans_errors": trans_errors, "object_cat_id": object_cat_id,
                         "sem_accu": sem_accu, "pixel_accu": pixel_accu, "ious": ious,
                         "joint_type_accu": joint_type_accus}
        return object_result

    @torch.no_grad()
    def online_inference(self, camera_pcd, view_res=False, denoise=True, cluster_eps=0.08, num_point_min=100, joint_type_to_name=False, ignore_label=0):
        if denoise:
            camera_pcd = radius_based_denoising_numpy(camera_pcd)
        camera_pcd, voxel_centroids = voxel_sample_points(camera_pcd, point_number=int(self.num_point * 1.25))
        point_cloud, center, scale = point_cloud_center_and_scale(camera_pcd)
        camcs_per_point = torch.from_numpy(point_cloud).to(self.device, dtype=torch.float).unsqueeze(0)
        fps_pcs_idx = furthest_point_sample(camcs_per_point, self.num_point).cpu().numpy()
        camera_pcd = camera_pcd[fps_pcs_idx[0]]
        camcs_per_point = index_points(camcs_per_point, fps_pcs_idx).transpose(1, 2)
        pred = self.forward(camcs_per_point)
        pre_function_mask = torch.argmax(pred["sem_logits"].reshape(-1, self.num_classes).detach(), dim=-1)
        pre_function_mask = pre_function_mask.detach().cpu().numpy()
        pre_joint_proj_vecs = pred["joint_proj_vecs"].detach().cpu().numpy()[0]
        pre_joint_directions = pred["joint_directions"].detach().cpu().numpy()[0]
        pre_point_center_offsets = pred["point_center_offsets"].detach().cpu().numpy()[0]
        point_cloud = camcs_per_point.detach().cpu().numpy()[0].transpose(1, 0)
        function_mask_ids = np.unique(pre_function_mask)
        results = []
        instance_id = 0
        instance_labels = np.zeros_like(pre_function_mask)
        for function_mask_id in function_mask_ids:
            if function_mask_id == ignore_label:
                continue
            if (pre_function_mask == function_mask_id).sum() < num_point_min:
                continue
            part_indexs = np.where(pre_function_mask == function_mask_id)[0]
            part_pcd = point_cloud[pre_function_mask == function_mask_id]
            pre_part_joint_direction = pre_joint_directions[pre_function_mask == function_mask_id]
            pre_part_joint_proj_vecs = pre_joint_proj_vecs[pre_function_mask == function_mask_id]
            pre_part_offset = pre_point_center_offsets[pre_function_mask == function_mask_id]
            pre_pcd_to_center = part_pcd + pre_part_offset
            pre_pcd_to_joint = part_pcd + pre_part_joint_proj_vecs
            pcd_feature = np.hstack((pre_pcd_to_joint, pre_pcd_to_center))
            part_labels = self.sklearn_cluster(pcd_feature, eps=cluster_eps)
            part_num = part_labels.max() + 1
            part_instance_mask = np.zeros_like(part_labels)
            for part_id in range(part_num):
                pre_joint_axis = np.median(pre_part_joint_direction[part_labels == part_id], axis=0)
                pre_joint_axis = pre_joint_axis / np.linalg.norm(pre_joint_axis)
                pred_joint_pts = pre_pcd_to_joint[part_labels == part_id]
                pred_joint_pt = np.median(pred_joint_pts, axis=0)
                joint_end = pred_joint_pt + pre_joint_axis
                pred_joint_pt = pred_joint_pt * scale + center
                joint_end = joint_end * scale + center
                pre_joint_axis = joint_end - pred_joint_pt
                pre_joint_axis = pre_joint_axis / np.linalg.norm(pre_joint_axis)
                instance_id += 1
                part_instance_mask[part_labels == part_id] = instance_id
                if (part_labels == part_id).sum() > num_point_min:
                    if joint_type_to_name:
                        if function_mask_id == 0:
                            joint_type = "revolute"
                        if function_mask_id == 1:
                            joint_type = "primastic"
                    else:
                        joint_type = function_mask_id
                    results.append(
                        {"joint_type": joint_type, "joint_translation": pred_joint_pt, "instance_id": instance_id,
                         "joint_direction": pre_joint_axis, "part_point_num": (part_labels == part_id).sum()})
            instance_labels[part_indexs] = part_instance_mask
        if view_res:
            view_point_cloud_parts_and_joint(camera_pcd, instance_labels, results)
        return results, instance_labels, camera_pcd

    def dist_between_3d_lines(self, p1, p2, e1, e2):
        orth_vect = np.cross(e1, e2)
        p = p1 - p2
        if np.linalg.norm(orth_vect) == 0:
            dist = np.linalg.norm(np.cross(p, e1)) / np.linalg.norm(e1)
        else:
            dist = np.linalg.norm(np.dot(orth_vect, p)) / np.linalg.norm(orth_vect)
        return dist

    def direction_error(self, e1, e2):
        cos_theta = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_radian = np.arccos(cos_theta) * 180 / np.pi
        return angle_radian

    def sklearn_cluster(self, point_cloud, eps=0.1, min_samples=100):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(point_cloud)
        return labels

    def calculate_point_cloud_part_iou(self, predicted_clusters, ground_truth_clusters, gt_sem_masks, pre_sem_masks, ignore_label=0):
        ground_truth_clusters = ground_truth_clusters
        predict_masks = np.unique(predicted_clusters)
        predict_masks = predict_masks[predict_masks != 0]
        gt_masks = np.unique(ground_truth_clusters)
        fixed_part_id = 0
        for gt_mask in gt_masks:
            if gt_sem_masks[ground_truth_clusters == gt_mask][0] == ignore_label:
                fixed_part_id = gt_mask
        gt_masks = gt_masks[gt_masks != fixed_part_id]
        iou_matrix = np.zeros((len(gt_masks), len(predict_masks)))
        gt_instance_sem_labels = np.zeros((len(gt_masks)))
        pre_instance_sem_labels = np.zeros((len(predict_masks)))
        for i in range(len(gt_masks)):
            for j in range(len(predict_masks)):
                intersection = (predicted_clusters == predict_masks[j]) & (ground_truth_clusters == gt_masks[i])
                union = (predicted_clusters == predict_masks[j]) | (ground_truth_clusters == gt_masks[i])
                iou = intersection.sum()/union.sum()
                iou_matrix[i, j] = iou
                part_type = np.argmax(np.bincount(gt_sem_masks[ground_truth_clusters == gt_masks[i]]))
                gt_instance_sem_labels[i] = part_type
                pre_instance_sem_labels[j] = pre_sem_masks[predicted_clusters == predict_masks[j]][0]
        row_inds, col_inds = linear_sum_assignment(-iou_matrix)
        if len(row_inds) > 0:
            gt_instances = gt_masks[row_inds]
            pre_instances = predict_masks[col_inds]
            mean_iou = iou_matrix[row_inds, col_inds].sum() / len(row_inds)
            part_ious = iou_matrix[row_inds, col_inds]
            gt_part_types = gt_instance_sem_labels[row_inds]
            pre_part_types = pre_instance_sem_labels[col_inds]
            return mean_iou, part_ious, gt_part_types, pre_part_types, gt_instances, pre_instances
        else:
            return None, None, None, None, None, None

