import glob
import os
from visual_model.object_articulation_part import gamma_model_net
import argparse
import torch
import numpy as np
from datasets.data_utilts import translate_pc_world_to_camera

def main(args):
    model = gamma_model_net(in_channel=args.in_channels, num_point=int(args.num_point), num_classes=int(args.num_classes), device=args.device).to(args.device)
    assert os.path.exists(args.model_path)
    print("load model from path:", args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    files = glob.glob("./example_data/*.npz")
    for path in files:
        file = np.load(path, allow_pickle=True)
        pcd_world = file["pcd_world"]
        extrinsic = file["extrinsic"]
        pcd_camera = translate_pc_world_to_camera(pcd_world, extrinsic)
        with torch.no_grad():
            model.eval()
            results, instance_labels, camera_pcd = model.online_inference(camera_pcd=pcd_camera, view_res=True, ignore_label=args.ignore_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bbox object')
    parser.add_argument('--model_path', type=str, default="./checkpoint/best.pth")
    parser.add_argument('--num_point', type=int, default=10000)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--ignore_label', type=int, default=2)
    parser.add_argument("--device", type=str, help="cuda or cpu", default="cuda")
    args = parser.parse_args()
    main(args)