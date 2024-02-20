import pdb
import os
from visual_model.object_articulation_part import gamma_model_net
import argparse
from datasets.gamma_dataset import GammaDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from typing import List, Tuple, Dict, Optional, Any, Union
import random
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
joint_types = {"revolute": 0, "prismatic": 1, "fixed": 2}

def collate_fn(batch: List[Dict]):
    keys = batch[0].keys()
    return {
        key: default_collate([item[key]for item in batch])
        if batch[0][key] is not None
        else None
        for key in keys
    }

def get_train_loader(args):
    train_dataset = GammaDataset(args.train_data_path, point_num=args.point_num)
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    return loader

def get_test_loader(args):
    dataset = GammaDataset(args.test_data_path, point_num=args.point_num)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    return loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args, writer):
    model = gamma_model_net(in_channel=args.in_channels, num_point=int(args.point_num), num_classes=int(args.num_classes), device=args.device).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch - args.warm_up, eta_min=args.lr/100.0)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=int(args.warm_up), after_scheduler=scheduler)
    optimizer.zero_grad()
    optimizer.step()
    scheduler_warmup.step()
    train_step = 0
    min_loss = 100
    if not args.train:
        test_load = get_test_loader(args)
        assert os.path.exists(args.model_path)
        print("load model from path:", args.model_path)
        model.load_state_dict(torch.load(args.model_path))
        object_cats = {}
        for sample in tqdm(test_load):
            model.eval()
            with torch.no_grad():
                object_result = model.evaluate(sample, ignore_label=args.ignore_label)
                object_cat_id = object_result["object_cat_id"]
                object_cat = object_cat_id.split("_")[0]
                if object_cat not in object_cats:
                    object_cats[object_cat] = {"angle_errors": [], "trans_errors": [], "ious": [], "joint_type_accu": []}

                for angle_error in object_result["angle_errors"]:
                    object_cats[object_cat]["angle_errors"] += object_result["angle_errors"][angle_error]

                for trans_error in object_result["trans_errors"]:
                    if trans_error == joint_types["revolute"]:
                        object_cats[object_cat]["trans_errors"] += object_result["trans_errors"][trans_error]

                for iou in object_result["ious"]:
                    object_cats[object_cat]["ious"] += object_result["ious"][iou]

                for joint_type_accu in object_result["joint_type_accu"]:
                    object_cats[object_cat]["joint_type_accu"] += object_result["joint_type_accu"][joint_type_accu]

        for object_cat in object_cats:
            print("*" * 100)
            print("object_cat: ", object_cat)
            direction_errors = np.array(object_cats[object_cat]["angle_errors"])
            print("mean direction_errors:", np.mean(direction_errors))
            trans_errors = np.array(object_cats[object_cat]["trans_errors"])
            print("mean trans_errors:", np.mean(trans_errors))
            ious = np.array(object_cats[object_cat]["ious"])
            ious = np.array(ious)
            print("mean ious:", np.mean(ious))
            print("part ious 50 :{} ".format((ious > 0.5).sum() / ious.shape[0]))
            print("part ious 75 :{} ".format((ious > 0.5).sum() / ious.shape[0]))
            joint_type_acc = object_cats[object_cat]["joint_type_accu"]
            joint_type_acc = np.array(joint_type_acc)
            print("joint_type_acc :{} ".format(np.mean(joint_type_acc)))
        return

    train_load = get_train_loader(args)
    test_load = get_test_loader(args)
    for epoch in tqdm(range(int(args.train_epoch))):
        model.train()
        epoch_loss = 0
        sem_acc = 0
        pixel_accu = 0
        epoch_step = 0
        # train
        for sample in tqdm(train_load):
            train_step = train_step + 1
            if epoch > args.offset_start:
                losses, result_dict = model.get_loss(sample, sem_only=False, ignore_label=args.ignore_label)
            else:
                losses, result_dict = model.get_loss(sample, sem_only=True, ignore_label=args.ignore_label)
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            epoch_loss += losses["total_loss"].item()
            sem_acc += result_dict["sem_all_accu"].item()
            pixel_accu += result_dict["pixel_accu"].item()
            epoch_step += 1
            loss_keys = losses.keys()
            for loss_key in loss_keys:
                writer.add_scalar("train/" + loss_key, losses[loss_key].item(), train_step)
            if train_step % 200 == 0:
                for loss_key in loss_keys:
                    print("train epoch: {}, step: {}  {}: {}".format(epoch, train_step, loss_key, losses[loss_key].item()))
                for result_key in result_dict:
                    print("train epoch: {}, step: {} {}: {}".format(epoch, train_step, result_key, result_dict[result_key].item()))
        print("train epoch:{} loss:{} sem_acc:{} pixel_accu:{}".format(epoch, epoch_loss/epoch_step, sem_acc/epoch_step, pixel_accu/epoch_step))
        writer.add_scalar("train/epoch_loss", epoch_loss / epoch_step, epoch)
        writer.add_scalar("train/epoch_sem_accu", sem_acc / epoch_step, epoch)
        writer.add_scalar("train/epoch_pixel_accu", pixel_accu / epoch_step, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        scheduler_warmup.step()
        # test
        if epoch > 0 and epoch % 5 == 0 or epoch == 0:
            model.eval()
            epoch_loss = 0
            sem_acc = 0
            pixel_accu = 0
            epoch_step = 0
            for sample in tqdm(test_load):
                if epoch > args.offset_start:
                    losses, result_dict = model.get_loss(sample, sem_only=False, ignore_label=args.ignore_label)
                else:
                    losses, result_dict = model.get_loss(sample, sem_only=True, ignore_label=args.ignore_label)
                optimizer.zero_grad()
                losses["total_loss"].backward()
                optimizer.step()
                epoch_loss += losses["total_loss"].item()
                sem_acc += result_dict["sem_all_accu"].item()
                pixel_accu += result_dict["pixel_accu"].item()
                epoch_step += 1
            writer.add_scalar("test/epoch_loss", epoch_loss / epoch_step, epoch)
            writer.add_scalar("test/epoch_sem_accu", sem_acc / epoch_step, epoch)
            writer.add_scalar("test/epoch_pixel_accu", pixel_accu / epoch_step, epoch)
            print("test epoch:{} loss:{} sem_acc:{} pixel_accu:{}".format(epoch, epoch_loss/epoch_step, sem_acc/epoch_step, pixel_accu/epoch_step))
            if epoch_loss/epoch_step < min_loss and epoch > 0:
                print("save fine-tune model, loss: ", epoch_loss/epoch_step)
                min_loss = epoch_loss/epoch_step
                save_path = args.log_dir + "/best.pth"
                print("best model: ", save_path, " save path: ", save_path)
                torch.save(model.state_dict(), save_path)
            if epoch % 10 == 0 and epoch > 0:
                save_path = args.log_dir + "/epoch" + str(epoch) + "_.pth"
                print("best model: ", save_path, " save path: ", save_path)
                torch.save(model.state_dict(), save_path)
        if epoch == args.train_epoch - 1:
            save_path = args.log_dir + "/last.pth"
            print("best model: ", save_path, " save path: ", save_path)
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bbox object')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train_data_path', type=str, default=["/hdd/gamma/train/"])
    parser.add_argument('--test_data_path', type=str, default=["/hdd/gamma/test"])
    parser.add_argument('--pointnet_type', type=str, default="pointnet2_msg")
    parser.add_argument('--point_num', type=int, default=10000)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument("--ignore_label", type=int, help="fixed part label", default=2)
    parser.add_argument("--warm_up", type=int, help="lr warm up", default=10)
    parser.add_argument("--offset_start", type=int, help="offset loss", default=5)
    parser.add_argument("--train_epoch", type=int, help="train epoch", default=101)
    parser.add_argument("--batch_size", type=int, help="data loader batch size", default=16)
    parser.add_argument("--num_workers", type=int, help="data loader num_workers", default=4)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("--device", type=str, help="cuda or cpu", default="cuda")
    parser.add_argument("--log_dir", type=str, help="log dir", default="./train_process/gamma/pointnet_articulation_model/")
    parser.add_argument("--model_path", type=str, help="check point", default="./train_process/gamma/pointnet_articulation_model/2024-02-20-10_pointnet2_msg_16/best.pth")
    args = parser.parse_args()
    if args.train:
        args.log_dir = args.log_dir + time.strftime("%Y-%m-%d-%H", time.localtime()) + "_" + args.pointnet_type + "_" + str(args.batch_size)
        os.makedirs(args.log_dir, exist_ok=True)
        print("Logging:", args.log_dir)
        writer = SummaryWriter(log_dir=args.log_dir, )
    else:
        writer = None
    setup_seed(args.seed)
    main(args, writer)