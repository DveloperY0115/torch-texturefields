"""
texturefield_conditional_trainer.py - Trainer for TextureFields conditional model.
"""

import os
from PIL import Image
from typing import Dict, Iterable, Tuple
from collections import namedtuple
from tqdm import tqdm

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from model.texture_field import TextureFieldsCls
from util.dataset import ShapeNetSingleClassDataset

from .base_trainer import BaseTrainer


class TextureFieldsConditionalTrainer(BaseTrainer):
    def __init__(self, opts: namedtuple, checkpoint: str = None):
        super().__init__(opts=opts)

        self.model = TextureFieldsCls(self.opts.experiment_setting, self.device).to(self.device)

        if self.opts.use_multi_gpu:
            self.model = nn.DataParallel(self.model)

        # optimizer & learning rate scheduler
        self.optimizer = self.configure_optimizer()
        self.lr_scheduler = self.configure_lr_scheduler()

        # dataset & data loader
        self.train_dataset, self.test_dataset = self.configure_dataset()
        self.train_loader, self.test_loader = self.configure_dataloader()

        # load checkpoint if available
        if checkpoint is not None:
            if self.load_checkpoint(checkpoint):
                print("[!] Successfully loaded checkpoint at {}".format(checkpoint))
            else:
                print("[!] Failed to load checkpoint at {}".format(checkpoint))

        # initialize W&B if requested
        if self.opts.log_wandb:
            wandb.init(project="torch-TextureFields")

    def train(self):
        print("Begin training..")

        for self.epoch in tqdm(range(self.initial_epoch, self.opts.num_epoch)):
            train_loss = self.train_one_epoch()
            test_loss = self.test_one_epoch()

            print("=======================================")
            print("Epoch {}".format(self.epoch))
            print("Training loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))
            print("=======================================")

            if self.opts.log_wandb:
                wandb.log({"Loss/Train": train_loss}, step=self.epoch)
                wandb.log({"Loss/Test": test_loss}, step=self.epoch)

            if (self.epoch + 1) % self.opts.save_period == 0:
                self.save_checkpoint()

    def train_one_epoch(self):
        """
        Train the model for one epoch.
        """
        train_loss = 0
        
        for train_batch in self.train_loader:
            # initialize gradient
            self.optimizer.zero_grad()

            # parse batch
            img, depth, camera_params, condition_img, pointcloud = train_batch
            cam_K = camera_params["K"]
            cam_R = camera_params["Rt"]

            # send data to device
            img = img.to(self.device).float()
            depth = depth.to(self.device)
            cam_K = cam_K.to(self.device).float()
            cam_R = cam_R.to(self.device).float()
            condition_img = condition_img.to(self.device).float()

            # parse point cloud data
            p = pointcloud[None].to(self.device)
            n = pointcloud["normals"].to(self.device)
            pointcloud = torch.cat([p, n], dim=1)

            # forward propagation
            out = self.model(depth, cam_K, cam_R, pointcloud, condition_img, img)

            loss = out["loss"]

            if self.opts.use_multi_gpu:
                loss = loss.mean()

            # back prop
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= self.opts.batch_size

        return train_loss

    def test_one_epoch(self):
        """
        Test the model every epoch.
        """
        with torch.no_grad():
            test_loss = 0
            gen_imgs = None

            for test_batch in self.test_loader:
                # parse batch
                img, depth, camera_params, condition_img, pointcloud = test_batch
                cam_K = camera_params["K"]
                cam_R = camera_params["Rt"]

                # send data to device
                img = img.to(self.device).float()
                depth = depth.to(self.device).float()
                cam_K = cam_K.to(self.device).float()
                cam_R = cam_R.to(self.device).float()
                condition_img = condition_img.to(self.device).float()

                # parse point cloud data
                p = pointcloud[None].to(self.device).float()
                n = pointcloud["normals"].to(self.device).float()
                pointcloud = torch.cat([p, n], dim=1)

                # forward propagation
                out = self.model(depth, cam_K, cam_R, pointcloud, condition_img, img)

                gen_imgs = out["img_pred"]
                loss = out["loss"]

                if self.opts.use_multi_gpu:
                    loss = loss.mean()

                test_loss += loss.item()
    
                assert gen_imgs is not None, "[!] Set of predicted images must not be empty."

                if self.opts.log_wandb:
                    # log RGB images
                    gen_imgs = gen_imgs.permute(0, 2, 3, 1).cpu().numpy()
                    img = img.permute(0, 2, 3, 1).cpu().numpy()
                    condition_img = condition_img.permute(0, 2, 3, 1).cpu().numpy()

                    # log depth maps
                    depth[torch.isinf(depth)] = 0
                    depth /= depth.max()
                    depth *= 255.0
                    depth = depth.type(torch.uint8)
                    depth = depth.permute(0, 2, 3, 1).cpu().numpy()

                    log_dict = {
                        "Images/GT": [wandb.Image(x) for x in img],
                        "Images/Predicted-Generated": [wandb.Image(x) for x in gen_imgs],
                        "Images/Depth": [wandb.Image(x) for x in depth],
                        "Images/Condition": [wandb.Image(x) for x in condition_img]
                    }

                    wandb.log(log_dict, step=self.epoch)

        test_loss /= self.opts.test_dataset_size

        return test_loss

    def inference(self, out_dir: str):
        print("[!] Begin inference..")

        if not os.path.exists(out_dir):
            print("[!] Directory %s does not exist." % out_dir)
            os.mkdir(out_dir)
            os.mkdir(os.path.join(out_dir, "real"))
            os.mkdir(os.path.join(out_dir, "fake"))
            print("[!] Created directory %s." % out_dir)   

        inf_total_loss = 0.0
        sample_idx = 0

        with torch.no_grad():
            for test_batch in self.test_loader:
                img, depth, camera_params, condition_img, pointcloud = test_batch
                cam_K = camera_params["K"]
                cam_R = camera_params["Rt"]

                # send data to device
                img = img.to(self.device).float()
                depth = depth.to(self.device).float()
                cam_K = cam_K.to(self.device).float()
                cam_R = cam_R.to(self.device).float()
                condition_img = condition_img.to(self.device).float()

                # parse point cloud data
                p = pointcloud[None].to(self.device).float()
                n = pointcloud["normals"].to(self.device).float()
                pointcloud = torch.cat([p, n], dim=1)

                # forward propagation
                out = self.model(depth, cam_K, cam_R, pointcloud, condition_img, img)

                gt_imgs = img
                gen_imgs = out["img_pred"]
                loss = out["loss"]

                # log images and loss
                for gt, fake in zip(gt_imgs, gen_imgs):
                    gt *= 255.0
                    fake *= 255.0
                    gt = gt.type(torch.uint8)
                    fake = fake.type(torch.uint8)
                    gt = gt.permute((1, 2, 0)).cpu().numpy()      # (C, H, W) -> (H, W, C)
                    fake = fake.permute((1, 2, 0)).cpu().numpy()  # (C, H, W) -> (H, W, C)

                    gt_img = Image.fromarray(gt)
                    fake_img = Image.fromarray(fake)

                    gt_img.save(os.path.join(out_dir, "real/{}.jpg".format(str(sample_idx).zfill(6))))
                    fake_img.save(os.path.join(out_dir, "fake/{}.jpg".format(str(sample_idx).zfill(6))))

                    sample_idx += 1

                inf_total_loss += loss.item()

            inf_total_loss /= (sample_idx)

        print("[!] Total inference loss computed over {} samples: {}".format(sample_idx, inf_total_loss))

    def configure_optimizer(self) -> torch.optim.Optimizer:
        optimizer = optim.Adam(self.model.parameters(), lr=self.opts.lr)
        return optimizer

    def configure_lr_scheduler(self) -> torch.optim.lr_scheduler:
        return None

    def configure_dataset(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        if self.opts.dataset_type == "shapenet":
            dataset = ShapeNetSingleClassDataset(
                "./data/shapenet/synthetic_cars_nospecular", img_size=128, num_pc_samples=2048
            )
            print("[!] Dataset used: ShapeNet")
        else:
            print(
                "[!] Please provide valid dataset type. Either 'shapenet' or 'pix3d' is supported by now"
            )
            exit()

        train_dataset, test_dataset = data.random_split(
            dataset,
            [len(dataset) - self.opts.test_dataset_size, self.opts.test_dataset_size],
            generator=torch.Generator().manual_seed(42),
        )

        return train_dataset, test_dataset

    def configure_dataloader(
        self,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_loader = data.DataLoader(
            self.train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=self.opts.num_workers,
            drop_last=True,
        )

        test_loader = data.DataLoader(
            self.test_dataset, 
            batch_size=self.opts.batch_size, 
            shuffle=False,
            num_workers=self.opts.num_workers // 2 if self.opts.num_workers > 1 else 1,
            drop_last=True,
        )

        return train_loader, test_loader

