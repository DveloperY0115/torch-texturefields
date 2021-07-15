"""
train.py - Training routine for Texture Fields architecture.
"""

import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from util.dataset import ShapeNetSingleClassDataset
from model.texture_field import TextureFieldsCls

parser = argparse.ArgumentParser(description="Training routine Texture Field")
parser.add_argument("--no-cuda", type=bool, default=False, help="CUDA is not used when True")
parser.add_argument("--batch_size", type=int, default=2, help="Size of a batch")
parser.add_argument("--test_set_size", type=int, default=10, help="Cardinality of test set")
parser.add_argument("--num_epoch", type=int, default=10000, help="Number of epochs for training")
parser.add_argument("--num_iter", type=int, default=1000, help="Number of iteration in one epoch")
parser.add_argument("--lr", type=float, default=0.001, help="Initial value of learning rate")
parser.add_argument("--beta1", type=float, default=0.9, help="Beta 1 of Adam")
parser.add_argument("--beta2", type=float, default=0.999, help="Beta 2 of Adam")
parser.add_argument(
    "--step_size", type=int, default=100, help="Step size for learning rate scheduling"
)
parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for learning rate scheduling")
parser.add_argument(
    "--out_dir", type=str, default="out", help="Directory where the outputs will be saved"
)
parser.add_argument(
    "--save_period", type=int, default=100, help="Number of epochs between checkpoints"
)
args = parser.parse_args()


def main():

    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("[!] Multiple GPU available, but not yet supported")

    # create output directory
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    log_dir = os.path.join(args.out_dir, "runs")

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # initialize writer for Tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # define model, optimizer, and LR scheduler
    model = TextureFieldsCls().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # define dataset, loaders
    dataset = ShapeNetSingleClassDataset(
        "data/shapenet/synthetic_cars_nospecular", img_size=128, num_pc_samples=2048
    )

    train_data, test_data = data.random_split(
        dataset,
        [len(dataset) - args.test_set_size, args.test_set_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size)
    test_loader = data.DataLoader(test_data, batch_size=args.test_set_size)

    # run training
    for epoch in tqdm(range(args.num_epoch), leave=False):

        train_loss = train_one_epoch(model, optimizer, scheduler, device, train_loader)

        with torch.no_grad():
            test_loss = test_one_epoch(model, device, test_loader)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

        if (epoch + 1) % args.save_period == 0:
            save_checkpoint(epoch, train_loss, model, optimizer, scheduler)

    writer.close()


def train_one_epoch(model, optimizer, scheduler, device, loader):
    """
    Train the model for one epoch.

    Args:
    - model (torch.nn.Module): Neural network to be trained.
    - optimizer (torch.optim.Optimizer): Optimizer used for training.
    - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    - device (torch.device): Object representing currently used device.
    - loader (torch.utils.data.Dataloader): Dataloader for training data.
    - writer (torch.utils.tensorboard.SummaryWriter): Writer for Tensorboard recording.

    Returns:
    - avg_loss (float): Average loss calculated during one epoch.
    """

    train_iter = iter(loader)

    train_loss = 0

    for _ in tqdm(range(args.num_iter)):

        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            train_batch = next(train_iter)

        # initialize gradient
        optimizer.zero_grad()

        # parse batch
        img, depth, camera_params, condition_img, pointcloud = train_batch
        cam_K = camera_params["K"]
        cam_R = camera_params["Rt"]

        # send data to device
        img = img.to(device)
        depth = depth.to(device)
        cam_K = cam_K.to(device)
        cam_R = cam_R.to(device)
        condition_img = condition_img.to(device)

        # parse point cloud data
        p = pointcloud[None].to(device)
        n = pointcloud["normals"].to(device)
        pointcloud = torch.cat([p, n], dim=1)

        # forward propagation
        img_pred = model(depth, cam_K, cam_R, pointcloud, condition_img)

        # calculate loss
        loss = nn.L1Loss()(img_pred, img)

        # back propagation
        loss.backward()
        optimizer.step()

        # update scheduler
        scheduler.step()

        train_loss += loss.item()

    avg_loss = train_loss / args.batch_size
    return avg_loss


def test_one_epoch(model, device, loader, writer, epoch):
    """
    Test the model every epoch.

    Args:
    - model (torch.nn.Module): Neural network to be trained.
    - device (torch.device): Object representing currently used device.
    - loader (torch.utils.data.Dataloader): Dataloader for training data.
    - writer (torch.utils.tensorboard.SummaryWriter): Writer for Tensorboard recording.
    - epoch (int): Index for current epoch.

    Returns:
    - avg_loss (float): Average test loss computed on the test set.
    """

    test_iter = iter(loader)

    test_loss = 0

    gen_imgs = None

    while True:
        try:
            test_batch = next(test_iter)
        except StopIteration:
            break

        # parse batch
        img, depth, camera_params, condition_img, pointcloud = test_batch
        cam_K = camera_params["K"]
        cam_R = camera_params["Rt"]

        # send data to device
        img = img.to(device)
        depth = depth.to(device)
        cam_K = cam_K.to(device)
        cam_R = cam_R.to(device)
        condition_img = condition_img.to(device)

        # forward propagation
        gen_imgs = model(depth, cam_K, cam_R, pointcloud, condition_img)

        # calculate loss
        loss = nn.L1Loss()(gen_imgs, img)

        test_loss += loss.item()

    assert gen_imgs is not None, "Set of predicted images should be empty."

    writer.add_images("Generated Image/test", gen_imgs, epoch)
    writer.add_images("GT Image/test", img, epoch)
    writer.add_images("GT Depth/test", depth, epoch)
    writer.add_images("GT Condition/test", condition_img, epoch)

    avg_loss = test_loss / args.test_set_size

    return avg_loss


def save_checkpoint(epoch, loss, model, optimizer, scheduler):
    """
    Save checkpoint.

    Args:
    - epoch (int): Index of epoch.
    - loss (float): Current value of loss
    - model (torch.nn.Module): Neural network to be trained.
    - optimizer (torch.optim.Optimizer): Optimizer used for training.
    - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    """
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    save_root = os.path.join(args.out_dir, "checkpoint")

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    save_path = os.path.join(save_root, "checkpoint-epoch-{}.pt".format(epoch + 1))

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        save_path,
    )
    print("[!] Saved model at: {}".format(save_path))


if __name__ == "__main__":
    main()
