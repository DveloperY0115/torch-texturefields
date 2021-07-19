"""
train.py - Training routine for Texture Fields architecture.
"""

import argparse
import os
import glob
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
parser.add_argument(
    "--experiment_setting",
    type=str,
    default="vae",
    help="Toggle experiment settings. Can be one of 'conditional', 'vae', and 'gan'",
)
parser.add_argument("--no-cuda", type=bool, default=False, help="CUDA is not used when True")
parser.add_argument(
    "--device_id", type=int, default=1, help="CUDA device ID if multiple devices available"
)
parser.add_argument(
    "--use_multi_gpu", type=bool, default=False, help="Use multiple GPUs if available"
)
parser.add_argument("--batch_size", type=int, default=64, help="Size of a batch")
parser.add_argument("--test_set_size", type=int, default=10, help="Cardinality of test set")
parser.add_argument("--num_epoch", type=int, default=10000, help="Number of epochs for training")
parser.add_argument("--num_iter", type=int, default=100, help="Number of iteration in one epoch")
parser.add_argument(
    "--num_workers", type=int, default=10, help="Number of workers for data loading"
)
parser.add_argument("--lr", type=float, default=0.0001, help="Initial value of learning rate")
parser.add_argument("--beta1", type=float, default=0.9, help="Beta 1 of Adam")
parser.add_argument("--beta2", type=float, default=0.999, help="Beta 2 of Adam")
parser.add_argument(
    "--step_size", type=int, default=200, help="Step size for learning rate scheduling"
)
parser.add_argument("--gamma", type=float, default=0.85, help="Gamma for learning rate scheduling")
parser.add_argument(
    "--out_dir", type=str, default="out", help="Directory where the outputs will be saved"
)
parser.add_argument(
    "--start_from_checkpoint",
    type=bool,
    default=True,
    help="Start from existing checkpoint if possible",
)
parser.add_argument(
    "--save_period", type=int, default=100, help="Number of epochs between checkpoints"
)
args = parser.parse_args()


def main():

    # configure device (cpu or gpu)
    device = configure_device(args.device_id)

    # directories
    _, _, log_dir, checkpoint_dir = configure_output_directories(
        args.out_dir, args.experiment_setting
    )

    # initialize writer for Tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # define model, optimizer, and LR scheduler
    if args.experiment_setting in ("conditional", "vae", "gan"):
        model = TextureFieldsCls(args.experiment_setting, device).to(device)
    else:
        print(
            "[!] Please provide valid argument. Experiment settings can be one of 'conditional', 'vae', and 'gan'"
        )
        return -1
    print("[!] Using experiment setting: {}".format(args.experiment_setting))

    # toggle data parallelism
    if args.use_multi_gpu:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    epoch_0 = 0

    # load checkpoint if exists
    if args.start_from_checkpoint:
        checkpoint = load_checkpoint(checkpoint_dir)

        # parse checkpoint and initialize state
        if checkpoint is not None:
            epoch_0 = checkpoint["epoch"]

            # set model, optimizer, scheduler parameters
            if args.use_multi_gpu:
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            model.train()

    # define dataset
    dataset = ShapeNetSingleClassDataset(
        "data/shapenet/synthetic_cars_nospecular", img_size=128, num_pc_samples=2048
    )

    # split dataset into train / test
    train_data, test_data = data.random_split(
        dataset,
        [len(dataset) - args.test_set_size, args.test_set_size],
        generator=torch.Generator().manual_seed(42),
    )

    # configure data loaders
    train_loader = data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = data.DataLoader(test_data, batch_size=args.test_set_size, shuffle=False)

    # run training
    for epoch in tqdm(range(epoch_0, args.num_epoch)):

        train_loss = train_one_epoch(model, optimizer, scheduler, device, train_loader)

        with torch.no_grad():
            test_loss = test_one_epoch(model, device, test_loader, writer, epoch)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

        if (epoch + 1) % args.save_period == 0:
            save_checkpoint(epoch, train_loss, model, optimizer, scheduler, checkpoint_dir)

    writer.close()


def configure_device(device_id):
    """
    Configure which device to run training on.
    Inform users various tips based on the status of their machine.

    Args:
    - device_id (int): Index of device to be used.

    Returns:
    - device (torch.device): Context-manager in Pytorch which designates the selected device.
    """

    device = torch.device(
        "cuda:{}".format(device_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print("[!] Using {} as default device".format(device))

    if torch.cuda.is_available() and args.no_cuda:
        print("[!] Your system is capable of GPU computing but is set not to use it.")
        print("[!] It's highly recommended to use GPUs for training!")

    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("[!] Multiple GPUs available.")

        if not args.use_multi_gpu:
            print("[!] But it's set to start with single GPU")
            print("[!] It's highly recommended to use multiple GPUs if possible!")

    return device


def configure_output_directories(out_root, experiment_setting):
    """
    Setup directories where outputs (log, checkpoint, etc) will be saved.
    Configure directory differently for different experiment settings.

    Args:
    - out_root (str or os.path): Root of the output directory.
    - experiment_setting (str): Indicator for experiment setting. Can be either 'conditional' or 'generative'.

    Returns:
    - Tuple of strings each representing directories created inside this function.
    """

    if not os.path.exists(out_root):
        os.mkdir(out_root)

    # experiment specific directory
    experiment_dir = os.path.join(out_root, experiment_setting)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    # log output directory
    log_dir = os.path.join(experiment_dir, "runs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # checkpoint directory
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")

    return out_root, experiment_dir, log_dir, checkpoint_dir


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
        out = model(depth, cam_K, cam_R, pointcloud, condition_img, img)

        loss = out["loss"]

        # back propagation
        loss.backward()
        optimizer.step()

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

        # parse point cloud data
        p = pointcloud[None].to(device)
        n = pointcloud["normals"].to(device)
        pointcloud = torch.cat([p, n], dim=1)

        # forward propagation
        out = model(depth, cam_K, cam_R, pointcloud, condition_img, img)

        gen_imgs = out["img_pred"]

        loss = out["loss"]

        test_loss += loss.item()

    assert gen_imgs is not None, "Set of predicted images should be empty."

    writer.add_images("Generated Image/test", gen_imgs, epoch)
    writer.add_images("GT Image/test", img, epoch)
    writer.add_images("GT Depth/test", depth, epoch)
    writer.add_images("GT Condition/test", condition_img, epoch)

    avg_loss = test_loss / args.test_set_size

    return avg_loss


def save_checkpoint(epoch, loss, model, optimizer, scheduler, save_dir):
    """
    Save checkpoint.

    Args:
    - epoch (int): Index of epoch.
    - loss (float): Current value of loss
    - model (torch.nn.Module): Neural network to be trained.
    - optimizer (torch.optim.Optimizer): Optimizer used for training.
    - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    - save_dir (str or os.path): Directory where checkpoint files will be saved.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, "checkpoint-epoch-{}.tar".format(epoch + 1))

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.module.state_dict()
            if args.use_multi_gpu
            else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        save_path,
    )
    print("[!] Saved model at: {}".format(save_path))


def load_checkpoint(checkpoint_root, load_latest=True):
    """
    Load existing checkpoint to resume training.

    Args:
    - checkpoint_root (str or os.path): Path to directory containing checkpoint files.
    - load_latest (bool): Determines whether to load the latest checkpoint or not.

    NOTE: This function currently supports load latest-only.

    Returns:
    - checkpoint (Dict): Dictionary containing parameters of model, optimizer, scheduler and other values for tracking progress.
        - epoch (int): Index of epoch.
        - loss (float): Current value of loss.
        - model_state_dict (torch.nn.Module.state_dict): Dictionary containing current status of neural network.
        - optimizer_state_dict (torch.optim.Optimizer.state_dict): Dictionary containing current status of optimizer.
        - scheduler_state_dict (torch.optim.lr_scheduler.state_dict): Dictionary containing current status of scheduler.
    - [Exceptionally] None: If there's no checkpoint directory or files.
    """

    if not os.path.exists(checkpoint_root):
        print("[!] No checkpoint loaded")
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_root, "*.{}".format("tar")))
    checkpoint_files.sort()

    if len(checkpoint_files) == 0:
        print("[!] No checkpoint loaded")
        return None

    latest_checkpoint = checkpoint_files[-1]

    print("[!] Loading the latest checkpoint {}".format(latest_checkpoint))
    return torch.load(latest_checkpoint)


if __name__ == "__main__":
    main()
