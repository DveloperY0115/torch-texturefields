"""
train.py
"""

import os
import sys
import argparse

sys.path.append(".")
sys.path.append("..")

from training.texturefield_conditional_trainer import TextureFieldsConditionalTrainer

parser = argparse.ArgumentParser()

# ----- CUDA -----
parser.add_argument("--no-cuda", type=bool, default=False, help="CUDA is not used when True")
parser.add_argument(
    "--device_id", type=int, default=2, help="CUDA device ID if multiple devices available"
)
parser.add_argument(
    "--use_multi_gpu", type=bool, default=False, help="Use multiple GPUs if available"
)

parser.add_argument("--experiment_setting", type=str, default="conditional")

# ----- dataset settings -----
parser.add_argument(
    "--dataset_type",
    type=str,
    default="shapenet",
    help="Name of the dataset to be used. Can be one of 'shapenet' or 'pix3d'",
)
parser.add_argument("--batch_size", type=int, default=64, help="Size of a batch")
parser.add_argument("--test_dataset_size", type=int, default=16, help="Cardinality of test set")

# ----- training parameters -----
parser.add_argument("--num_epoch", type=int, default=10000, help="Number of epochs for training")
parser.add_argument("--num_iter", type=int, default=100, help="Number of iteration in one epoch")
parser.add_argument(
    "--num_workers", type=int, default=10, help="Number of workers for data loading"
)
parser.add_argument("--lr", type=float, default=0.0001, help="Initial value of learning rate")
parser.add_argument("--beta1", type=float, default=0.9, help="Beta 1 of Adam")
parser.add_argument("--beta2", type=float, default=0.999, help="Beta 2 of Adam")

# ----- I/O -----
parser.add_argument(
    "--out_dir", type=str, default="out", help="Directory where the outputs will be saved"
)
parser.add_argument(
    "--log_wandb", type=bool, default=True, help="Determine whether to log the run using W&B"
)
parser.add_argument(
    "--save_period", type=int, default=100, help="Number of epochs between checkpoints"
)

args = parser.parse_args()


def main():
    trainer = TextureFieldsConditionalTrainer(opts=args)
    trainer.train()


if __name__ == "__main__":
    main()
