"""
remove_anomalies.py - Script for removing anomalies in the dataset
"""

import os
from tqdm import tqdm
import shutil


def main():
    dataset_dir = "data/shapenet/synthetic_cars_nospecular"

    samples = os.listdir(dataset_dir)

    cnt = 0

    for sample in tqdm(samples):
        full_path = os.path.join(dataset_dir, sample)

        contents = os.listdir(full_path)

        if len(contents) != 5:
            print("[!] Removing {} located at {}".format(sample, full_path))
            shutil.rmtree(full_path)
            cnt += 1

    print("[!] Remove {} invalid samples in the dataset".format(cnt))


if __name__ == "__main__":
    main()
