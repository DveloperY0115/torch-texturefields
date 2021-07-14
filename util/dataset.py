"""
dataset.py - Set of classes and functions for Texture Fields dataset
"""

import os
import yaml
import random
import glob
import imageio
import numpy as np

import torch
from torch.utils import data
import torchvision.transforms as T

from .transforms import *


class ShapeNetSingleClassDataset(data.Dataset):
    def __init__(self, dataset_directory, img_size, num_pc_samples, num_neighbors=None):
        """
        Constructor of ShapeNetSingleClassDataset.

        Specify the directory where data is stored,
        create the abstract dataset for torch.nn.Module.

        Assumes directories of form:
        ------------------------------------------------
        - shapenet
	        - classA (e.g. car, chair, etc)
		    - sample1
			    - depth/depth_files (.exr)
			    - image/img_files (.png)
			    - input_image/img_files (.png)
			    - visualize/img_files (.png)
		    - sample2
		    - and so on
        ------------------------------------------------
        Then for the argument 'dataset_directory', the value 'shapenet/classA/' should be provided.

        Args:
        - dataset_directory (str or os.path): Directory to where data is located.
        - img_size (int): Size of image along one dimension. (e.g. for img_size = 128, images will be resized to 128x128)
        - num_pc_samples (int): Number of points to be selected during point cloud subsampling.
        - num_neighbors (int): Number of neighbors used during KNN computation.
        """
        super(ShapeNetSingleClassDataset, self).__init__()

        self.dataset_directory = dataset_directory

        self.samples = os.listdir(self.dataset_directory)

        # read metadata if exists
        metadata_file = os.path.join(dataset_directory, "metadata.yaml")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = yaml.load(f)
        else:
            metadata = {}

        # transforms
        self.transform_img = T.Compose([ResizeImage((img_size, img_size), order=0),])
        self.transform_img_conditional = T.Compose([ResizeImage((224, 224), order=0),])
        self.transform_depth = T.Compose(
            [ImageToDepthValue(), ResizeImage((img_size, img_size), order=0),]
        )
        transform_pcl = [SubsamplePointcloud(num_pc_samples)]
        if num_neighbors is not None:
            transform_pcl.append(ComputeKNNPointcloud(num_neighbors))
        self.transform_pcl = T.Compose(transform_pcl)

    def __len__(self):
        """
        __len__ method of torch.utils.data.Dataset.

        Returns:
        - num_data (int): Number of samples in the dataset
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        __getitem__ method of torch.utils.data.Dataset.

        Returns:
        - Data object (Exact form is TBD)
        """

        # identify sample directories
        sample_dir = os.path.join(self.dataset_directory, self.samples[idx])

        # load image
        img, depth_map, camera_params = self.load_img_and_depth(sample_dir, [random.randint(0, 9)])

        # load conditional image
        condition_img = self.load_condition_img(sample_dir)

        # load point cloud
        pointcloud = self.load_pointcloud(sample_dir)

        return (img, depth_map, camera_params, condition_img, pointcloud)

    def load_img_and_depth(self, sample_directory, indices, with_camera=False):
        """
        Load images and associated depth maps rendered from the same viewpoint.
        Also, optionally load camera parameters for the viewpoint.

        Args:
        - sample_directory (str or os.path): Root directory of the sample.
        - indices (Iterable): List (or tuple) of indices specifying which pairs of image-depth to be loaded.
        - with_camera (bool): Determines whether to load camera parameters associated with image (depth) together as well.
                Set to false by default.

        Returns:
        - images (torch.Tensor): Tensor of shape (len(indices), 3, H, W). Collection of images.
        - depth_maps (torch.Tensor): Tensor of shape (len(indices), 1, H, W). Collection of depth maps associated to 'images'.
        - [Optional] camera_params (dict): Dictionary of lists of torch.Tensor instances containing camera parameters.
                - Rt (list): List of torch.Tensor instances representing camera world transform matrices.
                - K (list): List of torch.Tensor instances representing camera projection matrices.
        """
        # identify directories for each type of data
        image_dir = os.path.join(sample_directory, "image")
        depth_dir = os.path.join(sample_directory, "depth")

        assert os.path.exists(image_dir)
        assert os.path.exists(depth_dir)

        image_files = glob.glob(os.path.join(image_dir, "*.{}".format("png")))
        depth_files = glob.glob(os.path.join(depth_dir, "*.{}".format("exr")))

        assert len(image_files) == len(
            depth_files
        ), "[!] Number of images and depth maps should match."

        image_files.sort()
        depth_files.sort()

        images = []
        depth_maps = []

        for idx in indices:
            filename_image = image_files[idx]
            filename_depth = depth_files[idx]

            # retrieve image and depth map
            image = load_img(filename_image, self.transform_img)
            depth = load_depth_map(filename_depth, self.transform_depth)

            images.append(image.unsqueeze(0))
            depth_maps.append(depth.unsqueeze(0))

        images = torch.cat(images, dim=0)
        depth_maps = torch.cat(depth_maps, dim=0)

        camera_params = {"Rt": [], "K": []}

        if with_camera:
            camera_file = os.path.join(depth_dir, "cameras.npz")
            camera_dict = np.load(camera_file)

            for idx in indices:
                Rt = camera_dict["world_mat_%d" % idx].astype(np.float32)
                K = camera_dict["camera_mat_%d" % idx].astype(np.float32)
                camera_params["Rt"].unsqueeze(0).append(Rt)
                camera_params["K"].unsqueeze(0).append(K)

            camera_params["Rt"] = torch.cat(camera_params["Rt"], dim=0).squeeze()
            camera_params["K"] = torch.cat(camera_params["K"], dim=0).squeeze()

        return images.squeeze(), depth_maps.squeeze(), camera_params

    def load_condition_img(self, sample_directory, use_random=True):
        """
        Load images used in conditional setting (please refer to the paper for detail).
        These images are then used as appearance reference for Texture Field.

        Args:
        - sample_directory (str or os.path): Root directory of the sample.
        - indices (Iterable): List (or tuple) of indices specifying which pairs of image-depth to be loaded.
        - use_random (bool): Determines whether to sample condition image randomly or not.
            If set to false, then the first image in the directory is used.

        Returns:
        - images (torch.Tensor): Tensor of shape (len(indices), 3, 224, 224).
        """
        # identify directories for each type of data
        image_dir = os.path.join(sample_directory, "input_image")

        assert os.path.exists(image_dir)

        image_files = glob.glob(os.path.join(image_dir, "*.{}".format("jpg")))
        image_files.sort()

        if use_random:
            idx = random.randint(0, len(image_files) - 1)
        else:
            idx = 0

        filename_image = image_files[idx]

        # retrieve image and depth map
        image = load_img(filename_image, self.transform_img)

        return image

    def load_pointcloud(self, sample_directory, with_transforms=False):
        """
        Load a point cloud.

        Args:
        - sample_directory (str or os.path): Root directory of the sample.
        - with_transforms (bool): Determines whether to load additional transform information related to a point cloud as well.
            Set to false by default.

        Returns:
        - pointcloud (Dict): Dictionary of torch.Tensors containing various geometric features of a point cloud.
            For each key:
            - "None" -> torch.Tensor containing 3-coordinates of points of a point cloud.
            - "normal" -> torch.Tensor containing 3-vectors of normal vectors at each point.
            - [Optional] "loc" -> torch.Tensor containing (?) 
            - [Optional] "scale" -> torch.Tensor containing (?) 
        """
        filename = os.path.join(sample_directory, "pointcloud.npz")

        pointcloud_dict = np.load(filename)

        points = pointcloud_dict["points"].astype(np.float32)
        normals = pointcloud_dict["normals"].astype(np.float32)

        pointcloud = {None: points.T, "normals": normals.T}

        if with_transforms:
            pointcloud["loc"] = pointcloud_dict["loc"].astype(np.float32)
            pointcloud["scale"] = pointcloud_dict["scale"].astype(np.float32)

        pointcloud = self.transform_pcl(pointcloud)

        for key, array in pointcloud.items():
            pointcloud[key] = torch.tensor(array)

        return pointcloud


def load_depth_map(filename, transform=None):
    """
    Load a single depth map and the camera parameters associated to it.

    Args:
    - filename (str): Name of a image file to be loaded.
    - transform (torchvision.transform): Transforms to be applied.
    
    Returns:
    - depth (torch.Tensor): Tensor of shape (1, H, W).
    """
    depth = imageio.imread(filename)
    depth = np.asarray(depth)

    if transform is not None:
        depth = transform(depth)

    depth = depth.transpose(2, 0, 1)

    return torch.tensor(depth)


def load_img(filename, transform=None):
    """
    Load a single image.

    Args:
    - filename (str): Name of a image file to be loaded.
    - transform (torchvision.transform): Transforms to be applied.

    Returns:
    - image (torch.Tensor): Tensor of shape (C, H, W).
    """
    image = imageio.imread(filename)
    image = np.asarray(image)

    if len(image.shape) == 2:
        image = image.reshape(image.shape[0], image.shape[1], 1)
        image = np.concatenate([image, image, image], axis=2)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    # normaliize RGB to [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
    else:
        image = image.astype(np.float32)

    # apply designated transform if exists
    if transform is not None:
        image = transform(image)

    image = image.transpose(2, 0, 1)

    return torch.tensor(image)
