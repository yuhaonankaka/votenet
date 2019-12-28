# based off of https://github.com/angeladai/3DMV

import os
import math
import imageio
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

def read_lines_from_file(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    return lines


# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0,0] *= float(resize_width)/float(intrinsic_image_dim[0])
    intrinsic[1,1] *= float(image_dim[1])/float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0,2] *= float(image_dim[0]-1)/float(intrinsic_image_dim[0]-1)
    intrinsic[1,2] *= float(image_dim[1]-1)/float(intrinsic_image_dim[1]-1)
    return intrinsic


def load_pose(filename):
    pose = torch.Tensor(4, 4)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    return image


def load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims, normalize):
    color_image = imageio.imread(color_file)
    depth_image = imageio.imread(depth_file)
    pose = load_pose(pose_file)
    # preprocess
    depth_image = resize_crop_image(depth_image, depth_image_dims)
    color_image = resize_crop_image(color_image, color_image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0
    color_image =  np.transpose(color_image, [2, 0, 1])  # move feature to front
    color_image = normalize(torch.Tensor(color_image.astype(np.float32) / 255.0))
    return depth_image, color_image, pose

def load_frames_multi(data_path, batch_scan_names, num_images, depth_images, color_images, poses, color_mean, color_std, choice='even'):

    depth_files = []
    color_files = []
    pose_files = []

    for scan_name in batch_scan_names:
        depth_path = os.path.join(data_path, scan_name, "depth")
        color_path = os.path.join(data_path, scan_name, "color")
        pose_path = os.path.join(data_path, scan_name, "pose")

        img_list = np.array(sorted(os.listdir(os.path.join(color_path))))
        depth_list = np.array(sorted(os.listdir(os.path.join(depth_path))))
        pose_list = np.array(sorted(os.listdir(os.path.join(pose_path))))
        assert len(img_list) == len(depth_list) == len(pose_list), "Data in %r have inconsistent amount of files" % data_path

        indices = np.array([])
        if choice == 'random':
            # Choose <num_images> frames randomly
            if len(img_list) < num_images:
                remain = num_images
                while remain > len(img_list):
                    indices = np.concatenate((indices, np.arange(len(img_list), dtype='int32')))
                    remain -= len(img_list)
                indices = np.concatenate((indices.astype('int32'), np.arange(remain, dtype='int32')))
            else:
                indices = np.sort(np.random.choice(len(img_list), num_images, replace=False)).astype('int32')
        elif choice == 'even':
            # Choose <num_images> frames according to the total number of frames. For example:
            # if `total_num_frames=20`, and `num_images=6`--> `interval=round(20/6)=3`
            # then the indices of the chosen frames are: [0,3,6,9,12,15]
            if len(img_list) < num_images:
                while len(indices) < num_images:
                    indices = np.append(indices, np.arange(len(img_list)))
                indices = indices[:num_images].astype('int32')
            else:
                interval = round(len(img_list) / num_images)
                if interval * (num_images - 1) > len(img_list) - 1:  # just in case, not really necessary
                    interval -= 1
                indices = np.arange(0, len(img_list), interval)
                indices = indices[:num_images]
                indices = indices.astype('int32')
        else:
            Exception("choice='{}' is not valid, please choose from ['random', 'even']".format(choice))


        for idx in indices:
            color_files.append(os.path.join(data_path, scan_name, 'color', img_list[idx]))
            depth_files.append(os.path.join(data_path, scan_name, 'depth', depth_list[idx]))
            pose_files.append(os.path.join(data_path, scan_name, 'pose', pose_list[idx]))

        # color_files.append(img_list[indices])
        # depth_files.append(depth_list[indices])
        # pose_files.append(pose_list[indices])

    depth_image_dims = [depth_images.shape[2], depth_images.shape[1]]
    color_image_dims = [color_images.shape[3], color_images.shape[2]]
    normalize = transforms.Normalize(mean=color_mean, std=color_std)

    # load data
    for k in range(len(batch_scan_names) * num_images):
        depth_image, color_image, pose = load_depth_label_pose(depth_files[k], color_files[k], pose_files[k],
                                                               depth_image_dims, color_image_dims, normalize)
        color_images[k] = color_image
        depth_images[k] = torch.from_numpy(depth_image)
        poses[k] = pose




# def load_frames_multi(data_path, frame_indices, depth_images, color_images, poses, color_mean, color_std):
#     # construct files
#     num_images = frame_indices.shape[1] - 2
#     scan_names = ['scene' + str(scene_id).zfill(4) + '_' + str(scan_id).zfill(2) for scene_id, scan_id in frame_indices[:,:2].numpy()]
#     scan_names = np.repeat(scan_names, num_images)
#     frame_ids = frame_indices[:, 2:].contiguous().view(-1).numpy()
#     depth_files = [os.path.join(data_path, scan_name, 'depth', str(frame_id) + '.png') for scan_name, frame_id in zip(scan_names, frame_ids)]
#     color_files = [os.path.join(data_path, scan_name, 'color', str(frame_id) + '.jpg') for scan_name, frame_id in zip(scan_names, frame_ids)]
#     pose_files = [os.path.join(data_path, scan_name, 'pose', str(frame_id) + '.txt') for scan_name, frame_id in zip(scan_names, frame_ids)]
#
#     batch_size = frame_indices.size(0) * num_images
#     depth_image_dims = [depth_images.shape[2], depth_images.shape[1]]
#     color_image_dims = [color_images.shape[3], color_images.shape[2]]
#     normalize = transforms.Normalize(mean=color_mean, std=color_std)
#     # load data
#     for k in range(batch_size):
#         depth_image, color_image, pose = load_depth_label_pose(depth_files[k], color_files[k], pose_files[k], depth_image_dims, color_image_dims, normalize)
#         color_images[k] = color_image
#         depth_images[k] = torch.from_numpy(depth_image)
#         poses[k] = pose