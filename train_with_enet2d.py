# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from models.nlp import LanguageNet, getweights, get_word2idx
from utils.projection import ProjectionHelper
from models.enet import create_enet_for_3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
RAW_DATA_DIR = '/mnt/canis/Datasets/ScanNet/public/v2/scans/'
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pointnet2.pytorch_utils import BNMomentumScheduler
from utils import image_util
from utils.tf_visualizer import Visualizer as TfVisualizer
from models.ap_helper import APCalculator, parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log_Jan25', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps',
                    help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160',
                    help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
# =================
# 3DMV
# =================
parser.add_argument('--data_path_2d', default='/home/davech2y/frames_square', help='path to 2d train data')
parser.add_argument('--num_classes', default=18, help='#classes')
parser.add_argument('--num_nearest_images', type=int, default=10, help='#images')
parser.add_argument('--model2d_type', default='scannet', help='which enet (scannet)')
parser.add_argument('--model2d_path', default='./2d_scannet.pth', help='path to enet model')
parser.add_argument('--use_proxy_loss', dest='use_proxy_loss', action='store_true')
# 2d/3d
parser.add_argument('--depth_min', type=float, default=0.4, help='min depth (in meters)')
parser.add_argument('--depth_max', type=float, default=4.0, help='max depth (in meters)')
# scannet intrinsic params
parser.add_argument('--intrinsic_image_width', type=int, default=640, help='2d image width')
parser.add_argument('--intrinsic_image_height', type=int, default=480, help='2d image height')

FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG

#                    classes, color mean/std
# ENET_TYPES = {'scannet': (18, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}
ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}

input_image_dims = [320, 240]
proj_image_dims = [40, 30]  # feature dimension of ENet
# proj_image_dims = [41, 32]  # feature dimension of ENet
# proj_image_dims = [320, 240]  # feature dimension of ENet
color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.28623, 0.291129]



BATCH_SIZE = FLAGS.batch_size
NUM_IMAGES = FLAGS.num_nearest_images
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert (len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)' % (LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s' % (LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Create Dataset and Dataloader
if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd.sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
    from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig

    DATASET_CONFIG = SunrgbdDatasetConfig()
    TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=NUM_POINT,
                                                 augment=True,
                                                 use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
                                                 use_v1=(not FLAGS.use_sunrgbd_v2))
    TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=NUM_POINT,
                                                augment=False,
                                                use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
                                                use_v1=(not FLAGS.use_sunrgbd_v2))
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet.scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from scannet.model_util_scannet import ScannetDatasetConfig

    DATASET_CONFIG = ScannetDatasetConfig()
    TRAIN_DATASET = ScannetDetectionDataset('train', num_points=NUM_POINT,
                                            augment=True,
                                            use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
                                           augment=False,
                                           use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
else:
    print('Unknown dataset %s. Exiting...' % (FLAGS.dataset))
    exit(-1)
print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model)  # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1

if FLAGS.model == 'boxnet':
    Detector = MODEL.BoxNet
else:
    Detector = MODEL.VoteNet

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_images=FLAGS.num_nearest_images,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel + 128,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)

if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
# if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
#     checkpoint = torch.load(CHECKPOINT_PATH)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch']
#     log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)


# create model
# model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES['scannet'],
#                                                                           FLAGS.model2d_path, DATASET_CONFIG.num_class)
model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES['scannet'],
                                                                          FLAGS.model2d_path, DATASET_CONFIG.num_class)

# move to gpu
model2d_fixed = model2d_fixed.cuda()
model2d_fixed.eval()
model2d_trainable = model2d_trainable.cuda()
model2d_classifier = model2d_classifier.cuda()


def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS, 'train')
TEST_VISUALIZER = TfVisualizer(FLAGS, 'test')

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
               'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
               'per_class_proposal': True, 'conf_thresh': 0.05,
               'dataset_config': DATASET_CONFIG}


def get_batch_intrinsics(batch_scan_names):
    """ Read intrinsics from txt file
    :param scan_name:
    :return: numpy array of shape: [batch_size, 4, 4]
    """
    batch_intrinsics = []
    for scan_name in batch_scan_names:
        intrinsic_str = image_util.read_lines_from_file(FLAGS.data_path_2d + '/' + scan_name + '/intrinsic_depth.txt')
        fx = float(intrinsic_str[0].split()[0])
        fy = float(intrinsic_str[1].split()[1])
        mx = float(intrinsic_str[0].split()[2])
        my = float(intrinsic_str[1].split()[2])

        intrinsic = image_util.make_intrinsic(fx, fy, mx, my)
        intrinsic = image_util.adjust_intrinsic(intrinsic, [FLAGS.intrinsic_image_width, FLAGS.intrinsic_image_height],
                                                proj_image_dims)
        batch_intrinsics.append(np.array(intrinsic))

    return np.array(batch_intrinsics)

def get_random_frames(data_path_2d, scan_name):
    img_path = os.path.join(FLAGS.data_path_2d, scan_name, "color")
    img_list = list(sorted(os.listdir(os.path.join(img_path))))
    indices = np.random.choice(len(img_list), FLAGS.num_nearest_images, replace=False)
    chosen_imgs = img_list[indices]
    return chosen_imgs



# ------------------------------------------------------------------------- GLOBAL CONFIG END


def project_2d_features(batch_data_label):
    """
        Retrieve certain amount of images(NUM_IMAGES) corresponding to the scene and find the correspondence mapping
        between 3D and 2D points.

        Parameters
        ----------
        batch_data_label: dict
            got from dataloader.


        Returns:
            proj_ind_3d: indexes of 3D points, index 0 is the number of valid points
            proj_ind_2d: indexes of 2D points, correspondent to indexes in proj_ind_3d. index 0 is number of valid points.
            imageft: 2D feature maps extracted from given 2D CNN
    """

    batch_scan_names = batch_data_label["scan_name"]

    # Get camera intrinsics for each scene
    batch_intrinsics = get_batch_intrinsics(batch_scan_names)

    # Get 2d images and it's feature
    depth_images = torch.cuda.FloatTensor(BATCH_SIZE * NUM_IMAGES, proj_image_dims[1], proj_image_dims[0])
    color_images = torch.cuda.FloatTensor(BATCH_SIZE * NUM_IMAGES, 3, input_image_dims[1], input_image_dims[0])
    camera_poses = torch.cuda.FloatTensor(BATCH_SIZE * NUM_IMAGES, 4, 4)
    label_images = torch.cuda.LongTensor(BATCH_SIZE * NUM_IMAGES, proj_image_dims[1],
                                         proj_image_dims[0])  # for proxy loss

    image_util.load_frames_multi(FLAGS.data_path_2d, batch_scan_names, FLAGS.num_nearest_images,
                                 depth_images, color_images, camera_poses, color_mean, color_std, choice='even')

    # Convert aligned point cloud back to unaligned, so that we can do back-projection
    # using camera intrinsics & extrinsics
    batch_pcl_aligned = batch_data_label['point_clouds']
    batch_pcl_unaligned = []
    batch_scan_names = batch_data_label['scan_name']
    # find the align matrix according to scan_name
    batch_align_matrix = np.array([])
    for scan_name, pcl_aligned in zip(batch_scan_names, batch_pcl_aligned):
        # Load alignments
        lines = open(RAW_DATA_DIR + scan_name + "/" + scan_name + ".txt")
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                                     for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        # inv_axis_align_matrix_T = torch.inverse(torch.FloatTensor(axis_align_matrix.T))
        batch_align_matrix = np.append(batch_align_matrix, axis_align_matrix)
        inv_axis_align_matrix_T = np.linalg.inv(axis_align_matrix.T)

        # Numpy version:
        # Unalign the Point Cloud (See load_scannet_data.py as reference)
        pts = np.ones((pcl_aligned.size()[0], 4))
        pts[:, 0:3] = pcl_aligned[:, 0:3].cpu().numpy()
        pcl = np.dot(pts, inv_axis_align_matrix_T)
        batch_pcl_unaligned.append(torch.from_numpy(pcl).float())

        # # TEST
        # from test_helper import test_unalign_pcl
        # test_unalign_pcl(pcl_aligned.cpu().numpy(), pcl, scan_name, axis_align_matrix)
        # # END of TEST

        # Torch version:
        # Unalign the Point Cloud (See load_scannet_data.py as reference)
        # pcl = torch.ones(pcl_aligned.size()[0], 4)
        # pcl[:, 0:3] = pcl_aligned[:, 0:3]
        # pcl = torch.mm(pcl, inv_axis_align_matrix_T)
        # batch_pcl_unaligned.append(pcl)

    batch_pcl_unaligned = torch.stack(batch_pcl_unaligned)

    # Compute 3d <-> 2d projection mapping for each scene in the batch
    proj_mapping_list = []
    img_count = 0
    for d_img, c_pose in zip(depth_images, camera_poses):
        # TODO: double-check the curr_idx_batch
        curr_idx_batch = img_count // FLAGS.num_nearest_images
        if curr_idx_batch >= len(batch_scan_names):
            break
        # TEST
        # curr_scan_name = batch_scan_names[batch_idx]
        # pcl_root = "/home/kloping/Documents/TUM/3D_object_localization/data/scannet_point_clouds/"
        # pcl_path = os.path.join(pcl_root, curr_scan_name, curr_scan_name + "_vh_clean_2.ply")
        # destination = os.path.join(BASE_DIR, "utils", "test", "orig_mesh.ply")
        # shutil.copyfile(pcl_path, destination)
        # END of TEST
        projection = ProjectionHelper(batch_intrinsics[curr_idx_batch], FLAGS.depth_min, FLAGS.depth_max, proj_image_dims,
                                      NUM_POINT)
        proj_mapping = projection.compute_projection(batch_pcl_unaligned[curr_idx_batch], d_img, c_pose)
        proj_mapping_list.append(proj_mapping)
        img_count += 1

    if None in proj_mapping_list:  # invalid sample
        # print '(invalid sample)'
        return None, None, None
    proj_mapping = list(zip(*proj_mapping_list))
    proj_ind_3d = torch.stack(proj_mapping[0])
    proj_ind_2d = torch.stack(proj_mapping[1])

    # TODO: finish proxy loss part
    # if FLAGS.use_proxy_loss:
    #     data_util.load_label_frames(opt.data_path_2d, frames[v], label_images, num_classes)
    #     mask2d = label_images.view(-1).clone()
    #     for k in range(num_classes):
    #         if criterion_weights[k] == 0:
    #             mask2d[mask2d.eq(k)] = 0
    #     mask2d = mask2d.nonzero().squeeze()
    #     if (len(mask2d.shape) == 0):
    #         continue  # nothing to optimize for here

    # 2d features
    imageft_fixed = model2d_fixed(torch.autograd.Variable(color_images))
    imageft = model2d_trainable(imageft_fixed)
    # TODO: finish proxy loss part
    # if opt.use_proxy_loss:
    #     ft2d = model2d_classifier(imageft)
    #     ft2d = ft2d.permute(0, 2, 3, 1).contiguous()

    return proj_ind_3d, proj_ind_2d, imageft


word2index, zero_index = get_word2idx()

def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step()  # decay BN momentum
    net.train()  # set model to training mode

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if key == 'scan_name':
                batch_scan_names = batch_data_label[key]
                continue
            if key == 'description' or key=='objectid':
                continue
            batch_data_label[key] = batch_data_label[key].to(device)

        # =======================================
        # Get 3d <-> 2D Projection Mapping and 2D feature map
        # =======================================
        proj_ind_3d, proj_ind_2d, imageft = project_2d_features(batch_data_label)
        if proj_ind_3d is None or proj_ind_2d is None or imageft is None:
            warnings.warn("Current training script: Projection invalid with scans: {}".format(batch_scan_names))
            continue

        # TODO: XY flip is disable in dataloader, think about adding the flip back somewhere here.

        # Forward pass
        description = batch_data_label['description']
        token_sequence = []
        for i in range(BATCH_SIZE):
            tokens = description[i].split()
            indices = []
            for t in tokens:
                index = word2index[t]
                indices.append(index)
            token_sequence.append(torch.tensor(indices).long())
        batched_sequence = pad_sequence(token_sequence, batch_first=True, padding_value=zero_index)
        batched_sequence = batched_sequence.cuda()
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds'],
                  'batched_sequence': batched_sequence,
                  'objectid': batch_data_label['objectid']}
        # inputs3d = {'point_clouds_unaligned': batch_pcl_unaligned,
        #             'point_clouds': batch_data_label['point_clouds']}

        # testpts = batch_data_label['point_clouds'][0].type(torch.FloatTensor)
        # targetpts = batch_data_label['ground_truth_bbox'][0, 0:4].unsqueeze(0).type(torch.FloatTensor)
        # testpts = torch.cat((testpts,targetpts), 0).cpu().numpy()
        # color1 = np.zeros(shape=(20000, 3))
        # color2 = np.zeros(shape=(1,3))
        # color2.fill(255)
        # color = np.concatenate((color1,color2),axis=0)
        # write_ply_rgb(testpts,color, "./test_plys.ply")

        end_points = net(inputs, imageft, torch.autograd.Variable(proj_ind_3d), torch.autograd.Variable(proj_ind_2d), NUM_POINT)

        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)
        loss.backward()
        params = [p for p in net.parameters()]
        optimizer.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            TRAIN_VISUALIZER.log_scalars({key: stat_dict[key] / batch_interval for key in stat_dict},
                                         (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0


def evaluate_one_epoch():
    stat_dict = {}  # collect statistics
    ap_calculator = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
                                 class2type_map=DATASET_CONFIG.class2type)
    net.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % (batch_idx))
        for key in batch_data_label:
            if key == 'scan_name':
                batch_scan_names = batch_data_label[key]
                continue
            if key == 'description' or key == 'objectid':
                continue
            batch_data_label[key] = batch_data_label[key].to(device)

        # =======================================
        # Projection
        # =======================================
        proj_ind_3d, proj_ind_2d, imageft = project_2d_features(batch_data_label)
        if proj_ind_3d is None or proj_ind_2d is None or imageft is None:
            warnings.warn("Current training script: Projection invalid with scans: {]".format(batch_scan_names))
            continue

        # Forward pass
        description = batch_data_label['description']
        token_sequence = []
        for i in range(BATCH_SIZE):
            tokens = description[i].split()
            indices = []
            for t in tokens:
                index = word2index[t]
                indices.append(index)
            token_sequence.append(torch.tensor(indices).long())
        batched_sequence = pad_sequence(token_sequence, batch_first=True, padding_value=zero_index)
        batched_sequence = batched_sequence.cuda()
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds'],
                  'batched_sequence': batched_sequence,
                  'objectid': batch_data_label['objectid']}
        with torch.no_grad():
            # end_points = net(inputs)
            end_points = net(inputs, imageft, torch.autograd.Variable(proj_ind_3d),
                             torch.autograd.Variable(proj_ind_2d), NUM_POINT)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if FLAGS.dump_results and batch_idx == 0 and EPOCH_CNT % 10 == 0:
            MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

            # Log statistics
    TEST_VISUALIZER.log_scalars({key: stat_dict[key] / float(batch_idx + 1) for key in stat_dict},
                                (EPOCH_CNT + 1) * len(TRAIN_DATALOADER) * BATCH_SIZE)
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

    # Evaluate average precision
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f' % (key, metrics_dict[key]))

    mean_loss = stat_dict['loss'] / float(batch_idx + 1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()
        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9:  # Eval every 10 epochs
            loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': loss,
                     }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))


if __name__ == '__main__':
    train(start_epoch)
