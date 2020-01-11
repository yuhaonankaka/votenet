import os, sys
import shutil
import numpy as np
import torch
import warnings

from train_with_enet2d import project_2d_features, get_batch_intrinsics, my_worker_init_fn
from scannet import scannet_utils
from utils import pc_util
from torch.utils.data import DataLoader
from utils.projection import ProjectionHelper
from utils import image_util
from scannet.scannet_detection_dataset import ScannetDetectionDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)



def test_lin_indices_3d_2d(label, lin_indices_3d, lin_indices_2d):
    num_label_ft = 1
    output = label.new(num_label_ft, 240, 320).fill_(0)
    num_ind = lin_indices_3d[0]
    if num_ind > 0:
        vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1 + num_ind])
        output.view(num_label_ft, -1)[:, lin_indices_3d[1:1 + num_ind]] = vals

    print("finished")



def test_unalign_pcl(pcl_aligned: np.ndarray, pcl_unaligned: np.ndarray, scan_name: str, axis_align_matrix: np.ndarray):
    save_root = os.path.join(BASE_DIR, "test_tmp")
    # 1. Output original Mesh
    curr_scan_name = scan_name
    pcl_root = "/home/kloping/Documents/TUM/3D_object_localization/data/scannet_point_clouds/"
    pcl_path = os.path.join(pcl_root, curr_scan_name, curr_scan_name + "_vh_clean_2.ply")
    destination = os.path.join(save_root, "orig_pcl.ply")
    shutil.copyfile(pcl_path, destination)

    inv_axis_align_matrix_T = np.linalg.inv(axis_align_matrix.T)

    # 1.1 Read in Mesh Vertices
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(destination)
    # 1.2 Align the Mesh Vertices
    mesh_pts = np.ones((mesh_vertices.shape[0], 4))
    mesh_pts[:, 0:3] = mesh_vertices[:, 0:3]
    mesh_pts = np.dot(mesh_pts, axis_align_matrix.transpose())  # Nx4
    # 1.3 Store aligned mesh_vertices
    pc_util.write_ply(mesh_pts, save_root + '/mesh_aligned.ply')
    # 1.4 Unalign the Mesh Vertices and write to file
    mesh_unaligned = np.dot(mesh_pts, inv_axis_align_matrix_T)
    pc_util.write_ply(mesh_unaligned, save_root + '/mesh_unaligned.ply')

    # 2. Output aligned Point Cloud
    pcl_aligned = pcl_aligned.cpu().numpy()
    pc_util.write_ply(pcl_aligned, save_root + '/pcl_aligned.ply')

    # 3. Output unaligned Point Cloud
    pc_util.write_ply(pcl_unaligned, save_root + '/pcl_unaligned.ply')

    # 4. Output reversed-aligned Point Cloud
    pcl_reversed = np.dot(pcl_unaligned, axis_align_matrix.T)
    pc_util.write_ply(pcl_reversed, save_root + '/pcl_reversed.ply')

    print("Test results stored in {}".format(save_root))


def test_backprojection_coverage(num_images: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    proj_image_dims = [40, 30]
    input_image_dims = [320, 240]
    data_path_2d = "/home/kloping/Documents/TUM/3D_object_localization/data/frames_square"
    color_mean = [0.496342, 0.466664, 0.440796]
    color_std = [0.277856, 0.28623, 0.291129]
    depth_min=0.4
    depth_max=4.0

    BATCH_SIZE = 4
    NUM_POINT = 20000
    RAW_DATA_DIR = '/home/kloping/Documents/TUM/3D_object_localization/data/scannet_point_clouds/'

    TRAIN_DATASET = ScannetDetectionDataset('train', num_points=NUM_POINT,
                                            augment=True,
                                            use_color=False, use_height=True)
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
                                           augment=False,
                                           use_color=False, use_height=True)
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=4, worker_init_fn=my_worker_init_fn)

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        print("Testing {} batch".format(batch_idx))
        for key in batch_data_label:
            if key == 'scan_name':
                batch_scan_names = batch_data_label[key]
                continue
            batch_data_label[key] = batch_data_label[key].to(device)

        # ===============================================================
        # Find 3d <-> 2D Projection Mapping
        # ===============================================================
        batch_scan_names = batch_data_label["scan_name"]

        # Get camera intrinsics for each scene
        batch_intrinsics = get_batch_intrinsics(batch_scan_names)

        # Get 2d images and it's feature
        depth_images = torch.cuda.FloatTensor(BATCH_SIZE * num_images, proj_image_dims[1], proj_image_dims[0])
        color_images = torch.cuda.FloatTensor(BATCH_SIZE * num_images, 3, input_image_dims[1], input_image_dims[0])
        camera_poses = torch.cuda.FloatTensor(BATCH_SIZE * num_images, 4, 4)
        label_images = torch.cuda.LongTensor(BATCH_SIZE * num_images, proj_image_dims[1],
                                             proj_image_dims[0])  # for proxy loss

        image_util.load_frames_multi(data_path_2d, batch_scan_names, num_images,
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

        batch_pcl_unaligned = torch.stack(batch_pcl_unaligned)

        # Compute 3d <-> 2d projection mapping for each scene in the batch
        proj_mapping_list = []
        img_count = 0
        for d_img, c_pose in zip(depth_images, camera_poses):
            # TODO: double-check the curr_idx_batch
            curr_idx_batch = img_count // num_images
            if curr_idx_batch >= len(batch_scan_names):
                break

            projection = ProjectionHelper(batch_intrinsics[curr_idx_batch], depth_min, depth_max,
                                          proj_image_dims, NUM_POINT)
            proj_mapping = projection.compute_projection(batch_pcl_unaligned[curr_idx_batch], d_img, c_pose)
            proj_mapping_list.append(proj_mapping)
            img_count += 1

        if None in proj_mapping_list:  # invalid sample
            warnings.warn('(invalid sample)')
            continue
        proj_mapping = list(zip(*proj_mapping_list))
        proj_ind_3d = torch.stack(proj_mapping[0])
        proj_ind_2d = torch.stack(proj_mapping[1])

        # ===============================================================
        # ===============================================================

        # TODO: XY flip is disable in dataloader, think about adding the flip back somewhere here.

        assert len(proj_ind_3d) == len(proj_ind_2d)

        inputs3d = {'point_clouds': batch_data_label['point_clouds']}

        # =======================================================================================
        #        Check 3D <-> 2D mapping
        # 1. Compute depth value from lin_indices_2d & current depth map
        #    and got shape (1, NUM_POINT, batch_size*num_images)   --> ref_depth
        # 2. Compute the depth value from (x,y,z) for the valid indices (get from lin_indices_3d) --> calc_depth
        # 3. Compare ref_depth and calc_depth to see if they are close.
        # =======================================================================================
        for i in range(len(proj_ind_3d)):
            curr_idx_batch = i // num_images
            lin_indices_3d = proj_ind_3d[i]
            lin_indices_2d = proj_ind_2d[i]
            test_array_3d = lin_indices_3d.cpu().numpy()
            num_valid_idx = test_array_3d[0]
            testTTT = test_array_3d[num_valid_idx]
            test112 = test_array_3d[num_valid_idx + 1]
            test113 = test_array_3d[num_valid_idx + 2]
            test_array_3d = test_array_3d[1:num_valid_idx]


            camera_to_world = camera_poses[i].cpu().numpy()
            world_to_camera = np.linalg.inv(camera_to_world)

            assert lin_indices_3d[0] == lin_indices_2d[0]
            depth_map = depth_images[i]

            num_label_ft = 1  # just (x,y,z)
            # depth_map =
            output = depth_map.new(num_label_ft, NUM_POINT).fill_(0)
            num_ind = lin_indices_3d[0]
            if num_ind > 0:
                vals = torch.index_select(depth_map.view(num_label_ft, -1), 1, lin_indices_2d[1:1 + num_ind])
                output.view(num_label_ft, -1)[:, lin_indices_3d[1:1 + num_ind]] = vals
            output = output.cpu().numpy()
            non_zero_count = np.count_nonzero(output)
            non_zero_idx = np.nonzero(output)
            # non_zero_idx should be similar with test_array_3d

            for idx_3d in test_array_3d:
                pcl_unaligned = batch_pcl_unaligned[curr_idx_batch]
                pcl_unaligned = pcl_unaligned.cpu().numpy()
                # project to camera space
                pworld = pcl_unaligned[idx_3d].reshape(4, 1)
                pcamera = np.matmul(world_to_camera, pworld)  # (4,4) x (4,1) => (4,1)
                p = np.transpose(pcamera)  # (1,4)
                # project into image space
                p[:, 0] = (p[:, 0] * projection.intrinsic[0][0]) / p[:, 2] + projection.intrinsic[0][2]
                p[:, 1] = (p[:, 1] * projection.intrinsic[1][1]) / p[:, 2] + projection.intrinsic[1][2]
                pi = np.rint(p)
                idx_2d = pi[0][1] * 40 + pi[0][0]
                calc_depth = p[0][2]
                ref_depth = output[0][idx_3d]
                is_close = np.isclose(calc_depth, ref_depth, atol=0.1, rtol=0)
                if not is_close:
                    warnings.warn(batch_scan_names[curr_idx_batch])
                    sys.exit()
        # =========================================
        # End: Check 3D <-> 2D mapping
        # =========================================

    print("Test passed")








    #     # Accumulate statistics and print out
    #     for key in end_points:
    #         if 'loss' in key or 'acc' in key or 'ratio' in key:
    #             if key not in stat_dict: stat_dict[key] = 0
    #             stat_dict[key] += end_points[key].item()
    #
    #     batch_interval = 10
    #     if (batch_idx + 1) % batch_interval == 0:
    #         log_string(' ---- batch: %03d ----' % (batch_idx + 1))
    #         TRAIN_VISUALIZER.log_scalars({key: stat_dict[key] / batch_interval for key in stat_dict},
    #                                      (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * BATCH_SIZE)
    #         for key in sorted(stat_dict.keys()):
    #             log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
    #             stat_dict[key] = 0
    # print(data_path, num_images)


if __name__ == '__main__':
    print("This script contains helper functions for testing")
    test_backprojection_coverage(num_images=10)
