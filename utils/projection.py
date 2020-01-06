
import numpy as np
import torch
from torch.autograd import Function

import imageio
import os
from utils import pc_util

class ProjectionHelper():
    def __init__(self, intrinsic, depth_min, depth_max, image_dims):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims


    def depth_to_skeleton(self, ux, uy, depth):
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.Tensor([depth*x, depth*y, depth])

    def skeleton_to_depth(self, p):
        x = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        y = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        return torch.Tensor([x, y, p[2]])

    def compute_frustum_bounds(self, camera_to_world, axis_align_matrix=None):
        corner_points = camera_to_world.new(8, 4, 1).fill_(1)
        # depth min
        corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min).unsqueeze(1)
        corner_points[1][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min).unsqueeze(1)
        corner_points[2][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1,
                                                      self.depth_min).unsqueeze(1)
        corner_points[3][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        # depth max
        corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max).unsqueeze(1)
        corner_points[5][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max).unsqueeze(1)
        corner_points[6][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1,
                                                      self.depth_max).unsqueeze(1)
        corner_points[7][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)

        p = torch.bmm(camera_to_world.repeat(8, 1, 1), corner_points)
        # pl = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.floor(p)))
        # pu = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.ceil(p)))

        p = p.squeeze()
        p = p.cpu().numpy()

        if axis_align_matrix is not None:
            pts = np.ones((p.shape[0], 4))
            pts[:, 0:3] = p[:, 0:3]
            pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
            p[:, 0:3] = pts[:, 0:3]

        p = torch.from_numpy(p)

        p = torch.unsqueeze(p, 2)

        bbox_min0, _ = torch.min(p[:, :3, 0], 0)
        # bbox_min1, _ = torch.min(pu[:, :3, 0], 0)
        # bbox_min = np.minimum(bbox_min0, bbox_min1)
        bbox_max0, _ = torch.max(p[:, :3, 0], 0)
        # bbox_max1, _ = torch.max(pu[:, :3, 0], 0)
        # bbox_max = np.maximum(bbox_max0, bbox_max1)
        return bbox_min0, bbox_max0


        # TODO make runnable on cpu as well...
    def compute_bounds(self, depth, camera_to_world, axis_align_matrix=None):
        # compute projection by voxels -> image
        world_to_camera = torch.inverse(camera_to_world)
        voxel_bounds_min, voxel_bounds_max = self.compute_frustum_bounds(camera_to_world, axis_align_matrix)
        return voxel_bounds_min, voxel_bounds_max, world_to_camera

    def compute_projection(self, pcl, depth, camera_to_world, axis_align_matrix=None):
        """

        :param pcl: torch.Tensor, set of points [x,y,z] in point cloud
        :param depth: torch.Tensor, depth map
        :param camera_to_world:
        :param axis_align_matrix: transformation matrix that adjust the pose and the center location of the point cloud
        :return: lin_indices_3d, lin_indices_2d
        """

        self.proj_reference(pcl, depth, camera_to_world)  # TEST
        self.write_pcl(pcl, depth, camera_to_world)  # TEST

        pcl = torch.transpose(pcl, 0, 1).cuda()  # shape: (4, N_points)

        # Compute frustum bounds
        world_to_camera = torch.inverse(camera_to_world)
        bounds_min, bounds_max = self.compute_frustum_bounds(camera_to_world, axis_align_matrix)
        bounds_min = bounds_min.cuda()
        bounds_max = bounds_max.cuda()

        # Get point cloud coordinates within frustum bounds
        mask_frustum_bounds_min = torch.ge(pcl[0], bounds_min[0]) * torch.ge(pcl[1], bounds_min[1]) * torch.ge(pcl[2], bounds_min[2])
        mask_frustum_bounds_max = torch.le(pcl[0], bounds_max[0]) * torch.le(pcl[1], bounds_max[1]) * torch.le(pcl[2], bounds_max[2])
        mask_frustum_bounds = mask_frustum_bounds_min * mask_frustum_bounds_max

        # filter_x = torch.ge(pcl[0], bounds_min[0]) * torch.le(pcl[0], bounds_max[0])
        # filter_y = torch.ge(pcl[1], bounds_min[1]) * torch.le(pcl[1], bounds_max[1])
        # filter_z = torch.ge(pcl[2], bounds_min[2]) * torch.le(pcl[2], bounds_max[2])
        # ones_uint8 = torch.ones(filter_x.size()[0], dtype=torch.uint8).cuda()
        # mask_frustum_bounds = torch.stack([filter_x, filter_y, filter_z, ones_uint8])\

        if not mask_frustum_bounds.any():
            print('error: nothing in frustum bounds')
            return None
        valid_indices = torch.arange(0, pcl.size()[1])
        valid_indices = valid_indices[mask_frustum_bounds].cuda()
        valid_vertices = torch.index_select(pcl, 1, valid_indices)

        # Transform to current frame
        p = torch.mm(world_to_camera, valid_vertices)  # (4,4) x (4,N) => (4,N)

        # project into image
        p[0] = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        p[1] = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        pi = torch.round(p).long()

        valid_ind_mask = torch.ge(pi[0], 0) * torch.ge(pi[1], 0) * torch.lt(pi[0], self.image_dims[0]) * torch.lt(pi[1], self.image_dims[1])
        if not valid_ind_mask.any():
            print('error: no valid image indices')
            return None
        valid_image_ind_x = pi[0][valid_ind_mask]
        valid_image_ind_y = pi[1][valid_ind_mask]
        valid_image_ind_lin = valid_image_ind_y * self.image_dims[0] + valid_image_ind_x
        depth_vals = torch.index_select(depth.view(-1), 0, valid_image_ind_lin)


        # ==============================
        # TEST
        # ==============================
        pi = pi.cpu().numpy()
        p = p.cpu().numpy()
        pi = np.transpose(pi)
        p = np.transpose(p)

        x = pi[:, 0]
        x_filter = (x >= 0) & (x < self.image_dims[0])
        y = pi[:, 1]
        y_filter = (y >= 0) & (y < self.image_dims[1])
        filterxy = x_filter & y_filter

        pi = pi[filterxy]
        p = p[filterxy]

        reconstructed_depth_map = np.zeros((self.image_dims[1], self.image_dims[0]))
        p_combined = np.concatenate((pi[:, 0:2], p[:, 2:3]), axis=1)

        for p in p_combined:
            reconstructed_depth_map[int(p[1]), int(p[0])] = p[2]

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        imageio.imwrite(BASE_DIR + '/test/reconstructed_depth_map.png', reconstructed_depth_map)
        orig_depth_map = depth.cpu().numpy()
        imageio.imwrite(BASE_DIR + '/test/orig_depth_map.png', orig_depth_map)

        print("finished")


        # ==============================
        # END of TEST
        # ==============================



    def proj_reference(self, pcl, depth, camera_to_world, axis_align_matrix=None):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # pcl = torch.transpose(pcl, 0, 1)  # shape: (4, N_points)
        mesh_vertices = pcl.numpy()

        # Compute frustum bounds
        world_to_camera = torch.inverse(camera_to_world)
        bounds_min, bounds_max = self.compute_frustum_bounds(camera_to_world, axis_align_matrix)
        bounds_min = bounds_min.numpy()
        bounds_max = bounds_max.numpy()

        # filter out the vertices that are not in the frustum (here treated as box-shape)
        filter1 = mesh_vertices[:, 0]
        filter1 = (filter1 >= bounds_min[0]) & (filter1 <= bounds_max[0])
        filter2 = mesh_vertices[:, 1]
        filter2 = (filter2 >= bounds_min[1]) & (filter2 <= bounds_max[1])
        filter3 = mesh_vertices[:, 2]
        filter3 = (filter3 >= bounds_min[2]) & (filter3 <= bounds_max[2])
        filter_all = filter1 & filter2 & filter3
        valid_vertices = mesh_vertices[filter_all]

        # transform to current frame
        world_to_camera = world_to_camera.cpu().numpy()
        N = valid_vertices.shape[0]
        valid_vertices_T = np.transpose(valid_vertices)
        pcamera = np.matmul(world_to_camera, valid_vertices_T)  # (4,4) x (4,N) => (4,N)

        p = np.transpose(pcamera)  # shape: (N,4)

        # project into image
        p[:, 0] = (p[:, 0] * self.intrinsic[0][0]) / p[:, 2] + self.intrinsic[0][2]
        p[:, 1] = (p[:, 1] * self.intrinsic[1][1]) / p[:, 2] + self.intrinsic[1][2]
        pi = np.rint(p)

        x = pi[:, 0]
        x_filter = (x >= 0) & (x < self.image_dims[0])
        y = pi[:, 1]
        y_filter = (y >= 0) & (y < self.image_dims[1])
        filterxy = x_filter & y_filter

        pi = pi[filterxy]
        p = p[filterxy]

        reconstructed_depth_map = np.zeros((self.image_dims[1], self.image_dims[0]))
        p_combined = np.concatenate((pi[:, 0:2], p[:, 2:3]), axis=1)
        for p in p_combined:
            reconstructed_depth_map[int(p[1]), int(p[0])] = p[2]

        imageio.imwrite(BASE_DIR + '/test/reference_depth_map.png', reconstructed_depth_map)


    def write_pcl(self, pcl, depth, camera_to_world, axis_align_matrix=None):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # output point cloud from depth_image
        depth_map_to_compare = depth.cpu().numpy()
        pointstowrite = np.ones((320 * 240, 4))
        colors = np.ones((320 * 240, 4))

        pc_util.write_ply_rgb(pcl.numpy(), colors, BASE_DIR + '/test/sampled_pcl.obj')

        # TODO: convert this to matrix multiplication
        print("back-projection, depth-map -> 3d points")
        for i1 in range(self.image_dims[0]):
            for i2 in range(self.image_dims[1]):
                # pcamera = self.depth_to_skeleton(i1, i2, depth_map_to_compare[i2, i1]).unsqueeze(1).cpu().numpy()
                pcamera = self.depth_to_skeleton(i2, i1, depth_map_to_compare[i2, i1]).unsqueeze(1).cpu().numpy()
                pcamera = np.append(pcamera, np.ones((1, 1)), axis=0)
                camera2world = camera_to_world.cpu().numpy()
                world = np.matmul(camera2world, pcamera)
                world = world.reshape((1, 4))
                pointstowrite[i1 * i2, :] = world[0, :]
        pointstowrite = pointstowrite[:, 0:3]

        pc_util.write_ply(pointstowrite, BASE_DIR + '/test/testobject.ply')
        # pc_util.write_ply(pcl.numpy(), BASE_DIR + '/orig_object.ply')
        # pc_util.write_ply_rgb(pcl.numpy(), colors, BASE_DIR + '/test/sampled_pcl.obj')






# Inherit from Function
class Projection(Function):

    @staticmethod
    def forward(ctx, label, lin_indices_3d, lin_indices_2d, volume_dims):
        ctx.save_for_backward(lin_indices_3d, lin_indices_2d)
        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0]
        output = label.new(num_label_ft, volume_dims[2], volume_dims[1], volume_dims[0]).fill_(0)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1+num_ind])
            output.view(num_label_ft, -1)[:, lin_indices_3d[1:1+num_ind]] = vals
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_label = grad_output.clone()
        num_ft = grad_output.shape[0]
        grad_label.data.resize_(num_ft, 32, 41)
        lin_indices_3d, lin_indices_2d = ctx.saved_variables
        num_ind = lin_indices_3d.data[0]
        vals = torch.index_select(grad_output.data.contiguous().view(num_ft, -1), 1, lin_indices_3d.data[1:1+num_ind])
        grad_label.data.view(num_ft, -1)[:, lin_indices_2d.data[1:1+num_ind]] = vals
        return grad_label, None, None, None

