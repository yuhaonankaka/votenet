# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

from torch.nn.utils.rnn import pad_sequence

from nlp import LanguageNet, getweights, get_word2idx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import warnings
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper import get_loss
from utils.projection import Projection


class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_images, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_images = num_images
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        self.pooling = nn.MaxPool1d(kernel_size=num_images)

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling)

    def forward(self, inputs, imageft, projection_indices_3d, projection_indices_2d, num_sampled_points):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
            imageft: TODO
            projection_indices_3d: TODO
            projection_indices_2d: TODO
            num_sampled_points: TODO
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]
        description = inputs['description']
        # Back project imageft to 3d space and concat with point_clouds
        num_images = projection_indices_3d.shape[0] // batch_size
        imageft_back3d = [Projection.apply(ft, ind3d, ind2d, num_sampled_points)
                          for ft, ind3d, ind2d in zip(imageft, projection_indices_3d, projection_indices_2d)]
        imageft_back3d = torch.stack(imageft_back3d, dim=2)  # shape: (n_ft_channels, n_sampled_pts, batch_size*n_img)

        # Max Pool
        if num_images == self.num_images:
            imageft_back3d = self.pooling(imageft_back3d)  # shape: (n_ft_channels, n_sampled_pts, batch_size)
        else:
            warnings.warn("votenet.py: num_images != self.num_images")
            imageft_back3d = nn.MaxPool1d(kernel_size=num_images)(imageft_back3d)

        # Rearrange the dims
        imageft_back3d = imageft_back3d.permute(2, 1, 0)

        # Directly use the aligned pcl to concat features, because we already have the mappings of indices.
        # Alignment operation does not change indices, but only the (x,y,z) value.
        pcl_enriched = torch.cat((inputs['point_clouds'], imageft_back3d), dim=2)

        # end_points = self.backbone_net(inputs['point_clouds'], end_points)
        end_points = self.backbone_net(pcl_enriched, end_points)
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.pnet(xyz, features, end_points)

        word2index, zero_index = get_word2idx()
        languagenet = LanguageNet(getweights(), 128, 1)
        token_sequence = []
        for i in range(batch_size):
            tokens = description[i].split()
            indices = []
            for t in tokens:
                index = word2index[t]
                indices.append(index)
            token_sequence.append(torch.tensor(indices).long())
        batched_sequence = pad_sequence(token_sequence, batch_first=True, padding_value=zero_index)
        batch_lan_features = languagenet(batched_sequence)[1]
        enriched_nlp = torch.zeros(batch_size,256,256)
        aggregated_features = end_points['aggregated_vote_features']
        aggregated_features = aggregated_features.permute(0,2,1)
        enriched_nlp[:,:,0:128] = aggregated_features[:,:,0:128]
        batch_lan_features = torch.squeeze(batch_lan_features)
        nlp_features = batch_lan_features.repeat_interleave(256,dim=1).view(batch_size,128,256)
        nlp_features = nlp_features.permute(0,2,1)
        enriched_nlp[:,:,128:256] = nlp_features[:,:,0:128]
        #TODO object mask
        mlp = MLP()
        fusion_features = mlp(enriched_nlp)
        slp = SLP()
        final_score = slp(fusion_features)
        final_score = torch.squeeze(final_score)
        end_points['final_score'] = final_score
        return end_points


if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    from loss_helper import get_loss

    # Define model
    model = VoteNet(10,12,10,np.random.random((10,3))).cuda()
    
    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # Model forward pass
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        inputs = {'point_clouds': torch.rand((20000,3)).unsqueeze(0).cuda()}

    end_points = model(inputs)
    for key in end_points:
        print(key, end_points[key])

    try:
        # Compute loss
        for key in sample:
            end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
        loss, end_points = get_loss(end_points, DC)
        print('loss', loss)
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = np.ones((1,128))
        dump_results(end_points, 'tmp', DC)
    except:
        print('Dataset has not been prepared. Skip loss and dump.')


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layers(x)
        return x

class SLP(nn.Module):
    def __init__(self):
        super(SLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 1),
            # nn.Softmax(1),
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layers(x)
        return x
