import torch
import torch.nn as nn
from .layers import SpatialDescriptor, StructuralDescriptor, MeshConvolution


class MeshNet(nn.Module):
    def __init__(self, cfg):
        super(MeshNet, self).__init__()

        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor(cfg['structural_descriptor'])
        self.mesh_conv1 = MeshConvolution(cfg['mesh_convolution'], 64, 131, 256, 256)
        self.mesh_conv2 = MeshConvolution(cfg['mesh_convolution'], 256, 256, 512, 512)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(1792, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1)
        )

    def forward(self, graph):
        centers = graph.center
        corners = graph.corner
        normals = graph.normal
        neighbor_index = graph.neighbor_index

        spatial_fea0 = self.spatial_descriptor(centers)
        structural_fea0 = self.structural_descriptor(corners, normals, neighbor_index)

        spatial_fea1, structural_fea1 = self.mesh_conv1(spatial_fea0, structural_fea0, neighbor_index)
        spatial_fea2, structural_fea2 = self.mesh_conv2(spatial_fea1, structural_fea1, neighbor_index)
        spatial_fea3 = self.fusion_mlp(torch.cat([spatial_fea2, structural_fea2], 1))

        fea = self.concat_mlp(torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1))
        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)
        fea = self.classifier[:-1](fea)
        outputs = self.classifier[-1:](fea)

        return outputs
