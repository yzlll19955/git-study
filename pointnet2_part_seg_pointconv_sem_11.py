import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from pointconv_util import PointConvDensitySetAbstraction,PointConvSetAbstraction,PointConvFeaturePropagation
#做对比

#测试message文件
#测试message文件2
#测试message文件3
#测试message文件4
#测试message文件5
class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        # npoint, nsample, in_channel, mlp, bandwidth, group_all
        feature_dim = 3
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=feature_dim + 3, mlp=[32, 32, 64], bandwidth=0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=256, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128],bandwidth=0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=64, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth=0.4, group_all=False)
        # self.sa4 = PointConvDensitySetAbstraction(npoint=16, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512], bandwidth=0.8, group_all=False)
        # npoint, nsample, in_channel, out_put, mlp, bandwidth, group_all
        self.fp4 = PointConvFeaturePropagation(npoint=512, nsample=32, in_channel=256+3, out_put=384, mlp=[512, 256], bandwidth=0.8, group_all=False)
        self.fp3 = PointConvFeaturePropagation(npoint=256, nsample=32, in_channel=131, out_put=192, mlp=[256, 256], bandwidth=0.4, group_all=False)
        self.fp2 = PointConvFeaturePropagation(npoint=256, nsample=32, in_channel=64+3, out_put=64, mlp=[256, 128], bandwidth=0.2, group_all=False)
        # self.fp1 = PointConvFeaturePropagation(npoint=128, nsample=32, in_channel=128+3, out_put=128, mlp=[128, 128, 128], bandwidth=0.1, group_all=False)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = self.fp3(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = self.fp2(l0_xyz, l1_xyz, None, l1_points)
        # l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l1_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

#thank you
        return total_loss
#bbb1
