import torch.nn as nn
from functools import reduce
from operator import mul
import torch

class Reconstruction3DEncoder(nn.Module):
    def __init__(self, chnum_in):
        super(Reconstruction3DEncoder, self).__init__()

        # Dong Gong's paper code
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        feature_num_x3 = 512
        self.encoder0 = nn.Sequential(
            nn.Conv3d(self.chnum_in, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder1 = nn.Sequential(
            nn.Conv3d(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # self.encoder4 = nn.Sequential(
        #     nn.Conv3d(feature_num_x2, feature_num_x3, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(feature_num_x3),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        # x4 = self.encoder4(x3)
        return x3


class Reconstruction3DDecoder(nn.Module):
    def __init__(self, chnum_in):
        super(Reconstruction3DDecoder, self).__init__()

        # Dong Gong's paper code + Tanh
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        feature_num_x3 = 512

        # self.de0 = nn.Sequential(
        #     nn.ConvTranspose3d(feature_num_x3, feature_num_x3, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
        #                        output_padding=(1, 1, 1)),
        #     nn.BatchNorm3d(feature_num_x3),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        self.de1 = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.de2 = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.de3 = nn.Sequential(
            nn.ConvTranspose3d(feature_num, feature_num_2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.de4 = nn.Sequential(
            nn.ConvTranspose3d(feature_num_2, self.chnum_in, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.Tanh()
        )


    def forward(self, x):
        # x0 = self.de0(x)
        x1 = self.de1(x)
        x2 = self.de2(x1)
        x3 = self.de3(x2)
        x4 = self.de4(x3)
        return x1, x2, x3, x4

class Reconstruction3DDecoder0(nn.Module):
    def __init__(self, chnum_in):
        super(Reconstruction3DDecoder0, self).__init__()

        # Dong Gong's paper code + Tanh
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        feature_num_x3 = 512

        # self.de0 = nn.Sequential(
        #     nn.ConvTranspose3d(feature_num_x3, feature_num_x3, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
        #                        output_padding=(1, 1, 1)),
        #     nn.BatchNorm3d(feature_num_x3),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        self.de1 = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.de2 = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.de3 = nn.Sequential(
            nn.ConvTranspose3d(feature_num, feature_num_2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.de4 = nn.Sequential(
            nn.ConvTranspose3d(feature_num_2, self.chnum_in, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.Tanh()
        )


    def forward(self, x):
        # x0 = self.de0(x)
        x1 = self.de1(x)
        x2 = self.de2(x1)
        x3 = self.de3(x2)
        x4 = self.de4(x3)
        return x1, x2, x3, x4

