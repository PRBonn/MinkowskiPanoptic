import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn


class MinkEncoderDecoder(nn.Module):
    """
    Basic ResNet architecture using sparse convolutions
    """

    def __init__(self, cfg):
        super().__init__()

        cr = cfg.CR
        self.D = cfg.DIMENSION
        input_dim = cfg.INPUT_DIM
        self.res = cfg.RESOLUTION

        cs = cfg.CHANNELS
        cs = [int(cr * x) for x in cs]
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(
                input_dim, cs[0], kernel_size=3, stride=1, dimension=self.D
            ),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(
                cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D
            ),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2, D=self.D),
                nn.Sequential(
                    ResidualBlock(
                        cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1, D=self.D
                    ),
                    ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1, D=self.D),
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2, D=self.D),
                nn.Sequential(
                    ResidualBlock(
                        cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1, D=self.D
                    ),
                    ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1, D=self.D),
                ),
            ]
        )

        self.up3 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2, D=self.D),
                nn.Sequential(
                    ResidualBlock(
                        cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1, D=self.D
                    ),
                    ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1, D=self.D),
                ),
            ]
        )

        self.up4 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2, D=self.D),
                nn.Sequential(
                    ResidualBlock(
                        cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1, D=self.D
                    ),
                    ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1, D=self.D),
                ),
            ]
        )

    def forward(self, x):
        in_field = self.TensorField(x)

        x0 = self.stem(in_field.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)

        return y4, in_field

    def TensorField(self, x):
        """
        Build a tensor field from coordinates and features in the 
        input batch
        The coordinates are quantized using the provided resolution

        """
        feat_tfield = ME.TensorField(
            features=torch.from_numpy(np.concatenate(x["feats"], 0)).float(),
            coordinates=ME.utils.batched_coordinates(
                [i / self.res for i in x["pt_coord"]], dtype=torch.float32
            ),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device="cuda",
        )
        return feat_tfield


## Blocks


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                inc, outc, kernel_size=ks, stride=stride, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                outc, outc, kernel_size=ks, dilation=dilation, stride=1, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
        )

        self.downsample = (
            nn.Sequential()
            if (inc == outc and stride == 1)
            else nn.Sequential(
                ME.MinkowskiConvolution(
                    inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D
                ),
                ME.MinkowskiBatchNorm(outc),
            )
        )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out
