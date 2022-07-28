import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(x, size):
    if x.shape[-2] != size[0] or x.shape[-1] != size[1]:
        return F.interpolate(x, size, mode='bilinear', align_corners=True)
    else:
        return x


def bi_interaction(x_h, x_l):
    sizeH = (int(x_h.shape[-2]), int(x_h.shape[-1]))
    sizeL = (int(x_l.shape[-2]), int(x_l.shape[-1]))
    o_h = x_h + upsample(x_l, sizeH)
    o_l = x_l + upsample(x_h, sizeL)
    return o_h, o_l


def tr_interaction(x1, x2, x3):
    s1 = (int(x1.shape[-2]), int(x1.shape[-1]))
    s2 = (int(x2.shape[-2]), int(x2.shape[-1]))
    s3 = (int(x3.shape[-2]), int(x3.shape[-1]))
    o1 = x1 + upsample(x2, s1) + upsample(x3, s1)
    o2 = x2 + upsample(x1, s2) + upsample(x3, s2)
    o3 = x3 + upsample(x2, s3) + upsample(x1, s3)
    return o1, o2, o3


class ConvBNReLU3x3(nn.Sequential):

    def __init__(self, c_in, c_out, stride, deploy=False):
        if deploy:
            super(ConvBNReLU3x3, self).__init__(
                nn.Conv2d(c_in, c_out, 3, stride, 1, bias=True),
                nn.ReLU(inplace=True)
            )
        else:
            super(ConvBNReLU3x3, self).__init__(
                nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True)
            )


class BaseNet(nn.Module):

    def __init__(self, layers, channels, deploy=False):
        super(BaseNet, self).__init__()
        self.layers = layers
        assert len(self.layers) == 5
        self.channels = channels
        assert len(self.channels) == 5
        self.strides = (2, 2, 2, 2, 1)

        self.stages = nn.ModuleList()
        c_in = 3
        for l, c, s in zip(self.layers, self.channels, self.strides):
            self.stages.append(self._make_stage(c_in, c, l, s, deploy))
            c_in = c

    @staticmethod
    def _make_stage(c_in, c_out, numlayer, stride, deploy):
        layers = []
        for i in range(numlayer):
            layers.append(ConvBNReLU3x3(c_in if i == 0 else c_out, c_out, stride if i == 0 else 1, deploy))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for s in self.stages:
            out = s(out)
        return out


class BaseSegNet(nn.Module):

    def __init__(self, depth, width, resolution, num_classes=19):
        super(BaseSegNet, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        assert len(self.depth) == 5
        self.width = width
        assert len(self.width) == 5
        self.resolution = resolution
        if len(resolution) == 1:
            assert resolution[0] > 0
        else:
            for r1, r2 in zip(resolution[:-1], resolution[1:]):
                assert r1 >= r2

    def forward(self, x):
        raise NotImplementedError


class LPSNet1Path(BaseSegNet):

    def __init__(self, depth, width, resolution, num_classes=19, deploy=False):
        super(LPSNet1Path, self).__init__(depth, width, resolution, num_classes)
        self._check_resolution()
        self.net = BaseNet(self.depth, self.width, deploy)
        self.head = nn.Conv2d(self.width[-1], self.num_classes, 1, 1, 0, bias=True)

    def forward(self, x):
        return self.head(self.net(self._preprocess_input(x)))

    def _preprocess_input(self, x):
        r = self.resolution[0]
        return upsample(x, (int(x.shape[-2] * r), int(x.shape[-1] * r)))

    def _check_resolution(self):
        assert self.resolution[0] > 0
        for i in range(1, len(self.resolution)):
            assert self.resolution[i] == 0


class LPSNet2Path(BaseSegNet):

    def __init__(self, depth, width, resolution, num_classes=19, deploy=False):
        super(LPSNet2Path, self).__init__(depth, width, resolution, num_classes)
        self._check_resolution()
        self.netH = BaseNet(self.depth, self.width, deploy)
        self.netL = BaseNet(self.depth, self.width, deploy)
        self.head = nn.Conv2d(self.width[-1] * 2, self.num_classes, 1, 1, 0, bias=True)

    def _check_resolution(self):
        assert len(self.resolution) >= 2
        assert self.resolution[0] > 0
        assert self.resolution[1] > 0
        assert self.resolution[0] >= self.resolution[1]
        for i in range(2, len(self.resolution)):
            assert self.resolution[i] == 0

    def _preprocess_input(self, x):
        r1 = self.resolution[0]
        r2 = self.resolution[1]
        x1 = upsample(x, (int(x.shape[-2] * r1), int(x.shape[-1] * r1)))
        x2 = upsample(x, (int(x.shape[-2] * r2), int(x.shape[-1] * r2)))
        return x1, x2

    def forward(self, x):
        xh, xl = self._preprocess_input(x)
        xh, xl = self.netH.stages[0](xh), self.netL.stages[0](xl)
        xh, xl = self.netH.stages[1](xh), self.netL.stages[1](xl)
        xh, xl = self.netH.stages[2](xh), self.netL.stages[2](xl)
        xh, xl = bi_interaction(xh, xl)
        xh, xl = self.netH.stages[3](xh), self.netL.stages[3](xl)
        xh, xl = bi_interaction(xh, xl)
        xh, xl = self.netH.stages[4](xh), self.netL.stages[4](xl)
        x_cat = torch.cat([xh, upsample(xl, (int(xh.shape[-2]), int(xh.shape[-1])))], dim=-3)
        return self.head(x_cat)


class LPSNet3Path(BaseSegNet):

    def __init__(self, depth, width, resolution, num_classes=19, deploy=False):
        super(LPSNet3Path, self).__init__(depth, width, resolution, num_classes)
        self._check_resolution()
        self.net1 = BaseNet(self.depth, self.width, deploy)
        self.net2 = BaseNet(self.depth, self.width, deploy)
        self.net3 = BaseNet(self.depth, self.width, deploy)
        self.head = nn.Conv2d(self.width[-1] * 3, self.num_classes, 1, 1, 0, bias=True)

    def _check_resolution(self):
        assert len(self.resolution) >= 3
        assert self.resolution[0] > 0
        assert self.resolution[1] > 0
        assert self.resolution[2] > 0
        assert self.resolution[0] >= self.resolution[1] >= self.resolution[2]
        for i in range(3, len(self.resolution)):
            assert self.resolution[i] == 0

    def _preprocess_input(self, x):
        r1 = self.resolution[0]
        r2 = self.resolution[1]
        r3 = self.resolution[2]
        x1 = upsample(x, (int(x.shape[-2] * r1), int(x.shape[-1] * r1)))
        x2 = upsample(x, (int(x.shape[-2] * r2), int(x.shape[-1] * r2)))
        x3 = upsample(x, (int(x.shape[-2] * r3), int(x.shape[-1] * r3)))
        return x1, x2, x3

    def forward(self, x):
        x1, x2, x3 = self._preprocess_input(x)
        x1, x2, x3 = self.net1.stages[0](x1), self.net2.stages[0](x2), self.net3.stages[0](x3)
        x1, x2, x3 = self.net1.stages[1](x1), self.net2.stages[1](x2), self.net3.stages[1](x3)
        x1, x2, x3 = self.net1.stages[2](x1), self.net2.stages[2](x2), self.net3.stages[2](x3)
        x1, x2, x3 = tr_interaction(x1, x2, x3)
        x1, x2, x3 = self.net1.stages[3](x1), self.net2.stages[3](x2), self.net3.stages[3](x3)
        x1, x2, x3 = tr_interaction(x1, x2, x3)
        x1, x2, x3 = self.net1.stages[4](x1), self.net2.stages[4](x2), self.net3.stages[4](x3)
        x_cat = [x1,
                 upsample(x2, (int(x1.shape[-2]), int(x1.shape[-1]))),
                 upsample(x3, (int(x1.shape[-2]), int(x1.shape[-1])))]
        x_cat = torch.cat(x_cat, dim=-3)
        return self.head(x_cat)


def get_lpsnet(depth, width, resolution, num_classes=19, deploy=False):
    assert all([d > 0 for d in depth])
    assert all([w > 0 for w in width])
    assert num_classes > 1
    resolution_filter = list(filter(lambda x: x > 0, resolution))
    resolution_sorted = sorted(resolution_filter, reverse=True)
    if len(resolution_sorted) == 1:
        return LPSNet1Path(depth, width, resolution_sorted, num_classes, deploy)
    elif len(resolution_sorted) == 2:
        return LPSNet2Path(depth, width, resolution_sorted, num_classes, deploy)
    elif len(resolution_sorted) == 3:
        return LPSNet3Path(depth, width, resolution_sorted, num_classes, deploy)
    else:
        raise NotImplementedError


def get_lspnet_s(deploy=False):
    depth = (1, 3, 3, 10, 10)
    width = (8, 24, 48, 96, 96)
    resolution = (3 / 4, 1 / 4, 0)
    net = get_lpsnet(depth, width, resolution, deploy=deploy)
    return net


def get_lspnet_m(deploy=False):
    depth = (1, 3, 3, 10, 10)
    width = (8, 24, 48, 96, 96)
    resolution = (1, 1 / 4, 0)
    net = get_lpsnet(depth, width, resolution, deploy=deploy)
    return net

def get_lspnet_l(deploy=False):
    depth = (1, 3, 3, 10, 10)
    width = (8, 24, 64, 160, 160)
    resolution = (1, 1 / 4, 0)
    net = get_lpsnet(depth, width, resolution, deploy=deploy)
    return net
