import torch
import hashlib
import random
import numpy as np
from lpsnet import get_lpsnet
from latency import measure_latency_ms


class LPSNetExpansion:

    def __init__(self, depth, width, resolution, parent=None):
        self.depth = depth
        self.width = width
        self.resolution = resolution
        self._check_net_param()
        self.parent = parent
        self.latency = self.eval_latency()
        self.miou = -1

        self.delta_depth = ((0, 1, 1, 1, 1), (0, 0, 1, 1, 1), (0, 0, 0, 1, 1))
        self.delta_width = ((4, 8, 16, 32, 32), (0, 8, 16, 32, 32), (0, 0, 16, 32, 32), (0, 0, 0, 32, 32))
        self.delta_resolution = ((1/8, 0, 0), (0, 1/8, 0), (0, 0, 1/8))

    def _check_net_param(self):
        assert len(self.depth) == 5
        assert len(self.width) == 5
        assert len(self.resolution) == 3
        assert all([d > 0 for d in self.depth])
        assert all([w > 0 for w in self.width])
        assert any([r > 0 for r in self.resolution])

    def __str__(self):
        return '{}-{}-{}-{}-{}'.format(self.depth, self.width,
                                       sorted(self.resolution, reverse=True),
                                       self.latency, self.miou)

    @property
    def hash(self):
        arch = '{}-{}-{}'.format(self.depth, self.width, sorted(self.resolution, reverse=True))
        sha1 = hashlib.sha1(arch.encode("utf-8"))
        return sha1.hexdigest()

    def eval_latency(self, datashape=(1, 3, 1024, 2048), num_cls=19):
        net = get_lpsnet(self.depth, self.width, self.resolution, num_cls)
        data = torch.rand(datashape)
        lat = measure_latency_ms(net, data)
        return lat

    def _expand_depth(self, op, steplen=1):
        new_depth = [i + j * steplen for i, j in zip(self.depth, op)]
        arch = self.__class__(depth=new_depth, width=self.width, resolution=self.resolution, parent=self)
        print('    expand_depth: {} + {} * {} = {}, latency={:>0.4f}'.format(
            self.depth, op, steplen, arch.depth, arch.latency))
        return arch

    def _expand_width(self, op, steplen=1):
        new_width = [i + j * steplen for i, j in zip(self.width, op)]
        arch = self.__class__(depth=self.depth, width=new_width, resolution=self.resolution, parent=self)
        print('    expand_width: {} + {} * {} = {}, latency={:>0.4f}'.format(
            self.width, op, steplen, arch.width, arch.latency))
        return arch

    def _expand_resolution(self, op, steplen=1):
        new_resolution = [i + j * steplen for i, j in zip(self.resolution, op)]
        arch = self.__class__(depth=self.depth, width=self.width, resolution=new_resolution, parent=self)
        print('    expand_resolution: {} + {} * {} = {}, latency={:>0.4f}'.format(
            self.resolution, op, steplen, arch.resolution, arch.latency))
        return arch

    def _expand_depth_to_target(self, target):
        arch_save = []
        for op in self.delta_depth:
            arch_tmp = [self]
            steplen = 1
            while arch_tmp[-1].latency < target:
                arch_tmp.append(self._expand_depth(op, steplen))
                steplen += 1
            if abs(target - arch_tmp[-2].latency) < abs(target - arch_tmp[-1].latency):
                arch_save.append(arch_tmp[-2])
            else:
                arch_save.append(arch_tmp[-1])
        assert len(arch_save) == len(self.delta_depth)
        return arch_save

    def _expand_width_to_target(self, target):
        arch_save = []
        for op in self.delta_width:
            arch_tmp = [self]
            steplen = 1
            while arch_tmp[-1].latency < target:
                arch_tmp.append(self._expand_width(op, steplen))
                steplen += 1
            if abs(target - arch_tmp[-2].latency) < abs(target - arch_tmp[-1].latency):
                arch_save.append(arch_tmp[-2])
            else:
                arch_save.append(arch_tmp[-1])
        assert len(arch_save) == len(self.delta_width)
        return arch_save

    def _expand_resolution_to_target(self, target):
        arch_save = []
        for op in self.delta_resolution:
            arch_tmp = [self]
            steplen = 1
            while arch_tmp[-1].latency < target:
                arch_tmp.append(self._expand_resolution(op, steplen))
                steplen += 1
            if abs(target - arch_tmp[-2].latency) < abs(target - arch_tmp[-1].latency):
                arch_save.append(arch_tmp[-2])
            else:
                arch_save.append(arch_tmp[-1])
        assert len(arch_save) == len(self.delta_resolution)
        return arch_save

    def _measure_target_latency(self):
        arch_1step = []
        for op in self.delta_depth:
            arch_1step.append(self._expand_depth(op, 1))
        for op in self.delta_width:
            arch_1step.append(self._expand_width(op, 1))
        for op in self.delta_resolution:
            arch_1step.append(self._expand_resolution(op, 1))
        arch_1step_latency = [a.latency for a in arch_1step]
        return max(arch_1step_latency)

    def expand_all(self):
        print('measure target latency')
        target_latency = self._measure_target_latency()
        print('expand from {} to {}'.format(self.latency, target_latency))
        expanded_arch = []
        expanded_arch += self._expand_depth_to_target(target_latency)
        expanded_arch += self._expand_width_to_target(target_latency)
        expanded_arch += self._expand_resolution_to_target(target_latency)
        # De-duplication
        arch_hash = []
        arch = []
        for a in expanded_arch:
            if a.hash not in arch_hash:
                arch_hash.append(a.hash)
                arch.append(a)
        return arch

    def get_slope(self):
        assert self.parent is not None
        assert self.latency > 0
        assert self.miou > 0
        assert self.parent.latency > 0
        assert self.parent.miou > 0
        return (self.miou - self.parent.miou) / (self.latency - self.parent.miou)

    def update_miou(self):
        # train the network and evaluate the mIoU performance
        print('#' * 40)
        print('#' * 10 + ' For demonstration  ' + '#' * 10)
        print('#' * 10 + '  Use random mIoU   ' + '#' * 10)
        print('#' * 40)
        self.miou = random.random()  # dummy data


def main():
    # start point tiny network
    depth = (1, 1, 1, 1, 1)
    width = (4, 8, 16, 32, 32)
    resolution = (1 / 2, 0, 0)
    arch_init = LPSNetExpansion(depth, width, resolution, parent=None)
    arch_init.update_miou()
    print(arch_init)

    # expanding steps
    best_arch = [arch_init]
    for step in range(14):
        # get base arch from previous expansion
        base_arch = best_arch[-1]
        # expand arch along different dimensions
        arch_expand = base_arch.expand_all()
        # train the networks and get the mIoU
        for arch in arch_expand:
            arch.update_miou()
        # get the best tradeoff from the expanded networks
        slopes = np.asarray([arch.get_slope() for arch in arch_expand])
        best_arch.append(arch_expand[np.argmax(slopes)])
        print('current best arch is: {}'.format(best_arch[-1]))


if __name__ == '__main__':
    main()
