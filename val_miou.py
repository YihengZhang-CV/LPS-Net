import torch
import torch.utils
import torch.nn.functional as F
import torchvision.transforms as ts
import os
import numpy as np
from PIL import Image

from lpsnet import get_lspnet_s


class CityscapesVal(torch.utils.data.Dataset):

    def __init__(self, imgroot, imglist):
        assert os.path.exists(imgroot)
        assert os.path.exists(imglist)
        self.datalist = []
        with open(imglist, 'r') as f:
            lines = f.readlines()
            for l in lines:
                lsp = l.strip().split(' ')
                Pimg = os.path.join(imgroot, lsp[0])
                Plab = os.path.join(imgroot, lsp[1])
                assert os.path.exists(Pimg)
                assert os.path.exists(Plab)
                self.datalist.append((Pimg, Plab))
        print('Total {} images for validation.'.format(len(self.datalist)))
        self.transform_img = ts.Compose([
            ts.ToTensor(),
            ts.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, item):
        image = Image.open(self.datalist[item][0])
        label = Image.open(self.datalist[item][1])
        image = self.transform_img(image)
        label = torch.from_numpy(np.asarray(label).astype(np.int64))
        if torch.sum(torch.logical_and(label >= 19, label != 255)):
            raise RuntimeError('The label images should use trainId provided by official code of Cityscapes.')
        return image, label


@torch.no_grad()
def val_perf(net, data):
    net.eval()
    fea_cha = 19
    pixel_inter = torch.zeros([fea_cha], dtype=torch.double).cuda()
    pixel_union = torch.zeros([fea_cha], dtype=torch.double).cuda()
    for idx, (image, label) in enumerate(data):
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        logit = net(image)
        pred = torch.argmax(F.interpolate(logit.squeeze(), label.shape[-2:], mode='bilinear', align_corners=True), dim=-3)
        imPred = pred.ravel() + 1
        imAnno = label.ravel() + 1
        imAnno[imAnno > fea_cha] = 0
        imPred = imPred * (imAnno > 0)
        imPred = imPred.type(torch.uint8)
        imAnno = imAnno.type(torch.uint8)
        imUnion_anno = torch.bincount(imAnno, minlength=fea_cha + 1)
        cIntersection = torch.bincount(imPred * (imPred == imAnno), minlength=fea_cha + 1)
        imUnion_pred = torch.bincount(imPred, minlength=fea_cha + 1)
        area_union = imUnion_pred + imUnion_anno - cIntersection
        pixel_inter += cIntersection[1:]
        pixel_union += area_union[1:]
    pixel_inter = pixel_inter.cpu().numpy()
    pixel_union = pixel_union.cpu().numpy()
    IoU = pixel_inter / pixel_union
    meanIoU = np.nanmean(IoU)
    return meanIoU


def main():
    torch.cuda.set_device(0)

    # init dataset
    dataset = CityscapesVal('/opt/dataset/cityscapes', 'imagelist_val.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=8, shuffle=False, num_workers=8,
        pin_memory=True, drop_last=False,)

    # init LPS-Net
    net = get_lspnet_s(deploy=True).eval().cuda()
    net.load_state_dict(torch.load('LPS-Net-S.pth', map_location='cpu'))
    mIoU = val_perf(net, dataloader)
    print('LPS-Net-S on Cityscapes validation set: mean IoU={:>0.1f}%'.format(mIoU * 100))


if __name__ == '__main__':
    main()