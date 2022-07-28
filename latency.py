import torch
import random
import subprocess
import onnx
import os
from onnxsim import simplify
import warnings
warnings.filterwarnings('ignore')

from lpsnet import get_lspnet_s, get_lspnet_m, get_lspnet_l


@torch.no_grad()
def measure_latency_ms(model, data_in, device=0):
    def parse_latency(lines, pattern='[I] mean:'):
        lines = lines[::-1]
        for l in lines:
            if pattern in l:
                l_num = l[l.find(pattern) + len(pattern):l.find('ms')]
                return float(l_num.strip())
        return -1
    try:
        _ = subprocess.run('trtexec', stdout=subprocess.PIPE, check=True)
    except Exception:
        print('can\'t find the trtexec')
    else:
        onnx_file_name = ''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',10)) + '.onnx'
        torch.onnx.export(model, (data_in, ), onnx_file_name, verbose=False, opset_version=11)
        torch.cuda.empty_cache()
        model_simp, check = simplify(onnx.load(onnx_file_name))
        assert check, "Simplified ONNX model could not be validated"
        onnx.save_model(model_simp, onnx_file_name)
        trtcmd = ['trtexec', '--workspace=2048', '--duration=20', '--warmUp=1000',
                  '--onnx=' + onnx_file_name, '--device={}'.format(device)]
        ret = subprocess.run(trtcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        latency = parse_latency(ret.stdout.decode('utf-8').split('\n'))
        os.remove(onnx_file_name)
        return latency


def main():
    # init LPS-Net
    net = get_lspnet_s().eval()
    # eval inference latency with TensorRT
    latency = measure_latency_ms(net, torch.rand((1, 3, 1024, 2048)))
    print('Inference latency is {}ms/image on current device.'.format(latency))


if __name__ == '__main__':
    main()