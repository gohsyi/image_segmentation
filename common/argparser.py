import argparse

from common import util


parser = argparse.ArgumentParser()

# gpu setting
parser.add_argument('-gpu', type=str, default='-1')

# model selection
parser.add_argument('-model', type=str, default='segnet')

# path setting
parser.add_argument('-note', type=str, default='0')
parser.add_argument('-train_path', type=str, default='data/train.txt')
parser.add_argument('-test_path', type=str, default='data/test.txt')
parser.add_argument('-test', action='store_true', default=False)
parser.add_argument('-finetune', type=str, default=None)
parser.add_argument('-save_image', type=str, default='logs/segmentation/')

# segnet setting
parser.add_argument('-loss', type=str, default='dice')
parser.add_argument('-batch_size', type=int, default=5)
parser.add_argument('-total_epoches', type=int, default=int(1e5))
parser.add_argument('-learning_rate', type=float, default=1e-3)

# dataset setting
parser.add_argument('-image_h', type=int, default=240)
parser.add_argument('-image_w', type=int, default=240)
parser.add_argument('-image_c', type=int, default=3)
parser.add_argument('-num_classes', type=int, default=20)

args = parser.parse_args()


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
