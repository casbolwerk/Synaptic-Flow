import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import uuid
import pathlib
import argparse

from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(1)
    device = load.device(0)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset)
    test_loader = load.dataloader(args.dataset, 256, False, 4)

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    model1 = load.model(args.model, args.model_class)(input_shape,
                                                     num_classes,
                                                     False,
                                                     True,
                                                     args.model_path1).to(device)
    model1.cuda(0)
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    model2 = load.model(args.model, args.model_class)(input_shape,
                                                     num_classes,
                                                     False,
                                                     True,
                                                     args.model_path2).to(device)
    model2.cuda(0)

    loss = nn.CrossEntropyLoss()

    analysis = linear_interpolation(model1, model2, test_loader, loss, device)
    print(f'\r[instability analysis] Analysis returned an instability of {analysis*100}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Compression')

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist','cifar10','cifar100','tiny-imagenet','imagenet'],
                        help='dataset (default: mnist)')
    parser.add_argument('--model', type=str, default='fc', choices=['fc','conv',
                        'vgg11','vgg11-bn','vgg13','vgg13-bn','vgg16','vgg16-bn','vgg19','vgg19-bn',
                        'resnet18','resnet20','resnet20path','resnet32','resnet34','resnet44','resnet50',
                        'resnet56','resnet101','resnet110','resnet110','resnet152','resnet1202',
                        'wide-resnet18','wide-resnet20','wide-resnet32','wide-resnet34','wide-resnet44','wide-resnet50',
                        'wide-resnet56','wide-resnet101','wide-resnet110','wide-resnet110','wide-resnet152','wide-resnet1202'],
                        help='model architecture (default: fc)')
    parser.add_argument('--model-class', type=str, default='default', choices=['default','lottery','tinyimagenet','imagenet'],
                        help='model class (default: default)')
    parser.add_argument('--model-path1', type=str, default='',
                        help='dataset (default: mnist)')
    parser.add_argument('--model-path2', type=str, default='',
                        help='dataset (default: mnist)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd','momentum','adam','rms'],
                        help='optimizer (default: adam)')
    args = parser.parse_args()

    run(args)


