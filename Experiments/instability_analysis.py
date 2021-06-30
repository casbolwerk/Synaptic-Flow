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
    torch.cuda.manual_seed(1)
    device = load.device(0)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset)
    prune_loader = load.dataloader(args.dataset, 256, True, 4, 10 * num_classes)
    train_loader = load.dataloader(args.dataset, 128, True, 4)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    train_loader2 = load.dataloader(args.dataset, 128, True, 4)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    test_loader = load.dataloader(args.dataset, 256, False, 4)

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    model = load.model(args.model, args.model_class)(input_shape,
                                                     num_classes,
                                                     False,
                                                     True,
                                                     args.model_path1).to(device)
    model.cuda(0)
    model.to(device)

    loss = nn.CrossEntropyLoss()

    ## Pre-Train ##
    # print('Pre-Train for {} epochs.'.format(args.total_epochs))
    # pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader,
    #                              test_loader, device, args.total_epochs, True)

    ## Prune ##
    print('Pruning with {} for {} epochs.'.format('synflow', 100))
    pruner = load.pruner('synflow')(generator.masked_parameters(model, False, False, False))
    remaining_params, total_params = 0, 0
    for mask, _ in generator.masked_parameters(model, False, False, False):
        remaining_params += mask.detach().cpu().numpy().sum()
        total_params += mask.numel()
    sparsity = 10**(-float(1.0))
    prune_loop(model, loss, pruner, prune_loader, device, sparsity,
               'exponential', 'global', 100, False, False, False, False)

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    model2 = load.model(args.model, args.model_class)(input_shape,
                                                     num_classes,
                                                     False,
                                                     True,
                                                     args.model_path1).to(device)
    model2.cuda(0)
    model2.to(device)

    ## Prune ##
    print('Pruning with {} for {} epochs.'.format('synflow', 100))
    pruner = load.pruner('synflow')(generator.masked_parameters(model2, False, False, False))
    sparsity = 10**(-float(1.0))
    prune_loop(model2, loss, pruner, prune_loader, device, sparsity,
               'exponential', 'global', 100, False, False, False, False)

    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    insta_lr = np.random.normal(args.lr, args.lr/10)
    print(insta_lr)
    insta_lr = max(0.001, insta_lr)
    insta_wd = np.random.normal(args.weight_decay, args.weight_decay/10)
    print(insta_wd)
    insta_wd = max(0.0001, insta_wd)
    insta_optimizer = opt_class(generator.parameters(model2), lr=insta_lr, weight_decay=insta_wd, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)
    insta_scheduler = torch.optim.lr_scheduler.MultiStepLR(insta_optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    model1, model2 = instability_analysis(model, model2, loss, optimizer, insta_optimizer, scheduler, insta_scheduler, train_loader, train_loader2, test_loader, device, args.current_epoch, args.total_epochs, True)

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
    parser.add_argument('--model-path1', type=str, required=True, default='',
                        help='dataset (default: mnist)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd','momentum','adam','rms'],
                        help='optimizer (default: adam)')
    parser.add_argument('--current-epoch', type=int, default='0',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--total-epochs', type=int, default='160',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='learning rate (default: 0.0)')
    parser.add_argument('--lr-drops', type=int, nargs='*', default=[60, 120],
                        help='list of learning rate drops (default: [])')
    parser.add_argument('--lr-drop-rate', type=float, default=0.2,
                        help='multiplicative factor of learning rate drop (default: 0.1)')
    args = parser.parse_args()

    run(args)


