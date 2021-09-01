import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torch.autograd import Variable

import utils
from model import NetworkCIFAR as Network
from genotypes import *

# test of retrained discrete architectures

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='dataset', help='location of the data corpus')
parser.add_argument('--task', type=str, default='CIFAR10', help='task name')
parser.add_argument('--test_filter', type=int, default=0,
                    help='CIFAR100cf fine classes to filter per coarse class in test')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
args = parser.parse_args()

args.save = os.path.join(utils.get_dir(), os.path.split(args.model_path)[0], "test")
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.save, 'testlog.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.task == "CIFAR100":
    CIFAR_CLASSES = 100
elif args.task == "CIFAR100cf":
    CIFAR_CLASSES = 20
else:
    CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype_path = os.path.join(utils.get_dir(), os.path.split(args.model_path)[0], 'genotype.txt')
    print(genotype_path)
    if os.path.isfile(genotype_path):
        with open(genotype_path, "r") as f:
            geno_raw = f.read()
            genotype = eval(geno_raw)
    else:
        genoname = os.path.join(utils.get_dir(), os.path.split(args.model_path)[0], 'genoname.txt')
        if os.path.isfile(genoname):
            with open(genoname, "r") as f:
                args.arch = f.read()
            genotype = eval("genotypes.%s" % args.arch)
        else:
            genotype = eval("genotypes.ADMM")
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    utils.load(model, os.path.join(utils.get_dir(), args.model_path))

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    _, test_transform = utils._data_transforms_cifar10(args)
    datapath = os.path.join(utils.get_dir(), args.data)
    test_data = dset.CIFAR10(root=datapath, train=False, download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    if args.task == "CIFAR100cf":
        _, test_transform = utils._data_transforms_cifar100(args)

        test_data = utils.CIFAR100C2F(root=datapath, train=False, download=True, transform=test_transform)

        test_indices = test_data.filter_by_fine(args.test_filter)

        test_queue = torch.utils.data.DataLoader(
            torch.utils.data.Subset(test_data, test_indices), batch_size=args.batch_size,
            shuffle=False, pin_memory=True, num_workers=2)

        # TODO: extend each epoch or multiply number of epochs by 20%*args.class_filter

    else:
        if args.task == "CIFAR100":
            _, test_transform = utils._data_transforms_cifar100(args)
            test_data = dset.CIFAR100(root=datapath, train=False, download=True, transform=test_transform)
        else:
            _, test_transform = utils._data_transforms_cifar10(args)
            test_data = dset.CIFAR10(root=datapath, train=False, download=True, transform=test_transform)

        test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = args.drop_path_prob
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(test_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1 = utils.accuracy(logits, target, topk=(1,))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)

        if step % args.report_freq == 0:
            logging.info('test %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
