import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
from genotypes import *
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network

# evaluation of discretized networks

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='dataset', help='location of the data corpus')
parser.add_argument('--task', type=str, default='CIFAR10', help='task name')
parser.add_argument('--eval_filter', type=int, default=0, help='CIFAR100cf fine classes to filter per coarse class in eval')
parser.add_argument('--test_filter', type=int, default=0, help='CIFAR100cf fine classes to filter per coarse class in test')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--genotype_path', type=str, default='', help='path of search trial')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--test', action='store_true', default=False, help='automatically run on test split')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

if len(args.save) == 0:
    args.save = 'eval-{}-{}'.format(os.getenv('SLURM_JOB_ID'), time.strftime("%Y%m%d-%H%M%S"))

if args.genotype_path is not None:
    if "exp" not in args.genotype_path:
        args.genotype_path = os.path.join('exp', args.genotype_path)
    args.save = os.path.join(utils.get_dir(), args.genotype_path, args.save)
else:
    args.genotype_path = os.path.join('exp', args.genotype_path)
    args.save = os.path.join(utils.get_dir(), args.save)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
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
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype_path = os.path.join(utils.get_dir(), args.genotype_path, 'genotype.txt')
    if os.path.isfile(genotype_path):
        with open(genotype_path, "r") as f:
            geno_raw = f.read()
            genotype = eval(geno_raw)
    else:
        genotype = eval("genotypes.%s" % args.arch)

    f = open(os.path.join(args.save, 'genotype.txt'), "w")
    f.write(str(genotype))
    f.close()

    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    datapath = os.path.join(utils.get_dir(), args.data)

    if args.task == "CIFAR100cf":
        train_transform, valid_transform = utils._data_transforms_cifar100(args)

        train_data = utils.CIFAR100C2F(root=datapath, train=True, download=True, transform=train_transform)
        valid_data = utils.CIFAR100C2F(root=datapath, train=False, download=True, transform=valid_transform)

        train_indices = train_data.filter_by_fine(args.eval_filter)
        valid_indices = valid_data.filter_by_fine(args.eval_filter)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
            pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(
            torch.utils.data.Subset(valid_data, valid_indices), batch_size=args.batch_size,
            shuffle=False, pin_memory=True, num_workers=2)

        #TODO: extend each epoch or multiply number of epochs by 20%*args.class_filter

    else:
        if args.task == "CIFAR100":
            train_transform, valid_transform = utils._data_transforms_cifar100(args)
            train_data = dset.CIFAR100(root=datapath, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR100(root=datapath, train=False, download=True, transform=valid_transform)
        else:
            train_transform, valid_transform = utils._data_transforms_cifar10(args)
            train_data = dset.CIFAR10(root=datapath, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR10(root=datapath, train=False, download=True, transform=valid_transform)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    if args.test:
        torch.cuda.empty_cache()
        os.system(
            'python src/test.py --batch_size 8 --auxiliary --model_path %s --task %s --test_filter %s' %
            (os.path.join(args.save, 'weights.pt'), args.task, args.test_filter))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1 = utils.accuracy(logits, target, topk=(1, ))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1 = utils.accuracy(logits, target, topk=(1, ))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1[0].item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

