import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from model_admm import Network
from torch.autograd import Variable

import utils
from architect import Architect

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='dataset', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--rho', type=float, default=1e-3, help='admm relative weight')
parser.add_argument('--admm_freq', type=int, default=10, help='admm update frequency')
args = parser.parse_args()

args.save = os.path.join(utils.get_dir(),
                         'exp/admm-{}-{}'.format(os.getenv('SLURM_JOB_ID'), time.strftime("%Y%m%d-%H%M%S")))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('src/*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.rho)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    datapath = os.path.join(utils.get_dir(), args.data)
    train_data = dset.CIFAR10(root=datapath, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    model.initialize_Z_and_U()

    loggers = {"train": {"loss": [], "acc": [], "step": []}, "val": {"loss": [], "acc": [], "step": []},
               "infer": {"loss": [], "acc": [], "step": []}}

    for epoch in range(args.epochs):
        valid_iter = iter(valid_queue)
        model.clear_U()  # TODO: keep?

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(torch.relu(model.alphas_normal).tanh())
        print(torch.relu(model.alphas_reduce).tanh())

        # training
        train_acc, train_obj = train(train_queue, valid_iter, model, architect, criterion, optimizer, lr, loggers)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        utils.log_loss(loggers["infer"], valid_obj, valid_acc, 1)
        logging.info('valid_acc %f', valid_acc)

        utils.plot_loss_acc(loggers, args.save)

        # model.update_history()

        utils.save_file(recoder=model.alphas_normal_history, path=os.path.join(args.save, 'normalalpha'),
                        steps=loggers["train"]["step"])
        utils.save_file(recoder=model.alphas_reduce_history, path=os.path.join(args.save, 'reducealpha'),
                        steps=loggers["train"]["step"])
        utils.save_file(recoder=model.FI_normal_history, path=os.path.join(args.save, 'normalFI'),
                        steps=loggers["train"]["step"])
        utils.save_file(recoder=model.FI_reduce_history, path=os.path.join(args.save, 'reduceFI'),
                        steps=loggers["train"]["step"])

        scaled_FI_normal = scale(model.FI_normal_history, model.alphas_normal_history)
        scaled_FI_reduce = scale(model.FI_reduce_history, model.alphas_reduce_history)
        utils.save_file(recoder=scaled_FI_normal, path=os.path.join(args.save, 'normalFIscaled'),
                        steps=loggers["train"]["step"])
        utils.save_file(recoder=scaled_FI_reduce, path=os.path.join(args.save, 'reduceFIscaled'),
                        steps=loggers["train"]["step"])

        utils.plot_FI(loggers["train"]["step"], model.FI_history, args.save, "FI")

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)


def scale(FI_hist, alpha_hist):
    scaled_FI = {}
    for k in FI_hist.keys():
        scaled_FI[k] = np.divide(np.array(FI_hist[k]), np.array(alpha_hist[k][1:]))
    return scaled_FI


def train(train_queue, valid_iter, model, architect, criterion, optimizer, lr, loggers):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()

    batches = len(train_queue)
    for step, (input, target) in enumerate(train_queue):
        model.train()
        model.tick(1 / batches)
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda(non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue without replacement
        input_search, target_search = next(valid_iter)
        input_search = Variable(input_search, requires_grad=False).cuda(non_blocking=True)
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

        valid_loss = architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        utils.log_loss(loggers["val"], valid_loss, None, model.clock)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.mask_alphas()
        model.track_FI()
        model.update_history()

        prec1 = utils.accuracy(logits, target, topk=(1,))
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        utils.log_loss(loggers["train"], loss.item(), prec1[0].item(), model.clock)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)

        if (step + 1) % args.admm_freq == 0:
            model.update_Z()
            model.update_U()

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda(non_blocking=True)
            target = Variable(target).cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1 = utils.accuracy(logits, target, topk=(1,))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1[0].item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
