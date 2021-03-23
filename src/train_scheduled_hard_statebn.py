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
from torch.autograd import Variable

import utils
from architect import Architect
from model_admm_hard_statebn import Network

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
parser.add_argument('--save', type=str, default='', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--crb', action='store_true', default=False, help='use CRB activation instead of softmax')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--rho', type=float, default=1e-3, help='admm/prox relative weight')
parser.add_argument('--admm_freq', type=int, default=10, help='admm update frequency (if not dynamically scheduled')
parser.add_argument('--init_alpha_threshold', type=float, default=1.0, help='initial alpha threshold')
parser.add_argument('--threshold_multiplier', type=float, default=1.05, help='threshold multiplier')
parser.add_argument('--schedfreq', type=float, default=1.0, help='w steps per each alpha step')
parser.add_argument('--ewma', type=float, default=1.0, help='weight for exp weighted moving average (1.0 for no ewma)')
parser.add_argument('--zuewma', type=float, default=1.0,
                    help='weight for Z,U exp weighted moving average (1.0 for no ewma)')
parser.add_argument('--dyno_split', action='store_true', default=False,
                    help='use train/val split based on dynamic schedule')
parser.add_argument('--dyno_schedule', action='store_true', default=False, help='use dynamic schedule')
parser.add_argument('--reg', type=str, default='darts', help='reg/opt to use')
args = parser.parse_args()

if len(args.save) == 0:
    args.save = os.path.join(utils.get_dir(),
                             'exp/admmschedhbn-{}-{}'.format(os.getenv('SLURM_JOB_ID'), time.strftime("%Y%m%d-%H%M%S")))
else:
    args.save = os.path.join(utils.get_dir(), 'exp', args.save)
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

    if args.dyno_schedule:
        args.threshold_divider = np.exp(-np.log(args.threshold_multiplier) * args.schedfreq)
        print(args.threshold_divider, -np.log(args.threshold_multiplier) / np.log(args.threshold_divider))
    if args.dyno_split:
        args.train_portion = 1 - 1 / (1 + args.schedfreq)

    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.rho, args.crb, args.epochs,
                    args.ewma, reg=args.reg)
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

    loggers = {"train": {"loss": [], "acc": [], "step": []},
               "val": {"loss": [], "acc": [], "step": []},
               "infer": {"loss": [], "acc": [], "step": []},
               "ath": {"threshold": [], "step": []},
               "astep": [],
               "zustep": []}

    alpha_threshold = args.init_alpha_threshold
    alpha_counter = 0
    ewma = -1

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(model.activate(model.alphas_normal))
        print(model.activate(model.alphas_reduce))

        # training
        train_acc, train_obj, alpha_threshold, alpha_counter, ewma = train(train_queue, valid_queue, model,
                                                                           architect, criterion, optimizer,
                                                                           loggers, alpha_threshold,
                                                                           alpha_counter, ewma, args)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        utils.log_loss(loggers["infer"], valid_obj, valid_acc, model.clock)
        logging.info('valid_acc %f', valid_acc)

        utils.plot_loss_acc(loggers, args.save)

        utils.save_file(recoder=model.alphas_normal_history, path=os.path.join(args.save, 'normalalpha'), steps=loggers["train"]["step"])
        utils.save_file(recoder=model.alphas_reduce_history, path=os.path.join(args.save, 'reducealpha'), steps=loggers["train"]["step"])

        utils.plot_FI(loggers["train"]["step"], model.FI_history, args.save, "FI", loggers["ath"], loggers['astep'])
        utils.plot_FI(loggers["train"]["step"], model.FI_ewma_history, args.save, "FI_ewma", loggers["ath"], loggers['astep'])

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    f = open(os.path.join(args.save, 'genotype.txt'), "w")
    f.write(str(genotype))
    f.close()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, loggers, alpha_threshold, alpha_counter,
          ewma, args):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    valid_iter = iter(valid_queue)
    print("valid len:", len(valid_queue))

    batches = len(train_queue)
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        model.tick(1 / batches)

        valid_loss = 0.0

        loggers["ath"]["threshold"].append(alpha_threshold)
        loggers["ath"]["step"].append(model.clock)
        if (not args.dyno_schedule and step % int(args.schedfreq) == 0) or (
                model.FI_ewma > 0.0 and model.FI_ewma < alpha_threshold):
            print("alpha step")
            try:
                input_search, target_search = next(valid_iter)
            except StopIteration:
                print("reset valid iter")
                valid_iter = iter(valid_queue)
                input_search, target_search = next(valid_iter)
            input_search = Variable(input_search, requires_grad=False).cuda(non_blocking=True)
            target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

            valid_loss = architect.step(input_search, target_search)
            utils.log_loss(loggers["val"], valid_loss, None, model.clock)
            if args.dyno_schedule:
                alpha_threshold *= args.threshold_divider
            alpha_counter += 1
            loggers["astep"].append(model.clock)
        elif args.dyno_schedule:
            alpha_threshold *= args.threshold_multiplier

        input = Variable(input, requires_grad=False).cuda(non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        model.track_FI()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.mask_alphas()

        model.update_history()

        prec1 = utils.accuracy(logits, target, topk=(1,))
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        utils.log_loss(loggers["train"], loss.item(), prec1[0].item(), model.clock)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)

        if (args.reg == "admm") & ((alpha_counter + 1) % args.admm_freq == 0):
            model.update_Z()
            model.update_U()
            loggers["zustep"].append(model.clock)
            alpha_counter = 0

    utils.log_loss(loggers["val"], valid_loss, None, model.clock)
    return top1.avg, objs.avg, alpha_threshold, alpha_counter, ewma


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
