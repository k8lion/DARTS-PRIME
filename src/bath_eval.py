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
import genotypes
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkBathy as Network

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='dataset', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='BATH', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--depth_normalization', type=float, default=0.1, help='depth normalization factor')
parser.add_argument('--min_energy', type=float, default=0.1, help='minimum energy')
parser.add_argument('--max_energy', type=float, default=4.0, help='maximum energy')
parser.add_argument('--max_depth', type=float, default=40.0, help='maximum unnormalized depth')

args = parser.parse_args()

args.save = os.path.join(utils.get_dir(), 'exp/batheval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S")))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


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

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, 1, args.layers, args.auxiliary, genotype, input_channels=4)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.MSELoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # datapath = os.path.join(utils.get_dir(), args.data)
    # train_data = dset.CIFAR10(root=datapath, train=True, download=True, transform=train_transform)
    # valid_data = dset.CIFAR10(root=datapath, train=False, download=True, transform=valid_transform)
    train_data = utils.BathymetryDataset(args, "../mixed_train.csv")
    valid_data = utils.BathymetryDataset(args, "../mixed_validation.csv")

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    loggers = {"train":{"loss": [], "step": []}, "val":{"loss": [], "step": []}}

    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        _ = train(train_queue, model, criterion, optimizer, loggers["train"])

        infer_loss = infer(valid_queue, model, criterion)
        utils.log_loss(loggers["val"], infer_loss, None, 1)

        utils.plot_loss_acc(loggers, args.save)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer, train_logger):
    objs = utils.AverageMeter()
    model.train()

    batches = len(train_queue)
    for step, (input, target) in enumerate(train_queue):
        input = Variable(input.float()).cuda()
        target = Variable(target.float()).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(torch.squeeze(logits), target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        n = input.size(0)
        objs.update(loss.item(), n)
        utils.log_loss(train_logger, loss.item(), None, 1 / batches)

        if step % args.report_freq == 0:
            logging.info('train %03d %e', step, objs.avg)

    return objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input.float()).cuda()
            target = Variable(target.float()).cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(torch.squeeze(logits), target)

            n = input.size(0)
            objs.update(loss.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e', step, objs.avg)

    return objs.avg


if __name__ == '__main__':
    main()
