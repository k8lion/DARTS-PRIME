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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
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
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--depth_normalization', type=float, default=0.1, help='depth normalization factor')
parser.add_argument('--min_energy', type=float, default=0.1, help='minimum energy')
parser.add_argument('--max_energy', type=float, default=4.0, help='maximum energy')
parser.add_argument('--max_depth', type=float, default=40.0, help='maximum unnormalized depth')
args = parser.parse_args()


args.save = os.path.join(utils.get_dir(), 'exp/bath-{}-{}'.format(os.getenv('SLURM_JOB_NAME'), time.strftime("%Y%m%d-%H%M%S")))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('src/*.py'))

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

    criterion = nn.MSELoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, 1, args.layers, criterion, input_channels=4)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # datapath = os.path.join(utils.get_dir(), args.data)
    # train_data = dset.CIFAR10(root=datapath, train=True, download=True, transform=train_transform)
    # guyane = utils.BathymetryDataset("guyane/guyane.csv")
    # stl = utils.BathymetryDataset("saint_louis/saint_louis.csv")
    dataset = utils.BathymetryDataset(args, "guyane/guyane.csv")
    dataset.add(args, "saint_louis/saint_louis.csv")

    # num_train = len(guyane)
    # indices = list(range(num_train))
    # split = int(np.floor(args.train_portion * num_train))
    trains, vals = dataset.get_subset_indices(args.train_portion)

    train_queue = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(trains),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(vals),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    loggers = {"train":{"loss": [], "step": []}, "val":{"loss": [], "step": []}, "infer":{"loss": [], "step": []}}

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        _ = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, loggers)

        # validation
        infer_loss = infer(valid_queue, model, criterion)
        utils.log_loss(loggers["infer"], infer_loss, None, 1)

        utils.plot_loss_acc(loggers, args.save)

        model.update_history()

        utils.save_file(recoder=model.alphas_normal_history, path=os.path.join(args.save, 'normal'))
        utils.save_file(recoder=model.alphas_reduce_history, path=os.path.join(args.save, 'reduce'))

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    np.save(os.path.join(os.path.join(args.save, 'normal_weight.npy')),
            F.softmax(model.alphas_normal, dim=-1).data.cpu().numpy())
    np.save(os.path.join(os.path.join(args.save, 'reduce_weight.npy')),
            F.softmax(model.alphas_reduce, dim=-1).data.cpu().numpy())

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, loggers):
    objs = utils.AverageMeter()

    valid_iter = iter(valid_queue)
    batches = len(train_queue)
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input.float(), requires_grad=False).cuda(non_blocking=True)
        target = Variable(target.float(), requires_grad=False).cuda(non_blocking=True)

        input_search, target_search = next(valid_iter)
        input_search = Variable(input_search.float(), requires_grad=False).cuda(non_blocking=True)
        target_search = Variable(target_search.float(), requires_grad=False).cuda(non_blocking=True)

        valid_loss = architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        utils.log_loss(loggers["val"], valid_loss.item(), None, 1 / batches)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        objs.update(loss.item(), n)
        utils.log_loss(loggers["train"], loss.item(), None, 1 / batches)

        if step % args.report_freq == 0:
            logging.info('train %03d %e', step, objs.avg)

    return objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input.float()).cuda(non_blocking=True)
            target = Variable(target.float()).cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            n = input.size(0)
            objs.update(loss.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e', step, objs.avg)

    return objs.avg


if __name__ == '__main__':
    main()
