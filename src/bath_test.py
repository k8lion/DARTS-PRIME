import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from torch.autograd import Variable

import utils
from model import NetworkBathy as Network

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='BATH', help='which architecture to use')
parser.add_argument('--min_energy', type=float, default=0.1, help='minimum energy')
parser.add_argument('--max_energy', type=float, default=4.0, help='maximum energy')
parser.add_argument('--max_depth', type=float, default=40.0, help='maximum unnormalized depth')
parser.add_argument('--depth_normalization', type=float, default=0.1, help='depth normalization factor')

args = parser.parse_args()

args.save = os.path.join(utils.get_dir(), args.model_path[:-3])
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('src/*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'testlog.txt'))
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

    genotype_path = os.path.join(utils.get_dir(), os.path.split(args.model_path)[0], 'genotype.txt')
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
            genotype = eval("genotypes.BATH")
    model = Network(args.init_channels, 1, args.layers, args.auxiliary, genotype, input_channels=4)
    model = model.cuda()
    print(os.path.join(utils.get_dir(), args.model_path))
    utils.load(model, os.path.join(utils.get_dir(), args.model_path))

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.MSELoss()
    criterion = criterion.cuda()

    test_data_tne = utils.BathymetryDataset(args, "../29TNE.csv", root_dir="dataset/bathymetry/29TNE/dataset_29TNE",
                                            to_trim="/tmp/pbs.6233542.admin01/tmp_portugal/", to_filter=False)

    test_queue_tne = torch.utils.data.DataLoader(
        test_data_tne, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = args.drop_path_prob
    test_obj, targets, preds = infer(test_queue_tne, model, criterion, args.depth_normalization)
    logging.info('test_obj tne %f', test_obj)

    test_data_tne.write_results(targets, preds, os.path.join(args.save, 'tne_results.csv'))

    test_data_smd = utils.BathymetryDataset(args, "../29SMD.csv", root_dir="dataset/bathymetry/29SMD/dataset_29SMD",
                                            to_trim="/tmp/pbs.6233565.admin01/tmp_portugal/", to_filter=False)

    test_queue_smd = torch.utils.data.DataLoader(
        test_data_smd, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    test_obj, targets, preds = infer(test_queue_smd, model, criterion, args.depth_normalization)
    logging.info('test_obj smd %f', test_obj)

    test_data_smd.write_results(targets, preds, os.path.join(args.save, 'smd_results.csv'))


def infer(test_queue, model, criterion, depth_norm):
    objs = utils.AverageMeter()
    objs_ = utils.AverageMeter()
    model.eval()
    targets = []
    predicteds = []

    for step, (input, target) in enumerate(test_queue):
        input = Variable(input.float()).cuda()
        target = Variable(target.float()).cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(torch.squeeze(logits), target)
        loss_ = criterion(torch.squeeze(logits) * 10, target * 10)
        predicteds.extend(torch.squeeze(logits / depth_norm).cpu().tolist())
        targets.extend((target / depth_norm).cpu().tolist())

        n = input.size(0)
        objs.update(loss.item(), n)
        objs_.update(loss_.item(), n)

        if step % args.report_freq == 0:
            logging.info('test %03d %e %e', step, objs.avg, objs_.avg)

    return objs.avg, targets, predicteds


if __name__ == '__main__':
    main()
