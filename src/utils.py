import os
import shutil

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import json


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x

def get_dir():
    if os.path.exists("/tmpdir/maile/pydnas/"):
        dir = "/tmpdir/maile/pydnas/"
    elif os.path.exists("/projets/reva/kmaile/pydnas/"):
        dir = "/projets/reva/kmaile/pydnas/"
    else:
        dir = ""
    return dir

def create_exp_dir(path, scripts_to_save=None):

    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


blue = plt.get_cmap("Blues")
orange = plt.get_cmap("Oranges")
grey = plt.get_cmap("Greys")
COLORMAP = {
    'none': grey(0.8),
    'max_pool_3x3': orange(0.6),
    'avg_pool_3x3': orange(0.3),
    'skip_connect': grey(0.5),
    'sep_conv_3x3': blue(0.3),
    'sep_conv_5x5': blue(0.5),
    'dil_conv_3x3': blue(0.7),
    'dil_conv_5x5': blue(0.9)
}


def save_file(recoder, path='./'):
    if not os.path.exists(path):
        os.makedirs(path)
    has_none = False
    fig, axs = plt.subplots(4, 5, sharex="col", sharey="row")
    for (k, v) in recoder.items():
        outin = k[k.find("(")+1:k.find(")")].split(", ")
        src = int(outin[1])-2
        dest = int(outin[0])
        if src == -2:
            axs[dest, src + 2].set_ylabel(str(dest))
        if dest == 3:
            axs[dest, src + 2].set_xlabel(str(src))
        op = k.split("op: ")[1]
        axs[dest, src+2].plot(v, label=op, color=COLORMAP[op])
        if "none" in op:
            has_none = True
    for i in range(0, 3):
        for j in range(2+i, 5):
            axs[i, j].axis("off")
    handles, labels = axs[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.savefig(os.path.join(path, 'alphahistory.png'), bbox_inches='tight')
    print('save history weight in {}'.format(os.path.join(path, 'alphahistory.png')))
    plt.close()
    if has_none:
        fig, axs = plt.subplots(4, 5, sharex="col", sharey="row")
        for (k, v) in recoder.items():
            op = k.split("op: ")[1]
            if "none" not in op:
                outin = k[k.find("(") + 1:k.find(")")].split(", ")
                src = int(outin[1]) - 2
                dest = int(outin[0])
                if src == -2:
                    axs[dest, src + 2].set_ylabel(str(dest))
                if dest == 3:
                    axs[dest, src + 2].set_xlabel(str(src))
                axs[dest, src + 2].plot(v, label=op, color=COLORMAP[op])
        for i in range(0, 3):
            for j in range(2 + i, 5):
                axs[i, j].axis("off")
        handles, labels = axs[1, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.savefig(os.path.join(path, 'alphahistory-none.png'), bbox_inches='tight')
        print('save history weight without nones in {}'.format(os.path.join(path, 'alphahistory-none.png')))
        plt.close()

    with open(os.path.join(path, 'history_weight.json'), 'w') as outf:
        json.dump(recoder, outf)
        print('save history weight in {}'.format(os.path.join(path, 'history_weight.json')))


class BathymetryDataset(Dataset):
    def __init__(self, csv_file, root_dir="dataset/bathymetry", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with paths and labels.
            root_dir (string): Directory for relative paths
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_data = pd.read_csv(csv_file)
        self.csv_data[0] = self.csv_data[0].str.replace("/home/ad/alnajam/scratch/pdl/datasets/recorded_angles/", "")
        self.root_dir = os.path.join(get_dir(), root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.csv_data.iloc[idx, 0])

        if not img_path.endswith('.npy'):
            image = np.load(f'{img_path}.npy')
        else:
            image = np.load(img_path)

        depth = self.csv_data.iloc[idx, 1]
        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

#dataset/bathymetry/datasets_guyane_stlouis/