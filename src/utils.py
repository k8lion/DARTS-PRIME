import os
import shutil
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision.datasets import CIFAR10, CIFAR100
from typing import Iterator, Sequence
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.0
import json
from sklearn.utils import resample
from math import ceil

class AverageMeter(object):

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

NODE_LABELS = {
    "-2": "c_{k-2}",
    "-1": "c_{k-1}",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
}


def log_loss(logger, loss, acc, step):
    logger["loss"].append(loss)
    if acc is not None:
        logger["acc"].append(acc)
    logger["step"].append(step)


def plot_loss_acc(loggers, path):
    if not os.path.exists(path):
        os.makedirs(path)
    if "infer" in loggers.keys():
        infer = "infer"
    else:
        infer = "val"
    if "acc" in loggers["train"].keys():
        infer_stat = "acc"
        infer_legend = "accuracy"
    else:
        infer_stat = "loss"
        infer_legend = "loss"
    fig, axs = plt.subplots(2, sharex="col")
    axs[0].plot(loggers["train"]["step"], loggers["train"]["loss"], label="training CE loss (w)")
    axs[0].plot(loggers["val"]["step"], loggers["val"]["loss"], label="val CE loss (alpha)")
    axs[0].legend()
    axs[1].plot(loggers[infer]["step"], loggers[infer][infer_stat], label="infer "+infer_legend)
    axs[1].legend()
    fig.savefig(os.path.join(path, 'loss.png'), bbox_inches='tight')
    plt.close()
    fig, axs = plt.subplots(3, sharex="col")
    axs[0].plot(loggers["train"]["step"], loggers["train"]["loss"], label="training CE loss (w)")
    axs[1].plot(loggers["val"]["step"], loggers["val"]["loss"], label="val CE loss (alpha)", color="tab:orange")
    axs[0].legend()
    axs[1].legend()
    axs[2].plot(loggers[infer]["step"], loggers[infer][infer_stat], label="infer "+infer_legend)
    axs[2].legend()
    fig.savefig(os.path.join(path, 'loss_subplots.png'), bbox_inches='tight')
    plt.close()

def plot_FI(steps, FI_history, path, name, thresh_log = [], step_log = None):
    for scale in ["log", "linear"]:
        for xlims in [[0, 1], [0, 50], [49, 50]]:
            orig_xmax = xlims[1]
            if max(steps) < xlims[0]:
                continue
            if max(steps) < xlims[1]:
                xlims[1] = max(steps)
            elif max(steps)-1 > xlims[1]:
                continue
            if xlims[0] == 0 and orig_xmax == 50:
                fig, axs = plt.subplots(2, sharex="col")
                axs[0].set_yscale(scale)
                axs[0].set_xlim(xlims)
                if step_log is not None:
                    axs[1].hist(step_log, range(ceil(max(steps))+1), label="Number of steps")
                axs[0].plot(steps, FI_history, label="Fisher Information Trace")
                if len(thresh_log) > 0:
                    axs[0].plot(thresh_log["step"], thresh_log["threshold"], label="Threshold", alpha=0.5)
                axs[0].legend()
                axs[1].legend()
            else:
                fig, axs = plt.subplots(1)
                axs.set_yscale(scale)
                axs.set_xlim(xlims)
                if step_log is not None:
                    if xlims[1]-xlims[0] <= 1:
                        axs.vlines(step_log, axs.get_ylim()[0], min(FI_history+thresh_log["threshold"]), label="Step", color="k")
                    else:
                        axs.hist(step_log, range(ceil(max(steps)+1)), label="Number of steps")
                axs.plot(steps, FI_history, label="Fisher Information Trace")
                if len(thresh_log) > 0:
                    axs.plot(thresh_log["step"], thresh_log["threshold"], label="Threshold", alpha=0.5)
                axs.legend()
            fig.savefig(os.path.join(path, name+'_history_'+scale+'_'+str(xlims[0])+'-'+str(orig_xmax)+'.png'), bbox_inches='tight')
            plt.close()
    try:
        with open(os.path.join(path, name+'_history.json'), 'w') as outf:
            json.dump(FI_history, outf)
        if thresh_log is not None:
            with open(os.path.join(path, name + '_threshold_history.json'), 'w') as outf:
                json.dump(thresh_log, outf)
    except:
        pass

def save_file(recoder, path='./', steps=None):
    has_none = False
    fig, axs = plt.subplots(4, 5, sharex="col", sharey="row")
    if steps is None:
        k, v  = list(recoder.items())[0]
        steps = range(len(v))
    for (k, v) in recoder.items():
        outin = k[k.find("(")+1:k.find(")")].split(", ")
        src = int(outin[1])-2
        dest = int(outin[0])
        if src == -2:
            axs[dest, src + 2].set_ylabel(NODE_LABELS[str(dest)])
        if dest == 3:
            axs[dest, src + 2].set_xlabel(NODE_LABELS[str(src)])
        op = k.split("op: ")[1]
        try:
            axs[dest, src + 2].plot(steps, v, label=op, color=COLORMAP[op])
        except:
            axs[dest, src + 2].plot([0] + steps, v, label=op, color=COLORMAP[op])
        if "none" in op:
            has_none = True
    for i in range(0, 3):
        for j in range(2 + i, 5):
            axs[i, j].axis("off")
    handles, labels = axs[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.62, 0.52))
    fig.suptitle(os.path.split(path)[1][0:6])
    fig.savefig(path + '_history.png', bbox_inches='tight')
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
                    axs[dest, src + 2].set_ylabel(NODE_LABELS[str(dest)])
                if dest == 3:
                    axs[dest, src + 2].set_xlabel(NODE_LABELS[str(src)])
                try:
                    axs[dest, src + 2].plot(steps, v, label=op, color=COLORMAP[op])
                except:
                    axs[dest, src + 2].plot([0] + steps, v, label=op, color=COLORMAP[op])
        for i in range(0, 3):
            for j in range(2 + i, 5):
                axs[i, j].axis("off")
        handles, labels = axs[1, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.62, 0.52))
        fig.suptitle(os.path.split(path)[1][0:6])
        fig.savefig(path + '_alphahistory-none.png', bbox_inches='tight')
        plt.close()
    try:
        with open(path + '_history_weight.json', 'w') as outf:
            json.dump(recoder, outf)
    except:
        pass

cifar100c2f = ('beaver', 'dolphin', 'otter', 'seal', 'whale',
       'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
       'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
       'bottle', 'bowl', 'can', 'cup', 'plate',
       'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
       'clock', 'keyboard', 'lamp', 'telephone', 'television',
       'bed', 'chair', 'couch', 'table', 'wardrobe',
       'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
       'bear', 'leopard', 'lion', 'tiger', 'wolf',
       'bridge', 'castle', 'house', 'road', 'skyscraper',
       'cloud', 'forest', 'mountain', 'plain', 'sea',
       'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
       'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
       'crab', 'lobster', 'snail', 'spider', 'worm',
       'baby', 'boy', 'girl', 'man', 'woman',
       'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
       'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
       'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
       'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
       'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor')


class CIFAR100C2F(CIFAR100):
    meta = {
        'filename': 'meta',
        'fine_key': 'fine_label_names',
        'coarse_key': 'coarse_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
    ):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self._load_meta()

        self.data = []
        self.fine_targets = []
        self.coarse_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.fine_targets.extend(entry['fine_labels'])
                self.coarse_targets.extend(entry['coarse_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC



    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.fine_classes = data[self.meta['fine_key']]
            self.coarse_classes = data[self.meta['coarse_key']]
        self.fine_class_to_idx = {_class: i for i, _class in enumerate(self.fine_classes)}
        print(self.fine_class_to_idx)
        self.coarse_class_to_idx = {_class: i for i, _class in enumerate(self.coarse_classes)}

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, fine_target, coarse_target = self.data[index], self.fine_targets[index], self.coarse_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            fine_target = self.target_transform(fine_target)
            coarse_target = self.target_transform(coarse_target)

        return img, coarse_target

    # return indices such that a certain number of fine classes per coarse class are not included
    def filter_by_fine(self, classes_to_filter=0, init_indices=[]):
        cind2find = [self.fine_class_to_idx[fineclass] for fineclass in cifar100c2f]
        if len(init_indices) == 0:
            indices = [i for i in range(len(self.fine_targets)) if
                       cind2find.index(self.fine_targets[i]) % 5 + 1 > classes_to_filter]
        else:
            indices = [i for i in init_indices if
                       cind2find.index(self.fine_targets[i]) % 5 + 1 > classes_to_filter]
        return indices


class FillingSubsetRandomSampler(SubsetRandomSampler):
    """Samples elements randomly from a given list of indices with necessary replacement until
    total is reached, guaranteeing all indices are sampled n times before any are repeated more

    Args:
        indices (sequence): a sequence of indices
        total (int): total number of samples to take
        reshuffle (bool): whether to reshuffle again
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], total: int, reshuffle: bool = False, generator=None) -> None:
        self.indices = indices
        if total:
            self.total = total
        else:
            self.total = len(indices)
        self.reshuffle = reshuffle
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        iter_list = torch.randperm(len(self.indices), generator=self.generator)
        while len(iter_list) < self.total:
            iter_list = torch.cat((iter_list, torch.randperm(len(self.indices), generator=self.generator)))
            if len(iter_list) > self.total:
                iter_list = iter_list[:self.total]
        if self.reshuffle:
            iter_list = [iter_list[i] for i in torch.randperm(len(iter_list), generator=self.generator)]
        print(len(iter_list))
        return (self.indices[i] for i in iter_list)

    def __len__(self) -> int:
        return len(self.indices)


class BathymetryDataset(Dataset):
    def __init__(self, args, csv_file, root_dir="dataset/bathymetry/datasets_guyane_stlouis",
                 to_trim="/home/ad/alnajam/scratch/pdl/datasets/recorded_angles/", transform=None, to_filter=True):
        """
        Args:
            csv_file (string): Path to the csv file with paths and labels.
            root_dir (string): Directory for relative paths
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = os.path.join(get_dir(), root_dir)
        self.csv_data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        try:
            self.csv_data["Unnamed: 0"] = self.csv_data["Unnamed: 0"].str.replace(to_trim, "")
        except:
            self.csv_data.iloc[:, 0] = self.csv_data.iloc[:, 0].str.replace(to_trim, "")
        if to_filter:
            self.csv_data = self.csv_data[(self.csv_data["max_energy"] >= args.min_energy) & (self.csv_data["max_energy"] <= args.max_energy) & (self.csv_data["z"] <= args.max_depth)]
        self.transform = transform
        self.depth_norm_factor = args.depth_normalization
        if "mixed" in csv_file:
            self.lengths = [len(self.csv_data[self.csv_data.iloc[:, 0].str.contains(source)]) for source in ["guyane", "saint_louis"]]
        else:
            self.lengths = [len(self.csv_data)]
        self.to_filter = to_filter

    def add(self, args, csv_file, to_trim="/home/ad/alnajam/scratch/pdl/datasets/recorded_angles/", to_balance=True):
        new_csv_data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        try:
            new_csv_data["Unnamed: 0"] = new_csv_data["Unnamed: 0"].str.replace(to_trim, "")
        except:
            new_csv_data.iloc[:, 0] = new_csv_data.iloc[:, 0].str.replace(to_trim, "")
        if self.to_filter:
            new_csv_data = new_csv_data[(new_csv_data["max_energy"] >= args.min_energy) & (new_csv_data["max_energy"] <= args.max_energy) & (new_csv_data["z"] <= args.max_depth)]
        self.csv_data = self.csv_data.append(new_csv_data)
        if "mixed" in csv_file:
            self.lengths.extend([len(new_csv_data[new_csv_data.iloc[:, 0].str.contains(source)]) for source in ["guyane", "saint_louis"]])
        else:
            self.lengths.append(len(new_csv_data))
        if to_balance:
            self.rebalance(args.seed)

    def rebalance(self, seed):
        max_length = max(self.lengths)
        last_length = 0
        new_lengths = []
        new_data = []
        for length in self.lengths:
            csv_portion = self.csv_data[last_length:last_length+length]
            if max_length-length > 0:
                upsampled = resample(csv_portion, 
                                     replace=True,     
                                     n_samples=max_length-length,
                                     random_state=seed)
                csv_portion = csv_portion.append(upsampled)
            new_lengths.append(len(csv_portion))
            new_data.append(csv_portion)
            last_length += length
        self.lengths = new_lengths
        self.csv_data = pd.concat(new_data)
        
    def get_subset_indices(self, split_ratio):
        trains = []
        vals = []
        last_length = 0
        for length in self.lengths:
            indices = list(range(last_length, last_length+length))
            split = int(np.floor(split_ratio*length))
            trains.extend(indices[:split])
            vals.extend(indices[split:])
            last_length += length
        return trains, vals

    def write_results(self, targets, preds, path):
        self.csv_data["Target"] = targets
        self.csv_data["Predicted"] = preds
        self.csv_data.to_csv(path)

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
        image = np.transpose(image, (2, 1, 0))

        depth = self.csv_data.iloc[idx, 1]*self.depth_norm_factor
        sample = (image, depth)

        if self.transform:
            sample = self.transform(sample)

        return sample

