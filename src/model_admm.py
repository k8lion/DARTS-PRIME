from copy import deepcopy
from torch.autograd import Variable

from genotypes import Genotype
from genotypes import ADMMPRIMITIVES
from operations import *


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in ADMMPRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
        self._bn = nn.BatchNorm2d(C, affine=False)

    def forward(self, x, weights):
        return self._bn(sum(w * op(x) for w, op in zip(weights, self._ops) if w > 0.0))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, rho, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._rho = rho
        self._steps = steps
        self._multiplier = multiplier
        self._reduce = []

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                self._reduce.append(i)
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = torch.relu(self.alphas_reduce).tanh()
            else:
                weights = torch.relu(self.alphas_normal).tanh()
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self.admm_loss(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(ADMMPRIMITIVES)

        self.alphas_normal = Variable((1 / num_ops + 1e-3 * torch.randn(k, num_ops)).cuda(), requires_grad=True)
        self.alphas_reduce = Variable((1 / num_ops + 1e-3 * torch.randn(k, num_ops)).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        self._arch_mask = [
            torch.ones_like(self.alphas_normal),
            torch.ones_like(self.alphas_reduce),
        ]
        self.mask_alphas()

        self.FI_reduce = torch.zeros_like(self.alphas_reduce).cpu()
        self.FI_normal = torch.zeros_like(self.alphas_normal).cpu()
        self.FI = 0.0

        self.alphas_normal_history = {}
        self.alphas_reduce_history = {}
        self.FI_normal_history = {}
        self.FI_reduce_history = {}
        self.FI_history = []
        mm = 0
        last_id = 1
        node_id = 0
        for i in range(k):
            for j in range(num_ops):
                self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), ADMMPRIMITIVES[j])] = []
                self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), ADMMPRIMITIVES[j])] = []
                self.FI_normal_history['edge: {}, op: {}'.format((node_id, mm), ADMMPRIMITIVES[j])] = []
                self.FI_reduce_history['edge: {}, op: {}'.format((node_id, mm), ADMMPRIMITIVES[j])] = []
            if mm == last_id:
                mm = 0
                last_id += 1
                node_id += 1
            else:
                mm += 1
        self.update_history()

    def mask_alphas(self):
        with torch.no_grad():
            for param, mask in zip(self._arch_parameters, self._arch_mask):
                mask[param <= 0.0] = 0.0
                param[param <= 0.0] = 0.0
                param.mul_(mask)

    def arch_parameters(self):
        return self._arch_parameters

    def _parse(self, weights):
        gene = []
        z = torch.zeros_like(weights).cpu()
        wnp = weights.numpy()
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            W = wnp[start:end].copy()
            edges = sorted(range(i + 2),
                           key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[
                    :2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
                gene.append((ADMMPRIMITIVES[k_best], j))
                z[j+start, k_best] = 1.0
            start = end
            n += 1
        return gene, z

    def genotype(self):

        gene_normal, _ = self._parse(torch.tanh(self.alphas_normal).data.cpu())
        gene_reduce, _ = self._parse(torch.tanh(self.alphas_reduce).data.cpu())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def admm_loss(self, output, target):
        loss = self._criterion(output, target)
        for u, x, z in zip(self.U, self._arch_parameters, self.Z):
            print(self._rho / 2 * (torch.tanh(x).cpu() - z + u).norm())
            loss += self._rho / 2 * (torch.tanh(x).cpu() - z + u).norm()
        return loss

    def initialize_Z_and_U(self):
        self.Z = ()
        self.U = ()
        for param in self._arch_parameters:
            self.Z += (param.detach().cpu().clone(),)
            self.U += (torch.zeros_like(param).cpu(),)

    def update_Z(self):
        new_Z = ()
        idx = 0
        for x, u in zip(self._arch_parameters, self.U):
            _, z = self._parse(torch.tanh(x.detach().cpu().clone() + u).data.cpu())
            new_Z += (z,)
            idx += 1
            print(z)
        self.Z = new_Z


    def update_U(self):
        new_U = ()
        for u, x, z in zip(self.U, self._arch_parameters, self.Z):
            new_u = u + x.detach().cpu().clone() - z
            new_U += (new_u,)
            print(new_u)
        self.U = new_U

    def clear_U(self):
        new_U = ()
        for u in self.U:
            new_u = torch.zeros_like(u).cpu()
            new_U += (new_u,)
        self.U = new_U

    def states(self):
        return {
          'alphas_normal': self.alphas_normal,
          'alphas_reduce': self.alphas_reduce,
          'alphas_normal_history': self.alphas_normal_history,
          'alphas_reduce_history': self.alphas_reduce_history,
          'criterion': self._criterion
        }

    def update_history(self):
        mm = 0
        last_id = 1
        node_id = 0
        normal = torch.relu(self.alphas_normal).tanh().data.cpu().numpy()
        reduce = torch.relu(self.alphas_reduce).tanh().data.cpu().numpy()

        k, num_ops = normal.shape
        for i in range(k):
            for j in range(num_ops):
                self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), ADMMPRIMITIVES[j])].append(
                    float(normal[i][j]))
                self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), ADMMPRIMITIVES[j])].append(
                    float(reduce[i][j]))
            if mm == last_id:
                mm = 0
                last_id += 1
                node_id += 1
            else:
                mm += 1

    #TODO document
    #TODO turn into callback
    def track_FI(self):
        self.FI_reduce *= 0.0
        self.FI_normal *= 0.0
        self.FI *= 0.0
        for (n,p) in self.named_parameters():
            self.FI += torch.sum(p.grad.data ** 2).cpu()
            name = n.split(".")
            if name[0] == "cells" and name[3].isdigit() and name[5].isdigit():
                if int(name[1]) in self._reduce:
                    self.FI_reduce[int(name[3]), int(name[5])] += torch.sum(p.grad.data ** 2).cpu() / len(self._reduce)
                else:
                    self.FI_normal[int(name[3]), int(name[5])] += torch.sum(p.grad.data ** 2).cpu() / (self._layers - len(self._reduce))

        self.FI_history.append(float(self.FI))

        mm = 0
        last_id = 1
        node_id = 0
        normal = self.FI_normal.data.cpu().numpy()
        reduce = self.FI_reduce.data.cpu().numpy()

        k, num_ops = normal.shape
        for i in range(k):
            for j in range(num_ops):
                self.FI_normal_history['edge: {}, op: {}'.format((node_id, mm), ADMMPRIMITIVES[j])].append(
                    float(normal[i][j]))
                self.FI_reduce_history['edge: {}, op: {}'.format((node_id, mm), ADMMPRIMITIVES[j])].append(
                    float(reduce[i][j]))
            if mm == last_id:
                mm = 0
                last_id += 1
                node_id += 1
            else:
                mm += 1