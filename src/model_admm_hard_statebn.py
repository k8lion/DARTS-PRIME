import torch.nn.functional as F
from torch.autograd import Variable

from genotypes import Genotype
from genotypes import CRBPRIMITIVES, PRIMITIVES
from operations import *
import math


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in CRBPRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops) if w > 0.0)


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
            self._bns.append(nn.BatchNorm2d(C, affine=False))
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
            s = self._bns[i](sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states)))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, rho, crb, epochs, ewma=1.0, zuewma=1.0, reg="admm", steps=4,
                 multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._reg = reg
        self._rho = rho
        self._ewma = ewma
        self._zuewma = zuewma
        self._steps = steps
        self._multiplier = multiplier
        self._reduce = []
        self._crb = crb
        if self._crb:
            self.primitives = CRBPRIMITIVES
        else:
            self.primitives = PRIMITIVES
        self._num_ops = len(self.primitives)
        self.clock = 0.0
        self.total_epochs = epochs

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
        print(C_prev)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

        if self._reg == "admm":
            self.initialize_Z_and_U()

    def activate(self, alphas):
        if self._crb:
            return torch.clamp(alphas, min=0.0, max=1.0)
        else:
            return F.softmax(alphas, dim=-1)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.activate(self.alphas_reduce)
            else:
                weights = self.activate(self.alphas_normal)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def tick(self, step):
        self.clock += step

    def _loss(self, input, target):
        logits = self(input)
        if self._reg == "admm":
            return self.admm_loss(logits, target)
        elif self._reg == "darts":
            return self.darts_loss(logits, target)
        elif self._reg == "prox":
            return self.prox_loss(logits, target)
        elif self._reg == "proxadj":
            return self.proxadj_loss(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))

        if self._crb:
            self.alphas_normal = Variable((1 / self._num_ops + 1e-4 * torch.randn(k, self._num_ops)).cuda(),
                                          requires_grad=True)
            self.alphas_reduce = Variable((1 / self._num_ops + 1e-4 * torch.randn(k, self._num_ops)).cuda(),
                                          requires_grad=True)
        else:
            self.alphas_normal = Variable(1e-3 * torch.randn(k, self._num_ops).cuda(), requires_grad=True)
            self.alphas_reduce = Variable(1e-3 * torch.randn(k, self._num_ops).cuda(), requires_grad=True)
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
        self.FI_ewma = -1.0
        self.FI_alpha = 0.0

        self.alphas_normal_history = {}
        self.alphas_reduce_history = {}
        self.FI_history = []
        self.FI_ewma_history = []
        self.FI_alpha_history = []
        self.FI_alpha_history_step = []
        mm = 0
        last_id = 1
        node_id = 0
        for i in range(k):
            for j in range(self._num_ops):
                self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), self.primitives[j])] = []
                self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), self.primitives[j])] = []
            if mm == last_id:
                mm = 0
                last_id += 1
                node_id += 1
            else:
                mm += 1
        self.update_history()

    def mask_alphas(self):
        if self._crb:
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
                gene.append((self.primitives[k_best], j))
                z[j+start, k_best] = 1.0
            start = end
            n += 1
        return gene, z

    def genotype(self):

        gene_normal, _ = self._parse(self.activate(self.alphas_normal).data.cpu())
        gene_reduce, _ = self._parse(self.activate(self.alphas_reduce).data.cpu())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def admm_loss(self, output, target):
        loss = self._criterion(output, target)
        for u, x, z, m in zip(self.U, self._arch_parameters, self.Z, self._arch_mask):
            loss += self._rho / 2 * (
                (self.activate(x) - z.cuda() + u.cuda()).mul(m)).norm()
        return loss

    def darts_loss(self, output, target):
        return self._criterion(output, target)

    def prox_loss(self, output, target):
        loss = self._criterion(output, target)
        for x in self._arch_parameters:
            _, disc = self._parse(self.activate(x).detach().cpu().clone())
            prox_reg = self.clock / 50. * self._rho / 2 * ((self.activate(x) - disc.cuda()).norm())
            loss += prox_reg
            print(prox_reg)
        return loss
    
    def proxadj_loss(self, output, target):
        loss = self._criterion(output, target)
        for x in self._arch_parameters:
            if torch.isnan(x).any():
                print(x)
            clamped_x = self.activate(x)
            _, disc = self._parse(clamped_x.detach().cpu().clone())
            prox_reg = clamped_x - disc.cuda()
            adj_reg = 4 * torch.pow(
                (torch.pow(torch.clamp(clamped_x, min=1e-4, max=1.0), math.log(2) / math.log(self._num_ops)) - 1 / 2),
                2)
            prog = self.clock / self.total_epochs
            loss += self._rho / 2 * (prox_reg * ((1 - prog) * adj_reg + prog)).norm()
            print(self._rho / 2 * (prox_reg * ((1 - prog) * adj_reg + prog)).norm())
            if torch.isnan(self._rho / 2 * (prox_reg * ((1 - prog) * adj_reg + prog)).norm()).any():
                print(prox_reg, adj_reg)
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
            _, z = self._parse(self.activate(x.detach().cpu().clone() + u).data.cpu())
            new_Z += (z,)
            idx += 1
            print(z)
        self.Z = new_Z


    def update_U(self):
        new_U = ()
        for u, x, z, m in zip(self.U, self._arch_parameters, self.Z, self._arch_mask):
            new_u = ((self._zuewma)*u.cuda() + (x - z.cuda())).mul(m).detach().cpu()
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
        normal = self.activate(self.alphas_normal).data.cpu().numpy()
        reduce = self.activate(self.alphas_reduce).data.cpu().numpy()

        k, _ = normal.shape
        for i in range(k):
            for j in range(self._num_ops):
                self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), self.primitives[j])].append(
                    float(normal[i][j]))
                self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), self.primitives[j])].append(
                    float(reduce[i][j]))
            if mm == last_id:
                mm = 0
                last_id += 1
                node_id += 1
            else:
                mm += 1

    def track_FI(self, alpha_step = False):
        self.FI_reduce *= 0.0
        self.FI_normal *= 0.0
        self.FI *= 0.0
        self.FI_alpha *= 0.0
        for (n, p) in self.named_parameters():
            self.FI += torch.sum(p.grad.data ** 2).cpu()
            name = n.split(".")
            if name[0] == "cells" and name[3].isdigit() and name[5].isdigit():
                if int(name[1]) in self._reduce:
                    self.FI_reduce[int(name[3]), int(name[5])] += torch.sum(p.grad.data ** 2).cpu() / len(self._reduce)
                else:
                    self.FI_normal[int(name[3]), int(name[5])] += torch.sum(p.grad.data ** 2).cpu() / (self._layers - len(self._reduce))

        self.FI_history.append(float(self.FI))

        if alpha_step:
            for p in self._arch_parameters:
                self.FI_alpha += torch.sum(p.grad.data ** 2).cpu()

            self.FI_alpha_history.append(float(self.FI_alpha))
            self.FI_alpha_history_step.append(self.clock)

        if self.FI_ewma == -1:
            self.FI_ewma = self.FI
        else:
            self.FI_ewma = self._ewma*self.FI + (1-self._ewma)*self.FI_ewma
        self.FI_ewma_history.append(float(self.FI_ewma))
