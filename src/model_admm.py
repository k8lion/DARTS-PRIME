import torch.nn.functional as F
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
                weights = torch.tanh(self.alphas_reduce)
            else:
                weights = torch.tanh(self.alphas_normal)
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

    def mask_alphas(self):
        with torch.no_grad():
            for param in self._arch_parameters:
                param[param <= 0] = -float("inf")

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
        idx = 0
        loss = self._criterion(output, target)
        for param in self.arch_parameters():
            u = self.U[idx].cuda()
            z = self.Z[idx].cuda()
            loss += self._rho / 2 * (param - z + u).norm()
            #if args.l01:
                #loss += args.alpha * -F.mse_loss(param, torch.tensor(0.5, requires_grad=False).cuda())
            idx += 1
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
        self.Z = new_Z
        print(self.Z)


    def update_U(self):
        new_U = ()
        for u, x, z in zip(self.U, self._arch_parameters, self.Z):
            new_u = u + x.detach().cpu().clone() - z
            new_U += (new_u,)
        self.U = new_U
        print(self.U)
