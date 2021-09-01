import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from genotypes_rnn import CRBPRIMITIVES, PRIMITIVES, STEPS, CONCAT, Genotype
from model_rnn import DARTSCell, RNNModel


class DARTSCellSearch(DARTSCell):

    def __init__(self, ninp, nhid, dropouth, dropoutx, genotype, crb):
        super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None, crb=None)
        self.bn = nn.BatchNorm1d(nhid, affine=False)
        self._crb = crb
        if self._crb:
            self.primitives = CRBPRIMITIVES
        else:
            self.primitives = PRIMITIVES

    def activate(self, alphas):
        if self._crb:
            return torch.clamp(alphas, min=0.0, max=1.0)
        else:
            return F.softmax(alphas, dim=-1)

    def cell(self, x, h_prev, x_mask, h_mask):
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
        s0 = self.bn(s0)

        probs = self.activate(self.weights)

        offset = 0
        states = s0.unsqueeze(0)
        for i in range(STEPS):
            if self.training:
                masked_states = states * h_mask.unsqueeze(0)
            else:
                masked_states = states
            ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i + 1, -1, 2 * self.nhid)
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()

            s = torch.zeros_like(s0)
            for k, name in enumerate(self.primitives):
                if name == 'none':
                    continue
                fn = self._get_activation(name)
                unweighted = states + c * (fn(h) - states)
                s += torch.sum(probs[offset:offset + i + 1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
            s = self.bn(s)
            states = torch.cat([states, s.unsqueeze(0)], 0)
            offset += i + 1
        output = torch.mean(states[-CONCAT:], dim=0)
        return output


class RNNModelSearch(RNNModel):

    def __init__(self, crb, rho, ewma, reg, epochs, *args):
        super(RNNModelSearch, self).__init__(*args, cell_cls=DARTSCellSearch, genotype=None, crb=crb)
        self._crb = crb
        if self._crb:
            self.primitives = CRBPRIMITIVES
        else:
            self.primitives = PRIMITIVES
        self._rho = rho
        self._ewma = ewma
        self._reg = reg
        self._num_ops = len(self.primitives)
        self.clock = 0.0
        self.total_epochs = epochs
        self._args = args
        self._initialize_arch_parameters()

    def tick(self, step):
        self.clock += step

    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def activate(self, alphas):
        if self._crb:
            return torch.clamp(alphas, min=0.0, max=1.0)
        else:
            return F.softmax(alphas, dim=-1)

    def _initialize_arch_parameters(self):
        k = sum(i for i in range(1, STEPS + 1))
        if self._crb:
            self.weights = Variable((1 / self._num_ops + 1e-4 * torch.randn(k, self._num_ops)).cuda(),
                                    requires_grad=True)
        else:
            weights_data = torch.randn(k, len(self.primitives)).mul_(1e-3)
            self.weights = Variable(weights_data.cuda(), requires_grad=True)
        self._arch_mask = torch.ones_like(self.weights)
        self.mask_alphas()
        self.FI = 0
        self.FI_ewma = -1
        for rnn in self.rnns:
            rnn.weights = self.weights

    def arch_parameters(self):
        return [self.weights]

    def mask_alphas(self):
        if self._crb:
            with torch.no_grad():
                self._arch_mask[self.weights <= 0.0] = 0.0
                self.weights[self.weights <= 0.0] = 0.0
                self.weights.mul_(self._arch_mask)

    def _loss(self, hidden, input, target):
        log_prob, hidden_next = self(input, hidden, return_h=False)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
        if self._reg == "prox":
            activated = self.activate(self.weights)
            _, disc = self._parse(activated.detach().cpu().clone())
            prox_reg = self.clock / self.total_epochs * self._rho / 2 * ((activated - disc.cuda()).norm())
            print(prox_reg)
            loss += prox_reg
        return loss, hidden_next

    def _parse(self, probs):
        gene = []
        z = torch.zeros_like(probs).cpu()
        probs = probs.numpy()
        start = 0
        for i in range(STEPS):
            end = start + i + 1
            W = probs[start:end].copy()
            edges = sorted(range(i + 1),
                           key=lambda x: -max(W[x][k] for k in range(len(W[x])) if self.primitives[k] != "none"))
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if self.primitives[k] != "none":
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((self.primitives[k_best], j))
                z[j + start, k_best] = 1.0
            start = end
        return gene, z

    def genotype(self):
        gene = self._parse(self.activate(self.weights).data.cpu())
        genotype = Genotype(recurrent=gene, concat=range(STEPS + 1)[-CONCAT:])
        return genotype

    def track_FI(self):
        self.FI *= 0.0
        for (n, p) in self.named_parameters():
            self.FI += torch.sum(p.grad.data ** 2).cpu()

        # self.FI_history.append(float(self.FI))

        if self.FI_ewma == -1:
            self.FI_ewma = self.FI
        else:
            self.FI_ewma = self._ewma * self.FI + (1 - self._ewma) * self.FI_ewma
        # self.FI_ewma_history.append(float(self.FI_ewma))
