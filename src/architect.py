import torch


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def step(self, input_valid, target_valid):
        self.optimizer.zero_grad()
        loss = self._backward_step(input_valid, target_valid)
        self.optimizer.step()
        return loss

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()
        for x in self.model._arch_parameters:
            if torch.isnan(x).any():
                print(loss)
                print(x)
        return loss
