import torch.optim as optim

class Adam():
    def __init__(self, opt_params, args):
        self.optimizer = optim.Adam(
            opt_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
