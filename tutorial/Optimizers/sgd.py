import torch.optim as optim

class SGD():
    def __init__(self, opt_params, args):
        self.optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )

    def step(self):
        """Performs a single optimization step."""
        self.optimizer.step()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.optimizer.zero_grad()