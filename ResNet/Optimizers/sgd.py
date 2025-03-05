import torch

class SGD():
    def __init__(self, opt_params, args):
        """
        A custom implementation of SGD with PyTorch-compatible param_groups.

        Args:
            opt_params (iterable): Parameters to be optimized.
            args (Namespace): Arguments containing optimizer hyperparameters.
        """
        self.param_groups = [{"params": list(opt_params), "lr": args.learning_rate}]
        self.lr = args.learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.nesterov = args.nesterov
        self.velocity = {param: torch.zeros_like(param, device=param.device) for param in self.param_groups[0]["params"]}

    def step(self):
        """Performs a single optimization step."""
        with torch.no_grad():
            for param in self.param_groups[0]["params"]:
                if param.grad is not None:
                    grad = param.grad
                    
                    # Apply weight decay (L2 regularization)
                    if self.weight_decay > 0:
                        grad = grad + self.weight_decay * param
                    
                    # Momentum update
                    if self.momentum > 0:
                        self.velocity[param] = self.velocity[param].to(grad.device)  # **确保动量和梯度在相同设备**
                        self.velocity[param] = self.momentum * self.velocity[param] + grad

                        
                        # Nesterov momentum
                        if self.nesterov:
                            update = grad + self.momentum * self.velocity[param]
                        else:
                            update = self.velocity[param]
                    else:
                        update = grad
                    
                    # Update parameters using lr from param_groups
                    param -= self.param_groups[0]["lr"] * update

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.param_groups[0]["params"]:
            if param.grad is not None:
                param.grad.zero_()
