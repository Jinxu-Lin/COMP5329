import torch

class Adam:
    def __init__(self, opt_params, args):
        """
        A custom implementation of Adam optimizer.

        Args:
            opt_params (iterable): Parameters to be optimized.
            args (Namespace): Arguments containing optimizer hyperparameters.
                - args.learning_rate (float): Learning rate for Adam.
                - args.weight_decay (float): Weight decay (L2 regularization).
        """
        self.param_groups = [{"params": list(opt_params), "lr": args.learning_rate}]
        self.lr = args.learning_rate
        self.weight_decay = args.weight_decay
        self.beta1 = 0.9  # Default Adam beta1
        self.beta2 = 0.999  # Default Adam beta2
        self.eps = 1e-8  # Small constant for numerical stability
        self.t = 0  # Time step

        # Initialize first and second moment estimates (m & v)
        self.m = {param: torch.zeros_like(param, device=param.device) for param in self.param_groups[0]["params"]}
        self.v = {param: torch.zeros_like(param, device=param.device) for param in self.param_groups[0]["params"]}

    def step(self):
        """Performs a single optimization step using Adam update rule."""
        self.t += 1  # Increment time step
        with torch.no_grad():
            for param in self.param_groups[0]["params"]:
                if param.grad is not None:
                    grad = param.grad

                    # Apply weight decay (L2 regularization)
                    if self.weight_decay > 0:
                        grad = grad + self.weight_decay * param
                    
                    # Update biased first moment estimate (m)
                    self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad

                    # Update biased second raw moment estimate (v)
                    self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)

                    # Compute bias-corrected first moment estimate
                    m_hat = self.m[param] / (1 - self.beta1 ** self.t)

                    # Compute bias-corrected second raw moment estimate
                    v_hat = self.v[param] / (1 - self.beta2 ** self.t)

                    # Update parameter using Adam formula
                    param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.param_groups[0]["params"]:
            if param.grad is not None:
                param.grad.zero_()
