import torch
from torch.optim.optimizer import Optimizer

class LookSAM(Optimizer):
    """
    LookSAM: Efficient Sharpness-Aware Minimization via Lookahead.
    
    Combines SAM and Lookahead:
      - SAM perturbs parameters along the gradient, restores them, and then updates them.
      - Lookahead interpolates slow weights with fast weights every k steps.
    
    Args:
      params: Model parameters.
      base_optimizer: Base optimizer class, such as torch.optim.SGD or AdamW.
      rho: SAM perturbation radius.
      alpha: Lookahead interpolation coefficient.
      k: Number of steps between Lookahead updates.
      **kwargs: Additional arguments passed to the base optimizer.
    """
    def __init__(self, params, base_optimizer, rho=0.05, alpha=0.5, k=5, **kwargs):
        if rho < 0.0:
            raise ValueError("Invalid rho, should be non-negative: {}".format(rho))
        defaults = dict(rho=rho, alpha=alpha, k=k, **kwargs)
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.rho = rho
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        # Share parameter groups with the base optimizer.
        self.param_groups = self.base_optimizer.param_groups
        # Initialize Lookahead's slow weights from the model parameters.
        self.slow_weights = []
        for group in self.param_groups:
            group_slow = []
            for p in group['params']:
                group_slow.append(p.clone().detach())
            self.slow_weights.append(group_slow)
        super(LookSAM, self).__init__(params, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Apply the SAM perturbation step."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Compute a non-adaptive perturbation.
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # Apply the perturbation.
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore the parameters and update them with the base optimizer."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # Restore the original parameters.
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure):
        """
        Run one LookSAM step using a closure that recomputes the loss.

        The method applies the SAM perturbation, recomputes gradients, restores and
        updates the parameters, and updates Lookahead weights every k steps.
        """
        if closure is None:
            raise ValueError("LookSAM requires closure, but it was not provided")
        closure = torch.enable_grad()(closure)

        # SAM update
        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step(zero_grad=True)

        # Update slow weights and synchronize them with the model every k steps.
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            for group, slow_group in zip(self.param_groups, self.slow_weights):
                for p, slow in zip(group["params"], slow_group):
                    # Linearly interpolate the slow weights.
                    slow.add_(self.alpha * (p.data - slow))
                    p.data.copy_(slow)
        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        # Compute the L2 norm over all parameter gradients.
        device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2).to(device) for group in self.param_groups for p in group["params"] if p.grad is not None
        ]), p=2)
        return norm
