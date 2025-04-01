import torch
from torch.optim.optimizer import Optimizer

class LookSAM(Optimizer):
    """
    LookSAM: Efficient Sharpness-Aware Minimization via Lookahead.
    
    结合了 SAM 和 Lookahead 两种策略：
      - SAM：在每个 step 中首先在梯度方向上进行扰动，然后恢复再更新参数；
      - Lookahead：每隔 k 个 step 对“慢速权重”进行更新，与快速权重进行线性插值。
    
    参数：
      params: 模型参数
      base_optimizer: 基础优化器类，如 torch.optim.SGD 或 torch.optim.AdamW
      rho: SAM 扰动半径
      alpha: Lookahead 更新系数（慢速权重与快速权重融合比例）
      k: Lookahead 更新频率（每 k 个 step 更新一次慢速权重）
      **kwargs: 传给基础优化器的其他参数
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
        # 使用基础优化器的参数组
        self.param_groups = self.base_optimizer.param_groups
        # 初始化 Lookahead 的慢速权重（复制每个参数）
        self.slow_weights = []
        for group in self.param_groups:
            group_slow = []
            for p in group['params']:
                group_slow.append(p.clone().detach())
            self.slow_weights.append(group_slow)
        super(LookSAM, self).__init__(params, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """计算 SAM 扰动，并对参数进行上升步"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 计算扰动（此处不使用 adaptive 调整，可根据需求修改）
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # 在参数上添加扰动
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """恢复参数并使用基础优化器更新"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # 恢复原始参数
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure):
        """
        LookSAM 的 step 要求传入 closure 来重新计算损失。
        执行过程：
          1. 进行 SAM 的第一步（扰动）
          2. 使用 closure 重新计算损失并反向传播
          3. 恢复参数并执行 SAM 更新
          4. 每隔 k 个 step 执行 Lookahead 更新
        """
        if closure is None:
            raise ValueError("LookSAM requires closure, but it was not provided")
        closure = torch.enable_grad()(closure)

        # SAM update
        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step(zero_grad=True)

        # Lookahead update: 每 k 个 step 更新慢速权重，并同步到模型参数
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            for group, slow_group in zip(self.param_groups, self.slow_weights):
                for p, slow in zip(group["params"], slow_group):
                    # 线性插值更新慢速权重
                    slow.add_(self.alpha * (p.data - slow))
                    p.data.copy_(slow)
        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        # 计算所有参数梯度的 L2 范数
        device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2).to(device) for group in self.param_groups for p in group["params"] if p.grad is not None
        ]), p=2)
        return norm
