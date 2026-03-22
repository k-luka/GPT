import torch


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic polynomial instead of the standard cubic polynomial in order to converge 1.5x faster.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd != 0.0:
                    p.data.mul_(1 - lr * wd)
                g = p.grad
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g.ndim == 2, "Muon only supports 2D params"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = zeropower_via_newtonschulz5(buf)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.data.add_(g, alpha=-lr)


class DualOptimizer:
    def __init__(self, adam_opt, muon_opt):
        self.adam_opt = adam_opt
        self.muon_opt = muon_opt
        self.param_groups = adam_opt.param_groups + muon_opt.param_groups

    def zero_grad(self):
        self.adam_opt.zero_grad()
        self.muon_opt.zero_grad()

    def step(self):
        self.adam_opt.step()
        self.muon_opt.step()

    def get_adamw_params(self):
        return [p for group in self.adam_opt.param_groups for p in group["params"]]

    def set_lrs(self, adam_lr, muon_lr):
        for group in self.adam_opt.param_groups:
            group["lr"] = adam_lr
        for group in self.muon_opt.param_groups:
            group["lr"] = muon_lr

    def state_dict(self):
        return {
            "adam_opt": self.adam_opt.state_dict(),
            "muon_opt": self.muon_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.adam_opt.load_state_dict(state_dict["adam_opt"])
        self.muon_opt.load_state_dict(state_dict["muon_opt"])
