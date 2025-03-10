from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO
                # From respondent: order of implementation above is different
                # from Pytorch document of `AdamW`

                betas = group["betas"]
                if "m_t" not in state:
                    # initialization
                    state["m_t"] = 0
                state["m_t"] = betas[0] * state["m_t"] + (1 - betas[0]) * grad
                if "v_t" not in state:
                    # initialization
                    state["v_t"] = 0
                state["v_t"] = betas[1] * state["v_t"] + (1 - betas[1]) * torch.pow(grad, 2)

                if "t" not in state:
                    # initialization
                    state["t"] = 0
                state["t"] = state["t"] + 1

                p.data = p.data - alpha * group["weight_decay"] * p.data
                if group["correct_bias"]:
                    alpha = alpha * (math.sqrt(1 - math.pow(betas[1], state["t"]))) \
                        / (1 - math.pow(betas[0], state["t"]))

                p.data = p.data - alpha * state["m_t"] / (torch.sqrt(state["v_t"]) + group["eps"])


        return loss
