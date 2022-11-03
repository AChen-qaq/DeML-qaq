import torch
import math

class GradientBuffer:
    def __init__(self, model):
        self.model = model
        self.gradients = {name: torch.zeros_like(
            param, requires_grad=False) for name, param in model.named_parameters()}
        self.sparse_ratio = 0.01
        self.lr = 1e-2
        # print(name, param.shape)

    def catch_gradients(self):
        for name, param in self.model.named_parameters():
            self.gradients[name] += param.grad

    def construct_mask(self):
        grad_masks = {}
        for name, grad in self.gradients.items():
            numel=grad.numel()
            k = int(math.ceil(math.sqrt(numel)))
            num_samples=numel//k
            samples_indices = torch.randint(0, numel, (num_samples, ), device=grad.device)


            threshold = torch.max(torch.abs(grad.view(-1)[samples_indices]))

            # threshold = 0
            mask = (torch.abs(grad) >= threshold)
            # print(mask.sum(), numel)
            grad_masks[name] = mask

        return grad_masks

    @torch.no_grad()
    def apply_and_save_gradients(self):

        grad_masks = self.construct_mask()

        for name, param in self.model.named_parameters():
            grad = self.gradients[name]
            mask = grad_masks[name]
            param[mask] -= self.lr * grad[mask]
            grad[mask] = 0

        # for name, mask in grad_masks:
        #     grad = self.gradients[name]
        #     param = self.

