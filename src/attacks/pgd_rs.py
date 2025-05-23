import torch
import torch.nn.functional as F
import copy

from torchattacks.attack import Attack

class Noise():
    def __init__(self, noise_type, noise_sd):
        self.noise_type = noise_type
        self.noise_sd = noise_sd

    def __call__(self, img):
        if self.noise_type == "gaussian":
            noise = torch.randn_like(img.float())*self.noise_sd
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(img.float()) - 0.5)*2*self.noise_sd
        return noise


class PGD_RS(Attack):
    r"""
    PGD for randmized smoothing in the paper 'Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers'
    [https://arxiv.org/abs/1906.04584]
    Modification of the code from https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/attacks.py and 'torchattacks'

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        noise_type (str): guassian or uniform. (Default: guassian)
        noise_sd (float): standard deviation for normal distributio, or range for . (Default: 0.5)
        noise_batch_size (int): guassian or uniform. (Default: 32)
        batch_max (int): split data into small chunk if the total number of augmented data points, len(inputs)*noise_batch_size, are larger than batch_max, in case GPU memory is insufficient. (Default: 2048)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = PGD_RS(model, eps=8/255, alpha=1/255, steps=40, noise_type = "guassian", noise_sd = 0.5, noise_batch_size = 32)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.3, alpha=2/255, steps=40, 
                 noise_type = "guassian", noise_sd = 0.5, noise_batch_size = 32, batch_max = 2048):
        super().__init__("PGD_RS", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.noise_func = Noise(noise_type, noise_sd)
        self.noise_batch_size = noise_batch_size
        self._supported_mode = ['default', 'targeted']
        self.batch_max = batch_max

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if inputs.shape[0]*self.noise_batch_size > self.batch_max:
            size_comp = len(self.model.device_ids) if hasattr(self.model, 'device_ids') else 1
            img_list = []
            inputs_split = torch.split(inputs, split_size_or_sections = int(self.batch_max/self.noise_batch_size*size_comp))
            labels_split = torch.split(labels, split_size_or_sections = int(self.batch_max/self.noise_batch_size*size_comp))
            for img_sub, lab_sub in zip(inputs_split, labels_split):
                img_adv = self._forward(img_sub, lab_sub)
                img_list.append(img_adv)
            return torch.vstack(img_list)
        else:
            return self._forward(inputs, labels)

    def _forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        #expend the inputs over noise_batch_size
        shape_exp = torch.Size([images.shape[0], self.noise_batch_size]) + images.shape[1:]
        inputs_exp = images.unsqueeze(1).expand(shape_exp)
        inputs_exp = inputs_exp.reshape(torch.Size([-1]) + inputs_exp.shape[2:])

        delta = torch.zeros((len(labels), *inputs_exp.shape[1:]), requires_grad=True, device=self.device)
        delta_last = torch.zeros((len(labels), *inputs_exp.shape[1:]), requires_grad=False, device=self.device)

        for _ in range(self.steps):
            delta.requires_grad = True
            #img_adv is the perturbed data for randmized smoothing
            img_adv = inputs_exp + delta.unsqueeze(1).expand(shape_exp).reshape(inputs_exp.shape)
            img_adv = torch.clamp(img_adv, min=0, max=1)

            noise_added = self.noise_func(img_adv.view(len(img_adv), -1))
            noise_added = noise_added.view(img_adv.shape)

            adv_noise = img_adv + noise_added
            logits = self.model(adv_noise)

            softmax = F.softmax(logits, dim=1)
            # average the probabilities across noise
            average_softmax = softmax.reshape(-1, self.noise_batch_size, logits.shape[-1]).mean(1, keepdim=True).squeeze(1)
            logsoftmax = torch.log(average_softmax.clamp(min=1e-20))
            ce_loss = F.nll_loss(logsoftmax, labels)

            grad = torch.autograd.grad(ce_loss, delta, retain_graph=False, create_graph=False)[0]
            delta = delta_last + self.alpha*torch.sign(grad)
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            delta_last.data = copy.deepcopy(delta.data)

        return torch.clamp(images + delta, min=0, max=1).detach()