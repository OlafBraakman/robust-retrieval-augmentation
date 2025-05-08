import torch.nn as nn
import torch
import math
from .attack import Attack
from tqdm import tqdm
from torchmetrics.classification import Accuracy

class PGD(Attack):

    def __init__(self, num_classes, eps=0.1, alpha=2/255, max_iters=30, device="cuda", **kwargs) -> None:
        super().__init__()

        self.eps = eps
        self.alpha = alpha
        self.max_iters = max_iters

        self.device = device

        self.original_accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='micro').to(device)
        self.original_accuracy_macro = Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)

        self.perturb_accs = [Accuracy(task='multiclass', num_classes=num_classes, average='micro').to(device) for _ in range(self.max_iters)]
        self.perturb_accs_macro = [Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device) for _ in range(self.max_iters)]

    def run(self, model, dataloader):

        def iter_cb(iter, X, y):
            # save_img(X, 'pgd', iter)
            perturb_pred = model(X)#.argmax(dim=1)

            self.perturb_accs[iter](perturb_pred, y)
            self.perturb_accs_macro[iter](perturb_pred, y)

        self.iter_cb = iter_cb

        for (X,y), _ in tqdm(dataloader):
            X = X.float().to(self.device)
            y = y.to(self.device)
    
            # save_img(X, 'pgd', 'orig')
            pred = model(X)
            self.original_accuracy(pred, y)
            self.original_accuracy_macro(pred, y)

            self.pgd_attack(model, X, y)

        return self.original_accuracy.compute().item(), \
        self.original_accuracy_macro.compute().item(), \
        [acc.compute().item() for acc in self.perturb_accs], \
        [acc.compute().item() for acc in self.perturb_accs_macro]


    def pgd_attack(self, model, images, labels) :
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
            
        ori_images = images.data
            
        for i in range(self.max_iters) :    
            images.requires_grad = True
            outputs = model(images)

            model.zero_grad()
            cost = loss(outputs, labels).to(self.device)
            cost.backward()

            adv_images = images + self.alpha*images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

            self.iter_cb(i, images, labels)
                
        return images

    def save(self, root_dir):
        pass

# def pgd_attack_smooth(model, images, labels, eps=0.1, alpha=2/255, iters=30, total_samples=1000, mini_batch_size=32, sigma=0.5, loss_fn=None, device="cuda"):
#     images = images.to(device)
#     labels = labels.to(device)
#     if loss_fn is None:
#         loss_fn = torch.nn.CrossEntropyLoss()

#     ori_images = images.data

#     # PGD iteration loop
#     for i in range(iters):
#         images.requires_grad = True
        
#         # Compute the output from the model to get the "smoothed" prediction.
#         # Here, we assume that in the smoothed branch,
#         # you need to average the gradients over noise samples.
#         # Instead of building a huge graph for all 1000 samples,
#         # use mini-batch based gradient accumulation.
#         model.compute_loss_pred(images, )
        
#         # total_loss = compute_noise_loss(images, labels, loss_fn, total_samples=total_samples, mini_batch_size=mini_batch_size, sigma=sigma)
        
#         # Clear existing gradients:
#         model.zero_grad()
        
#         # Backpropagate the accumulated loss.
#         total_loss.backward()
        
#         # Update the images using the calculated gradients.
#         # For example, this might be a simple projected gradient update.
#         # (Alpha is the step size for your PGD, ensure it matches your desired behavior)
#         images = images + alpha * images.grad.sign()
#         images = torch.clamp(images, ori_images - eps, ori_images + eps)
#         images = torch.clamp(images, 0, 1).detach()
        
#     return images