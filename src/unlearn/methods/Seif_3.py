#Always deepcopy the pre-trained model so the methods don't contaminate each other.

import sys
import os
import subprocess
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
import torch.nn.init as init

#from unlearn.methods import finetune, competition
#model_u = competition.unlearning(copy.deepcopy(model), retain_loader, forget_loader, val_loader)


sys.path.insert(0, os.path.abspath("../src")
                )

# It's really important to add an accelerator to your notebook, as otherwise the submission will fail.
# We recomment using the P100 GPU rather than T4 as it's faster and will increase the chances of passing the time cut-off threshold.


def unpack_batch(sample):
    if isinstance(sample, dict):
        images = sample.get(("image"), sample.get("images"))
        labels = None
        for key in ("label", "labels", "target", "targets", "age_group"):
            if key in sample:
                labels = sample[key]
                break
        if images is None or labels is None:
            raise KeyError(f"Could not find images and labels in katch keys: {list(sample.keys())}")
        
    elif isinstance(sample, (tuple, list)):
        images, labels = sample[0], sample[1]
       
    else:
        images, labels = sample[0], sample[1]
          
    return images, labels

def unlearning(
    net = None, 
    retain_loader = None, 
    forget_loader = None, 
    val_loader = None,
    device = None):
    
        
    class CustomCrossEntropyLoss(nn.Module):
        def __init__(self, class_weights=None):
            super(CustomCrossEntropyLoss, self).__init__()
            self.class_weights = class_weights

        def forward(self, input, target):
            # Compute the standard cross-entropy loss
            ce_loss = nn.functional.cross_entropy(input, target)

            # Apply class weights to the loss if provided
            if self.class_weights is not None:
                # Calculate the weights for each element in the batch based on the target
                weights = torch.tensor([self.class_weights[i] for i in target], device=input.device)
                ce_loss = torch.mean(ce_loss * weights)

            return ce_loss

    

    # Define the vision_confuser function
    def vision_confuser(model, std = 0.6):
        for name, module in model.named_children():
            if hasattr(module, 'weight'):
                if 'conv' in name:
   
                    actual_value = module.weight.clone().detach()
                    new_values = torch.normal(mean=actual_value, std=std)
                    module.weight.data.copy_(new_values)
                
    vision_confuser(net)

    epochs = 4
    
    w = 0.05
    
    class_weights = [1, w, w, w, w, w, w, w, w, w] 
    criterion = CustomCrossEntropyLoss(class_weights)
    
    
    optimizer = optim.SGD(net.parameters(), lr=0.0007,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    
    net.train()
    i=0
    
    for ep in range(epochs):
        i=0
        net.train()
        ### this needs to be fixed to l ook at a dictionary of the sample, not just the image and label
        for sample in retain_loader:
            inputs, targets = unpack_batch(sample)

            inputs = inputs.to(device)
            targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (ep == epochs-2):
            vision_confuser(net , 0.005) # increase model robustness before last training epoch
            
        scheduler.step()
        print(f"Epoch [{ep+1}/{epochs}], Loss: {loss.item():.4f}")

    net.eval()
    return net

'''
#implementing methods

import copy
import importlib
import torch
from unlearn.device import DEVICE
from unlearn.data import get_cifar10_loaders
from unlearn.models import load_cifar10_model
from unlearn.metrics import accuracy, mia_accuracy
from unlearn.methods import finetune, Seif_3

importlib.reload(Seif_3)


from unlearn.methods import finetune
ft_model = finetune.unlearning(copy.deepcopy(model), retain_loader, forget_loader, val_loader)

ft_model.eval()

# 2. Seif_3 method
model_u = Seif_3.unlearning(
    net=copy.deepcopy(model),
    retain_loader=retain_loader,
    forget_loader=forget_loader,
    val_loader=val_loader,
    device=DEVICE,
)


# 3. Evaluate both methods using the same data loaders
results = {}

for name, trained_model in [
    ("Fine-tuning", ft_model),
    ("Seif_3", model_u),
]:
    retain_acc = accuracy(trained_model, retain_loader)
    test_acc = accuracy(trained_model, test_loader)
    mia = mia_accuracy(
        trained_model, forget_loader, test_loader
    )

    results[name] = {
        "Retain accuracy": retain_acc,
        "Test accuracy": test_acc,
        "MIA": mia,
    }

    print(f"\n{name}")
    print(f"Retain accuracy: {100 * retain_acc:.3f}%")
    print(f"Test accuracy:   {100 * test_acc:.3f}%")
    print(f"MIA:             {mia:.3f}")

'''

