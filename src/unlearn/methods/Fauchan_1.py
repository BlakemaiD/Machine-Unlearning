#Always deepcopy the pre-trained model so the methods don't contaminate each other.
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.insert(0, os.path.abspath("../src")
                )
#from unlearn.methods import finetune, competition
#model_u = competition.unlearning(copy.deepcopy(model), retain_loader, forget_loader, val_loader)

def kl_loss_sym(x,y):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    return kl_loss(nn.LogSoftmax(dim=-1)(x),y)


'''
Fauchan Unlearning Method
Stage 1: Encouraging Uniform predictions on forgotten samples
Stage 2: Contrastive learning between forgotten and retained samples
Returns: updated model
'''

#reproducibility
torch.manual_seed(3047) #initializes model weight randomness for reproducibility

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
        net,
        retain_loader,
        forget_loader,
        val_loader=None,
        epochs=None,
        retain_bs=None,
        device=None,
        temperature=None,
):
    """Simple unlearning by finetuning."""
    print('-----------------------------------')

    if device is None:
        device = next(net.parameters()).device #iterates over model's weight tensors, and gets the device of the first one
    net = net.to(device) #reads where the tensor lives (cpu/cuda)

    if temperature is None:
        temperature = 1.15

    if epochs is None:
        epochs = 8

    if retain_bs is None:
        retain_bs = 256

    criterion = nn.CrossEntropyLoss()

    #optimizer setup for 1st forgetting stage
    optimizer = optim.SGD(net.parameters(), lr=0.005,
                          momentum=0.9, weight_decay=0)

    #optimizer setup for retained data fine tuning
    optimizer_retain = optim.SGD(net.parameters(), lr=0.001*retain_bs/64, momentum=0.9, weight_decay=1e-2)
    ##the learning rate is associated with the batchsize we used
    
    #optimizer setup for contrastive-style forgetting stage
    optimizer_forget = optim.SGD(net.parameters(), lr=3e-4, momentum=0.9, weight_decay=0)

    #rebuilding loaders with batch sizes used in the original code
    retain_ld = DataLoader(retain_loader.dataset, batch_size=retain_bs, shuffle=True)
    retain_ld4fgt = DataLoader(retain_loader.dataset, batch_size=256, shuffle=True)
    
    #Cosine learning-rate schedule for forgetting stage
    total_step = int(len(forget_loader)*epochs)
    scheduler = CosineAnnealingLR(optimizer_forget, T_max=total_step, eta_min=1e-6)
    
    #Stage 1: Encouraging Uniform predictions on forgotten samples
    net.train()

    for sample in forget_loader: 
        images, labels = unpack_batch(sample) #what does this do? Idk
        images = images.to(device)

        optimizer.zero_grad()

    
        outputs = net(images)

        uniform_label = torch.ones_like(outputs).to(device) / outputs.shape[1] ##uniform pseudo label
        loss = kl_loss_sym(outputs, uniform_label) ##optimize the distance between logits and pseudo labels
        loss.backward()
        optimizer.step()

    #Stage 2: alternate between forgetting and retaining rounds
    for ep in range(epochs):
        net.train()

        #forget round
        for sample_forget, sample_retain in zip(forget_loader, retain_ld4fgt):##Forget Round
            inputs_forget, __ = unpack_batch(sample_forget) 
            inputs_retain, __ = unpack_batch(sample_retain)
            
            inputs_forget = inputs_forget.to(device)
            inputs_retain = inputs_retain.to(device)
            optimizer_forget.zero_grad()
            
            outputs_forget,outputs_retain = net(inputs_forget),net(inputs_retain).detach()
            loss = (-1 * nn.LogSoftmax(dim=-1)(outputs_forget @ outputs_retain.T/temperature)).mean() ##Contrastive Learning loss
            loss.backward()
            optimizer_forget.step()
            scheduler.step()


        #retain round 
        for sample in retain_ld: 
            inputs, labels = unpack_batch(sample) #sample["image"],sample["age_group"]
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            optimizer_retain.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_retain.step()
            
    print('-----------------------------------')
    return net

def evaluation(net, dataloader, criterion, device = 'cuda'): ##evaluation function
    net.eval()
    total_samp = 0
    total_acc = 0
    total_loss = 0.0
    for sample in dataloader:
        images, labels = sample['image'].to(device), sample['age_group'].to(device)
        _pred = net(images)
        total_samp+=len(labels)
        #print(f'total_samp={total_samp}')
        loss = criterion(_pred, labels)
        total_loss += loss.item()
        total_acc+=(_pred.max(1)[1] == labels).float().sum().item()
        #print(f'total_acc={total_acc}')
    #print(f'total_sample={total_samp}')
    mean_loss = total_loss / len(dataloader)
    mean_acc = total_acc/total_samp
    print(f'loss={mean_loss}')
    print(f'acc={mean_acc}')
    return mean_loss, mean_acc