#metrics of how the different algorithms compare and perform according to metrics of the competition.
import numpy as np
import torch
from torch import nn
from sklearn import linear_model, model_selection

from .device import DEVICE

def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    net.to(DEVICE)
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


@torch.no_grad()
def compute_losses(net, loader):
    """Per-sample cross-entropy losses."""
    net.to(DEVICE)
    net.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        losses = criterion(net(inputs), targets).numpy(force=True)
        all_losses.extend(losses)
    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")
    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def mia_accuracy(net, forget_loader, test_loader, seed=0):
    """Forget-vs-test MIA accuracy, class-balanced. Replaces the repeated notebook cells."""
    f = compute_losses(net, forget_loader)
    t = compute_losses(net, test_loader)
    if len(f) > len(t):
        f = np.random.default_rng(seed).permutation(f)[: len(t)]
    X = np.concatenate((t, f)).reshape(-1, 1)
    y = [0] * len(t) + [1] * len(f)
    return simple_mia(X, y).mean()