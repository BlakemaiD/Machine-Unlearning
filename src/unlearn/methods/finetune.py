from torch import nn, optim
from ..device import DEVICE


def unlearning(net, retain, forget, validation, epochs=5, lr=0.1, momentum=0.9, weight_decay=5e-4):
    """Unlearning by fine-tuning on the retain set only (forget/validation unused)."""
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()

    for _ in range(epochs):
        for inputs, targets in retain:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(inputs), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

    net.eval()
    return net
