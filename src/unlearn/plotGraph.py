import matplotlib.pyplot as plt

from .metrics import accuracy, compute_losses, mia_accuracy     

def plot_forget_vs_test(models, forget_loader, test_loader, xlim=None):
    """models: dict of name -> net. One panel per model, forget vs test loss histograms."""
   
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True, squeeze=False)
    max_loss = 0
    for ax, (name, net) in zip(axes[0], models.items()):
        f = compute_losses(net, forget_loader)
        t = compute_losses(net, test_loader)
        max_loss = max(max_loss, t.max())
        mia = mia_accuracy(net, forget_loader, test_loader)
        ax.set_title(f"{name}\nAttack accuracy: {mia:.2f}")
        ax.hist(t, density=True, alpha=0.5, bins=50, label="Test set")
        ax.hist(f, density=True, alpha=0.5, bins=50, label="Forget set")
        ax.set_yscale("log")
        ax.set_xlabel("Loss")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[0]:
        ax.set_xlim(0, xlim or max_loss)
    axes[0][0].set_ylabel("Frequency")
    axes[0][0].legend(frameon=False)
    return fig