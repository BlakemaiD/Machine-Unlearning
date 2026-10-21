from unlearn.methods import METHODS
ft_model = METHODS["finetune"](copy.deepcopy(model), retain_loader, forget_loader, val_loader)

