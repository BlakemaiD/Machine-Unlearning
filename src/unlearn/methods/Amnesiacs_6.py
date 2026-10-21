#Always deepcopy the pre-trained model so the methods don't contaminate each other.

from unlearn.methods import finetune, competition
model_u = competition.unlearning(copy.deepcopy(model), retain_loader, forget_loader, val_loader)

def unlearning(net, retain, forget, validation):
    raise NotImplementedError("competition code not added yet")