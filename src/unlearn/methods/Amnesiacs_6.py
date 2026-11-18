#Always deepcopy the pre-trained model so the methods don't contaminate each other.

import os
import subprocess

import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset

import random
import torch.nn.functional as F
import numpy as np

#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#There are 2 models, one the teacher and the other the student.
#Teacher: keeps the original weights and provides target predictions.
#Student: starts from the original weights, then has conv1 and fc reinitialized.



#from unlearn.methods import finetune, competition
#model_u = competition.unlearning(copy.deepcopy(model), retain_loader, forget_loader, val_loader)
#def unlearning(net, retain, forget, validation):
#    raise NotImplementedError("competition code not added yet")