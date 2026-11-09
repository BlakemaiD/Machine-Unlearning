<img src="https://github.com/unlearning-challenge/starting-kit/assets/277639/d1fa7889-5d91-4e6d-8082-7d59ef728f9c" style="width: 100px">



# Starting kit for the NeurIPS 2023 Machine Unlearning Challenge

This repository contains the starting kit for the NeurIPS 2023 Machine Unlearning Challenge. The starting kit currently contains the following examples:

### Unlearning on CIFAR10

![sample images from the CIFAR10 dataset](https://github.com/unlearning-challenge/starting-kit/assets/277639/acee217a-9ecd-484b-be81-8dcf5992eece)


The notebook [`01_starter_kit.ipynb`](https://nbviewer.org/github/unlearning-challenge/starting-kit/tree/main/unlearning-CIFAR10.ipynb) provides a foundation for participants to build their unlearning models on the CIFAR-10 dataset. This jupyter notebook can be run locally, [on Colab](https://colab.research.google.com/github/unlearning-challenge/starting-kit/blob/main/unlearning-CIFAR10.ipynb), or [on Kaggle](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/unlearning-challenge/starting-kit/main/unlearning-CIFAR10.ipynb)


## Community-contributed examples

* [Translation of the CIFAR-10 notebook to Keras](https://www.kaggle.com/code/ipythonx/keras-core-machine-unlearning-challenge) by [Mohammed Innat](https://github.com/innat) 

## How to run the official score

From the repository root, open `Notebooks/04_official_score.ipynb` in Jupyter or
VS Code and run its cells. It evaluates the retain-only fine-tuning baseline
and `Kookmin_2` against retain-only models retrained from scratch, reporting
official forgetting quality F, retain/test utility, final score, runtime, N,
and the loss-based MIA diagnostic.

The notebook defaults to N=8 and a fixed 256-example forget-set subsample to
keep development runs manageable. Keep N, model seeds, and the forget sample
seed/size the same when comparing methods. Use the full forget set for final
numbers. The retrained-model cache is written to
`outputs/retrain_confs.npz`; it records the recipe, seeds, forget-index hash,
and data splits and refuses to load if any of them differ.

The starter-kit notebook documents SGD (learning rate 0.1, momentum 0.9,
weight decay 5e-4) and CIFAR-10 crop/flip augmentation for the pretrained
checkpoint, but not its training duration or schedule. The evaluation therefore
uses an explicit 30-epoch cosine-schedule scratch recipe with those documented
settings. Treat resulting utility ratios and final scores as a local reference,
not an exact reproduction of the unknown checkpoint-training recipe.
