"""CASIA-SURF (the competition's real evaluation data), mirroring the official loader.

Port of ``surf.py`` in https://github.com/google-deepmind/unlearning_evaluation
(copy in references/code/unlearning_evaluation). Same images, labels and splits:

* label = age group: ages binned into 10 equal-width groups;
* images: ``WBackground/<person>/real.rssdk/color/*.jpg`` resized to 32x32,
  kept as raw 0-255 floats (the official pipeline does NOT normalise);
* examples shuffled with ``random.Random(43)``, then 80/10/10 train/val/test;
* forget set = all training images of 15 people (``LeavePGroupsOut(15)``,
  first split), retain = the rest of train.

The dataset is licence-gated: request it from the CASIA-SURF owners (see the
official README); place ``Age_Gender.txt`` and the unzipped ``WBackground``
folder under ``root``. Decoding is slow, so the processed tensors are cached
to ``root/surf32_cache.pt`` on first use.

Note: ``torchvision.transforms.Resize`` on uint8 tensors uses antialiasing by
default in recent torchvision versions; the organisers' version may have
differed, so pixel values can differ marginally from the competition's.
"""

import glob
import os
import random

import numpy as np
import torch
from sklearn import model_selection

NUM_CLASSES = 10


def _decode(root, size=32):
    from torchvision import io
    import torchvision.transforms as T

    labels_csv = np.loadtxt(
        os.path.join(root, "Age_Gender.txt"),
        dtype={"names": ("id", "age", "gender"), "formats": ("S11", "i4", "S1")},
        delimiter=" ",
    )
    users = np.array([u.decode("utf-8") for u in np.unique(labels_csv["id"])])
    ages = np.array([a for _, a, _ in labels_csv])
    groups = np.linspace(np.min(ages), np.max(ages) + 1, 11)
    age_group = np.digitize(ages, groups) - 1
    label_of = {labels_csv[i][0].decode("utf-8"): int(age_group[i]) for i in range(len(ages))}

    resize = T.Resize((size, size))
    examples = []
    for f in sorted(glob.glob(os.path.join(root, "WBackground", "*/*"))):
        person = f.split("/")[-1]
        for img_path in sorted(glob.glob(os.path.join(f, "real.rssdk", "color", "*.jpg"))):
            img = resize(io.read_image(img_path))
            examples.append((img, int(np.where(person == users)[0][0]), label_of[person]))
    random.Random(43).shuffle(examples)
    x = torch.stack([e[0] for e in examples])           # uint8 [N,3,32,32]
    pid = torch.tensor([e[1] for e in examples])
    y = torch.tensor([e[2] for e in examples], dtype=torch.long)
    return x, y, pid


def load_surf(root):
    """Return dict with uint8 images, labels, person ids and split indices."""
    cache = os.path.join(root, "surf32_cache.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
    else:
        x, y, pid = _decode(root)
        d = {"x": x, "y": y, "pid": pid}
        torch.save(d, cache)
    n = len(d["y"])
    tr, va = int(0.8 * n), int(0.9 * n)
    train_idx = np.arange(0, tr)
    ids = d["pid"][:tr].numpy()
    lpgo = model_selection.LeavePGroupsOut(n_groups=15)
    retain_rel, forget_rel = next(lpgo.split(np.arange(tr).reshape(-1, 1), None, ids))
    d.update({
        "train": train_idx,
        "retain": train_idx[retain_rel],
        "forget": train_idx[forget_rel],
        "val": np.arange(tr, va),
        "test": np.arange(va, n),
    })
    return d
