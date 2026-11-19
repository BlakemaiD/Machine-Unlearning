"""Unlearning methods. Every module exposes

    unlearning(net, retain, forget, validation, **hparams) -> net

(the competition's interface; loaders yield ``(image, label)`` batches), plus
``INFO`` (rank / team / family) and, for new-style modules, ``DEFAULTS``
(published hyperparameters; overrides are validated by name).

Competition methods are named ``<Team>_<final rank>``; baselines are lowercase.
"""

from . import (Amnesiacs_6, Fanchuan_1, Forget_8, Kookmin_2, Sebastian_4, Seif_3, Sun_7,
               ToshiK_5, bad_teacher, boundary_expanding, boundary_shrink, finetune, l1_sparse,
               l2ul, lash, lfssd, neggrad_plus, npo, original, random_label, rum, salun, scar,
               scissorhands, scrub, sfron, ssd)

# NeurIPS 2023 Machine Unlearning final private leaderboard, places 1-8.
# (9th / 10th place write-ups are not publicly readable; see document.md.)
COMPETITION = {
    "fanchuan_1": Fanchuan_1,
    "kookmin_2": Kookmin_2,
    "seif_3": Seif_3,
    "sebastian_4": Sebastian_4,
    "toshik_5": ToshiK_5,
    "amnesiacs_6": Amnesiacs_6,
    "sun_7": Sun_7,
    "forget_8": Forget_8,
}

# Literature baselines used by the organisers' paper (arXiv:2406.09073) and
# DeepUnlearn (arXiv:2410.01276), plus the no-op reference.
BASELINES = {
    "original": original,
    "finetune": finetune,
    "neggrad_plus": neggrad_plus,
    "random_label": random_label,
    "salun": salun,
    "scrub": scrub,
    "l1_sparse": l1_sparse,
    "ssd": ssd,
}

# Further state-of-the-art methods (2023-2024), added after the first run.
SOTA = {
    "bad_teacher": bad_teacher,
    "boundary_shrink": boundary_shrink,
    "boundary_expanding": boundary_expanding,
    "lfssd": lfssd,
    "scissorhands": scissorhands,
    "scar": scar,
    "npo": npo,
    "rum": rum,
    "sfron": sfron,
    "l2ul": l2ul,
}

# Our own methods.
OURS = {
    "lash": lash,
}

MODULES = {**BASELINES, **COMPETITION, **SOTA, **OURS}
METHODS = {name: mod.unlearning for name, mod in MODULES.items()}

__all__ = ["METHODS", "MODULES", "COMPETITION", "BASELINES", "SOTA", "OURS", *sorted({m.__name__.split(".")[-1]
                                                                       for m in MODULES.values()})]
