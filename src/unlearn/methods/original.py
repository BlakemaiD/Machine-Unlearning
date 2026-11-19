"""Reference: no unlearning (returns the original model unchanged).

Evaluated like any method, it gives the "do nothing" end of every metric. Its
N runs are identical, so the competition metric assigns it the worst
forgetting quality, a useful sanity check of the evaluation pipeline.
"""

from ..device import DEVICE
from ..utils import merge_hparams

INFO = {"rank": None, "team": "reference", "family": "no-op"}
DEFAULTS = {}


def unlearning(net, retain, forget, validation, **hparams):
    merge_hparams(DEFAULTS, hparams, "original")
    return net.to(DEVICE).eval()
