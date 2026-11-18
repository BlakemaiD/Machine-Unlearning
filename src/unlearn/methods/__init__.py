from . import Amnesiacs_6, Fauchan_1, Kookmin_2, Seif_3, finetune

METHODS = {
    "Amnesiacs_6": Amnesiacs_6.unlearning,
    "Fauchan_1": Fauchan_1.unlearning,
    "Kookmin_2": Kookmin_2.unlearning,
    "Seif_3": Seif_3.unlearning,
    "finetune": finetune.unlearning,
}

__all__ = [
    "Amnesiacs_6",
    "Fauchan_1",
    "Kookmin_2",
    "METHODS",
    "Seif_3",
    "finetune",
]
