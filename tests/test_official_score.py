import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from unlearn.evaluate import (
    EvaluationResults,
    TrainingRecipe,
    make_forget_loader,
    run_method,
    score,
    train_or_load_retrained,
)
from unlearn.metrics import accuracy_eval, get_forget_confs
from unlearn.official_metric import forget_score, logit_scaled_confidence


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, inputs):
        return self.linear(inputs)


class OfficialMetricTests(unittest.TestCase):
    def test_separated_distributions_have_zero_score(self):
        rng = np.random.default_rng(42)
        retrained = rng.normal(size=(32, 6))
        unlearned = retrained + 8.0
        self.assertEqual(forget_score(unlearned, retrained), 0.0)

    def test_retrained_reference_scores_above_finetune_baseline(self):
        rng = np.random.default_rng(42)
        retrained = rng.normal(size=(32, 6))
        independently_retrained = rng.normal(size=(32, 6))
        finetune_baseline = independently_retrained + 8.0
        self.assertGreater(
            forget_score(independently_retrained, retrained),
            forget_score(finetune_baseline, retrained),
        )

    def test_confidence_uses_correct_class_probability(self):
        logits = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        confidence = logit_scaled_confidence(logits, np.array([0, 1]))
        np.testing.assert_allclose(confidence, np.log(1.0 / (np.exp(-2.0))), atol=1e-12)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        self.features = torch.randn(32, 2, generator=generator)
        self.labels = torch.arange(32) % 2
        self.dataset = TensorDataset(self.features, self.labels)
        self.retain = DataLoader(
            TensorDataset(self.features[:12], self.labels[:12]),
            batch_size=4,
            shuffle=True,
        )
        self.forget = DataLoader(
            TensorDataset(self.features[12:], self.labels[12:]),
            batch_size=4,
            shuffle=True,
        )
        self.test = DataLoader(self.dataset, batch_size=4, shuffle=False)
        self.val = DataLoader(self.dataset, batch_size=4, shuffle=False)

    def test_evaluation_helpers_restore_mode_and_preserve_order(self):
        model = TinyClassifier()
        model.train()
        accuracy_eval(model, self.test)
        self.assertTrue(model.training)
        loader = make_forget_loader(self.forget, max_examples=None, sample_seed=2)
        self.assertEqual(len(get_forget_confs(model, loader)), len(self.forget.dataset))
        ordered_labels = torch.cat([targets for _, targets in loader])
        torch.testing.assert_close(ordered_labels, self.labels[12:])
        self.assertTrue(model.training)

    def test_final_score_and_efficiency_ratios(self):
        confs = np.tile(np.linspace(-1.0, 1.0, 32)[:, None], (1, 2))
        seeds = tuple(range(32))
        retrained = EvaluationResults(
            name="retrained",
            seeds=seeds,
            forget_confs=confs,
            retain_accs=np.full(32, 0.8),
            test_accs=np.full(32, 0.6),
            forget_accs=np.full(32, 0.2),
            runtimes=np.full(32, 10.0),
            forget_indices_hash="same",
            forget_sample_seed=4,
        )
        unlearned = EvaluationResults(
            name="same-confidence reference",
            seeds=seeds,
            forget_confs=confs.copy(),
            retain_accs=np.full(32, 0.4),
            test_accs=np.full(32, 0.3),
            forget_accs=np.full(32, 0.1),
            runtimes=np.full(32, 1.0),
            forget_indices_hash="same",
            forget_sample_seed=4,
        )
        result = score(unlearned, retrained)
        self.assertAlmostEqual(result["F"], 1.0)
        self.assertAlmostEqual(result["final_score"], 0.25)
        self.assertAlmostEqual(result["runtime_ratio"], 0.1)

    def test_run_method_and_cache_metadata(self):
        def identity_method(net, retain, forget, validation):
            return net.eval()

        model = TinyClassifier()
        seeds = list(range(3, 11))
        results = run_method(
            identity_method,
            8,
            seeds,
            original_model=model,
            retain_loader=self.retain,
            forget_loader=self.forget,
            val_loader=self.val,
            test_loader=self.test,
            max_forget_examples=20,
            forget_sample_seed=9,
        )
        self.assertEqual(results.forget_confs.shape, (8, 20))
        self.assertEqual(results.mia_accs.shape, (8,))

        recipe = TrainingRecipe(
            epochs=1,
            batch_size=4,
            num_workers=0,
            scheduler="none",
            augment=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "retrain_confs.npz"
            retrained = train_or_load_retrained(
                8,
                recipe,
                retain_loader=self.retain,
                forget_loader=self.forget,
                test_loader=self.test,
                seeds=seeds,
                cache_path=cache_path,
                max_forget_examples=20,
                forget_sample_seed=9,
                model_factory=TinyClassifier,
            )
            self.assertEqual(retrained.forget_confs.shape, (8, 20))
            loaded = train_or_load_retrained(
                8,
                recipe,
                retain_loader=self.retain,
                forget_loader=self.forget,
                test_loader=self.test,
                seeds=seeds,
                cache_path=cache_path,
                max_forget_examples=20,
                forget_sample_seed=9,
                model_factory=TinyClassifier,
            )
            np.testing.assert_array_equal(loaded.forget_confs, retrained.forget_confs)
            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                train_or_load_retrained(
                    8,
                    recipe,
                    retain_loader=self.retain,
                    forget_loader=self.forget,
                    test_loader=self.test,
                    seeds=list(range(4, 12)),
                    cache_path=cache_path,
                    max_forget_examples=20,
                    forget_sample_seed=9,
                    model_factory=TinyClassifier,
                )


if __name__ == "__main__":
    unittest.main()
