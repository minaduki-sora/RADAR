import unittest

import torch

from eagle.model.eye import RADAR


class RadarPolicyTest(unittest.TestCase):
    def test_forward_shapes(self):
        model = RADAR()
        logits, (hidden, cell) = model(torch.randn(2, 7, 10))

        self.assertEqual(logits.shape, (2, 7, 2))
        self.assertEqual(hidden.shape, (1, 2, 128))
        self.assertEqual(cell.shape, (1, 2, 128))

    def test_reset_hidden_uses_model_dtype(self):
        model = RADAR().to(dtype=torch.float64)
        hidden, cell = model.reset_hidden(batch_size=3)

        self.assertEqual(hidden.shape, (1, 3, 128))
        self.assertEqual(cell.shape, (1, 3, 128))
        self.assertEqual(hidden.dtype, torch.float64)
        self.assertEqual(cell.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
