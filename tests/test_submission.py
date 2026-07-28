import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.submission import make_submission


class TestSubmission(unittest.TestCase):
    def test_valid_submission_schema(self):
        result = make_submission([11, 12], [[0.7, 0.1, 0.2], [0.2, 0.3, 0.5]], path=False)
        self.assertEqual(result.columns.tolist(), ["id", "Status_C", "Status_CL", "Status_D"])
        np.testing.assert_allclose(result.iloc[:, 1:].sum(axis=1), 1.0)

    def test_rejects_wrong_shape_or_length(self):
        with self.assertRaises(ValueError):
            make_submission([1], [0.7, 0.1, 0.2], path=False)
        with self.assertRaises(ValueError):
            make_submission([1, 2], [[0.7, 0.1, 0.2]], path=False)

    def test_rejects_duplicate_ids_and_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            make_submission([1, 1], [[0.7, 0.1, 0.2], [0.2, 0.3, 0.5]], path=False)
        for bad in ([-0.1, 0.6, 0.5], [np.nan, 0.5, 0.5], [1.1, 0.0, -0.1]):
            with self.assertRaises(ValueError):
                make_submission([1], [bad], path=False)

    def test_writes_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "submission.csv"
            make_submission([1], [[0.7, 0.1, 0.2]], path=path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
