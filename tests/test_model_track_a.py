import unittest

import numpy as np
import pandas as pd

from src.model_track_a import (
    build_knn_pipeline,
    build_logreg_pipeline,
    validate_fold_assignments,
)


class TrackAPipelineTests(unittest.TestCase):
    @staticmethod
    def sample_data(n=60):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(
            {
                "id": np.arange(n),
                "Followup_Days": rng.integers(10, 2000, n),
                "Treatment_Assignment": rng.choice(["Treatment_A", "Treatment_B"], n),
                "Patient_Age_Days": rng.integers(10000, 30000, n),
                "Patient_Sex": rng.choice(["Female", "Male"], n),
                "Ascites_Indicator": rng.choice(["Absent", "Present"], n),
                "Liver_Enlargement": rng.choice(["Absent", "Present"], n),
                "Spider_Angioma": rng.choice(["Absent", "Present"], n),
                "Edema_Status": rng.choice(["No", "Yes", None], n),
                "Bilirubin_Level": rng.uniform(0.1, 10, n),
                "Cholesterol_Level": rng.uniform(100, 500, n),
                "Albumin_Level": rng.uniform(1.5, 5, n),
                "Copper_Level": rng.uniform(5, 200, n),
                "Alkaline_Phosphatase": rng.uniform(100, 8000, n),
                "AST_Level": rng.uniform(10, 300, n),
                "Triglyceride_Level": rng.uniform(30, 300, n),
                "Platelet_Count": rng.uniform(50, 500, n),
                "Prothrombin_Time": rng.uniform(8, 20, n),
                "Clinical_Stage": rng.integers(1, 5, n),
            }
        )
        frame.loc[::7, "Copper_Level"] = np.nan
        target = pd.Series(np.tile([0, 1, 2], n // 3))
        return frame, target

    def test_both_models_return_valid_multiclass_probabilities(self):
        X, y = self.sample_data()
        for model in [build_knn_pipeline(), build_logreg_pipeline()]:
            model.fit(X, y)
            proba = model.predict_proba(X.iloc[:5])
            self.assertEqual(proba.shape, (5, 3))
            self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_fold_validation_requires_ten_folds(self):
        validate_fold_assignments(pd.Series(np.tile(np.arange(10), 2)))
        with self.assertRaises(ValueError):
            validate_fold_assignments(pd.Series(np.tile(np.arange(5), 2)))


if __name__ == "__main__":
    unittest.main()
