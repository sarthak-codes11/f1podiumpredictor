"""
encoders.py
-----------
Shared encoder class used by both features.py and predict.py.
Must live in its own module so pickle can resolve it consistently
regardless of which module loads the saved encoders.pkl file.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder


class SafeLabelEncoder(LabelEncoder):
    """
    LabelEncoder that handles unseen labels at prediction time.
    Maps any unknown label to 'unknown' instead of crashing.
    """
    def fit(self, y):
        y = list(y) + ['unknown']  # always register 'unknown' as valid
        return super().fit(y)

    def transform(self, y):
        known = set(self.classes_)
        y_safe = [label if label in known else 'unknown' for label in y]
        return super().transform(y_safe)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)