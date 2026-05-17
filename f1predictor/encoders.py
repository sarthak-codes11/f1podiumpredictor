"""
encoders.py
-----------
Shared encoder class used by both features.py and predict.py.
Lives in its own module so pickle resolves it consistently
regardless of which module loads encoders.pkl.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder


class SafeLabelEncoder(LabelEncoder):
    """
    LabelEncoder that handles unseen labels at prediction time.
    Maps any unknown label to 'unknown' instead of crashing.
    """
    def fit(self, y):
        y = list(y) + ['unknown']
        return super().fit(y)

    def transform(self, y):
        known = set(self.classes_)
        y_safe = [label if label in known else 'unknown' for label in y]
        return super().transform(y_safe)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)