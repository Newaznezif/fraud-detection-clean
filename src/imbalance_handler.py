from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from collections import Counter

def handle_imbalance(
    X,
    y,
    strategy="smote",
    random_state=42,
    sampling_ratio=0.5
):
    """
    Handle class imbalance using different strategies.

    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    strategy : str
        One of ['smote', 'undersample', 'none']
    random_state : int
        Random seed
    sampling_ratio : float
        Sampling ratio for minority class (used in SMOTE / undersampling)

    Returns
    -------
    X_resampled, y_resampled
    """

    if strategy == "smote":
        # Determine minority class count
        counter = Counter(y)
        minority_count = min(counter.values())
        k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

        sampler = SMOTE(
            sampling_strategy=sampling_ratio,
            random_state=random_state,
            k_neighbors=k_neighbors
        )

    elif strategy == "undersample":
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_ratio,
            random_state=random_state
        )

    elif strategy == "none":
        return X, y

    else:
        raise ValueError(
            "Invalid strategy. Choose from ['smote', 'undersample', 'none']"
        )

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled



