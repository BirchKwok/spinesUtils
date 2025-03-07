import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import balanced_accuracy_score

from spinesUtils.logging import Logger


class MultiClassBalanceClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom classifier that balances multi-class datasets by training separate models on different subsets of the data.

    The classifier creates three different meta-estimators:
    - L1 is trained on the original dataset.
    - L2 is trained on a balanced subset where the number of examples of the top classes (by count) is reduced.
    - L3 is trained on the examples where L1 and L2 predictions are inconsistent.

    Parameters
    ----------
    base_estimator : estimator object
        The base estimator from which the ensemble is built.

    n_classes : int
        The number of classes in the target variable.

    top_classes_ratio : float, default=0.75
        The ratio of the top classes to include in the second level training.

    min_samples_per_class : int, default=10
        The minimum number of samples per class to ensure in L2 and L3 datasets.

    weights : list of float, default=None
        Custom weights for each meta-estimator in prediction. If None, weights
        will be determined adaptively based on validation performance.

    early_stopping_threshold : float, default=0.95
        If the agreement between L1 and L2 is above this threshold, L3 training will be skipped.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    validation_fraction : float, default=0.1
        Fraction of training data to set aside for estimator performance evaluation.

    verbose : bool, default=True
        Enables verbose output.

    Attributes
    ----------
    meta_estimators : list of estimators
        The collection of fitted sub-estimators.

    estimator_weights_ : array of shape (3,)
        The weight given to each estimator in the ensemble.

    estimator_scores_ : array of shape (3,)
        The performance scores of the estimators (balanced accuracy).

    classes_ : array of shape (n_classes,)
        The class labels.

    Methods
    -------
    fit(X, y):
        Fit the model according to the given training data.

    predict(X):
        Predict class labels for samples in X.

    predict_proba(X):
        Predict class probabilities for samples in X.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0, n_classes=3, random_state=1)
    >>> clf = MultiClassBalanceClassifier(base_estimator=DecisionTreeClassifier(), n_classes=3, random_state=1)
    >>> clf.fit(X, y)
    Training learner L1...
    Training learner L2...
    Training learner L3...
    >>> clf.predict(X)
    array([...])
    """
    def __init__(self, base_estimator, n_classes, top_classes_ratio=0.75, min_samples_per_class=10,
                 weights=None, early_stopping_threshold=0.95, random_state=None, 
                 validation_fraction=0.1, verbose=True):
        self.base_estimator = base_estimator
        self.n_classes = n_classes
        self.top_classes_ratio = top_classes_ratio
        self.min_samples_per_class = min_samples_per_class
        self.weights = weights
        self.early_stopping_threshold = early_stopping_threshold
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        
        self.meta_estimators = [clone(base_estimator) for _ in range(3)]
        self.estimator_weights_ = None
        self.estimator_scores_ = None
        self.classes_ = None
        self.logger = Logger(with_time=False)

    def fit(self, X, y):
        """Fit the model according to the given training data."""
        # Store class labels
        self.classes_ = np.unique(y)
        
        # Split data for validation if needed
        if self.weights is None:
            rng = np.random.RandomState(self.random_state)
            val_mask = rng.rand(X.shape[0]) < self.validation_fraction
            X_train, X_val = X[~val_mask], X[val_mask]
            y_train, y_val = y[~val_mask], y[val_mask]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Train L1 on the original dataset
        if self.verbose:
            self.logger.info("Training learner L1...")
        self.meta_estimators[0].fit(X_train, y_train)

        # Train L2 on a balanced dataset
        X_L2, y_L2 = self._prepare_dataset_L2(X_train, y_train)
        if self.verbose:
            self.logger.info("Training learner L2...")
        self.meta_estimators[1].fit(X_L2, y_L2)

        # Check agreement between L1 and L2
        if X_val is not None:
            y_pred_L1 = self.meta_estimators[0].predict(X_val)
            y_pred_L2 = self.meta_estimators[1].predict(X_val)
            agreement = np.mean(y_pred_L1 == y_pred_L2)
            
            if self.verbose:
                self.logger.info(f"Agreement between L1 and L2: {agreement:.4f}")
        else:
            # Use training data to estimate agreement if no validation data
            y_pred_L1 = self.meta_estimators[0].predict(X_train)
            y_pred_L2 = self.meta_estimators[1].predict(X_train)
            agreement = np.mean(y_pred_L1 == y_pred_L2)

        # Skip L3 training if agreement is high enough
        skip_L3 = agreement > self.early_stopping_threshold
        
        if not skip_L3:
            # Train L3 on samples where L1 and L2 disagree
            X_L3, y_L3 = self._prepare_dataset_L3(X_train, y_train)
            if X_L3.shape[0] > 0:
                if self.verbose:
                    self.logger.info("Training learner L3...")
                self.meta_estimators[2].fit(X_L3, y_L3)
            else:
                if self.verbose:
                    self.logger.info("Skipping L3 training (no disagreements)")
                skip_L3 = True
        else:
            if self.verbose:
                self.logger.info("Skipping L3 training (high agreement)")

        # Compute estimator weights based on validation performance
        if self.weights is None and X_val is not None and y_val is not None:
            self.estimator_scores_ = np.zeros(3)
            
            for i, estimator in enumerate(self.meta_estimators[:2 if skip_L3 else 3]):
                y_pred = estimator.predict(X_val)
                self.estimator_scores_[i] = balanced_accuracy_score(y_val, y_pred)
            
            if skip_L3:
                self.estimator_scores_[2] = 0
            
            if self.verbose:
                self.logger.info(f"Estimator scores: {self.estimator_scores_}")
            
            # Compute weights based on performance
            self.estimator_weights_ = self.estimator_scores_ / np.sum(self.estimator_scores_[:2 if skip_L3 else 3])
            
            if skip_L3:
                self.estimator_weights_[2] = 0
        else:
            # Use user-defined weights or equal weights
            self.estimator_weights_ = self.weights if self.weights is not None else np.ones(3) / (2 if skip_L3 else 3)
            
            if skip_L3:
                self.estimator_weights_[2] = 0
                # Renormalize weights
                if self.weights is None:
                    self.estimator_weights_[:2] = self.estimator_weights_[:2] / np.sum(self.estimator_weights_[:2])

        if self.verbose:
            self.logger.info(f"Final estimator weights: {self.estimator_weights_}")

        return self

    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        num_classes = self.n_classes
        proba_predictions = []

        for i, clf in enumerate(self.meta_estimators):
            # Skip L3 if its weight is 0
            if self.estimator_weights_[i] == 0:
                continue
                
            proba = clf.predict_proba(X)

            if proba.shape[1] < num_classes:
                # if some classes are missing from the probability arrays returned by some models
                # we create a new array and fill the missing class probabilities with zeros
                adjusted_proba = np.zeros((proba.shape[0], num_classes))
                for j, class_label in enumerate(clf.classes_):
                    adjusted_proba[:, class_label] = proba[:, j]
                proba_predictions.append((adjusted_proba, self.estimator_weights_[i]))
            else:
                proba_predictions.append((proba, self.estimator_weights_[i]))

        # Compute weighted average of probabilities
        avg_proba = np.zeros((X.shape[0], num_classes))
        total_weight = 0
        
        for proba, weight in proba_predictions:
            avg_proba += proba * weight
            total_weight += weight
            
        if total_weight > 0:
            avg_proba /= total_weight
            
        return avg_proba

    def predict(self, X):
        """Predict class labels for samples in X."""
        avg_proba = self.predict_proba(X)
        final_prediction = np.argmax(avg_proba, axis=1)
        return final_prediction

    def _prepare_dataset_L2(self, X, y):
        """Prepare the training data for the second meta-estimator."""
        rng = np.random.RandomState(self.random_state)
        class_counts = np.bincount(y, minlength=self.n_classes)
        
        # Determine the top classes by frequency
        top_class_count = int(self.n_classes * self.top_classes_ratio)
        top_classes = np.argsort(class_counts)[-top_class_count:] if top_class_count > 0 else []
        
        # Get predictions from L1
        y_pred_L1 = self.meta_estimators[0].predict(X)
        
        # Initialize lists to store selected samples
        X_selected, y_selected = [], []
        
        # Process each class separately
        for class_label in range(self.n_classes):
            class_indices = np.where(y == class_label)[0]
            
            if len(class_indices) == 0:
                continue
                
            # Determine target sample count based on class frequency
            if class_label in top_classes:
                # For top classes, select a balanced subset
                min_class_count = np.min(class_counts[class_counts > 0])
                target_count = max(min_class_count, self.min_samples_per_class)
            else:
                # For minority classes, keep all samples
                target_count = len(class_indices)
                
            # Split into correctly and incorrectly classified samples
            correct_indices = class_indices[y_pred_L1[class_indices] == class_label]
            incorrect_indices = class_indices[y_pred_L1[class_indices] != class_label]
            
            # Prioritize incorrectly classified samples, then add correctly classified if needed
            if len(incorrect_indices) >= target_count:
                selected_indices = rng.choice(incorrect_indices, target_count, replace=False)
            else:
                selected_indices = incorrect_indices
                remaining_count = target_count - len(incorrect_indices)
                
                if len(correct_indices) > 0 and remaining_count > 0:
                    additional_indices = rng.choice(
                        correct_indices,
                        min(remaining_count, len(correct_indices)),
                        replace=False
                    )
                    selected_indices = np.concatenate([selected_indices, additional_indices])
            
            # Add selected samples for this class
            X_selected.append(X[selected_indices])
            y_selected.append(y[selected_indices])
                
        # Combine all selected samples
        if len(X_selected) > 0:
            X_combined = np.vstack(X_selected)
            y_combined = np.concatenate(y_selected)
            
            # Shuffle the combined dataset
            shuffle_idx = rng.permutation(len(y_combined))
            X_combined = X_combined[shuffle_idx]
            y_combined = y_combined[shuffle_idx]
            
            return X_combined, y_combined
        else:
            return X, y

    def _prepare_dataset_L3(self, X, y):
        """Prepare the training data for the third meta-estimator."""
        rng = np.random.RandomState(self.random_state)
        
        # Get predictions from L1 and L2
        y_pred_L1 = self.meta_estimators[0].predict(X)
        y_pred_L2 = self.meta_estimators[1].predict(X)
        
        # Find samples where L1 and L2 disagree
        inconsistent_indices = np.where(y_pred_L1 != y_pred_L2)[0]
        
        if len(inconsistent_indices) == 0:
            return np.array([]), np.array([])
            
        X_inconsistent = X[inconsistent_indices]
        y_inconsistent = y[inconsistent_indices]
        
        # Group by class
        X_by_class, y_by_class = [], []
        min_class_count = self.min_samples_per_class
        
        for class_label in range(self.n_classes):
            class_indices = np.where(y_inconsistent == class_label)[0]
            
            if len(class_indices) > 0:
                X_by_class.append(X_inconsistent[class_indices])
                y_by_class.append(y_inconsistent[class_indices])
                min_class_count = min(min_class_count, len(class_indices))
            else:
                # For classes not in the inconsistent set, sample from original dataset
                orig_class_indices = np.where(y == class_label)[0]
                
                if len(orig_class_indices) > 0:
                    sampled_indices = rng.choice(
                        orig_class_indices,
                        min(self.min_samples_per_class, len(orig_class_indices)),
                        replace=False
                    )
                    X_by_class.append(X[sampled_indices])
                    y_by_class.append(y[sampled_indices])
        
        # Combine and balance classes
        X_combined, y_combined = [], []
        
        for i, (X_class, y_class) in enumerate(zip(X_by_class, y_by_class)):
            # Limit samples per class for better balance
            if len(X_class) > min_class_count * 2:
                indices = rng.choice(len(X_class), min_class_count * 2, replace=False)
                X_combined.append(X_class[indices])
                y_combined.append(y_class[indices])
            else:
                X_combined.append(X_class)
                y_combined.append(y_class)
        
        if len(X_combined) > 0:
            X_result = np.vstack(X_combined)
            y_result = np.concatenate(y_combined)
            
            # Shuffle the final dataset
            shuffle_idx = rng.permutation(len(y_result))
            X_result = X_result[shuffle_idx]
            y_result = y_result[shuffle_idx]
            
            return X_result, y_result
        else:
            return np.array([]), np.array([])
    
    def score(self, X, y):
        """Return the balanced accuracy score on the given test data and labels."""
        return balanced_accuracy_score(y, self.predict(X))
    
    def get_estimator_importances(self):
        """Return a dict with the importance of each estimator in the ensemble."""
        if self.estimator_weights_ is None:
            raise ValueError("Estimator not fitted yet. Call 'fit' before using this method.")
        
        return {
            "L1_weight": self.estimator_weights_[0],
            "L2_weight": self.estimator_weights_[1],
            "L3_weight": self.estimator_weights_[2],
            "L1_score": self.estimator_scores_[0] if self.estimator_scores_ is not None else None,
            "L2_score": self.estimator_scores_[1] if self.estimator_scores_ is not None else None,
            "L3_score": self.estimator_scores_[2] if self.estimator_scores_ is not None else None
        }
