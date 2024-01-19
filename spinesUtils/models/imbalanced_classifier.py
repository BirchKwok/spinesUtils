import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

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

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    verbose : bool, default=True
        Enables verbose output.

    Attributes
    ----------
    meta_estimators : list of estimators
        The collection of fitted sub-estimators.

    Methods
    -------
    fit(X, y):
        Fit the model according to the given training data.

    predict(X):
        Predict class labels for samples in X.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0, n_classes=3, random_state=1)
    >>> clf = MultiClassBalanceClassifier(base_estimator=DecisionTreeClassifier(), n_classes=3, random_state=1)
    >>> clf.fit(X, y) # doctest: +NORMALIZE_WHITESPACE
    Training learner L1...
    Training learner L2...
    Training learner L3...
    >>> clf.predict(X) # doctest: +ELLIPSIS
    array([...])
    """
    def __init__(self, base_estimator, n_classes, top_classes_ratio=0.75, random_state=None, verbose=True):
        np.random.seed(random_state)

        self.base_estimator = base_estimator
        self.n_classes = n_classes
        self.top_classes_ratio = top_classes_ratio
        self.verbose = verbose
        self.meta_estimators = [clone(base_estimator) for _ in range(3)]

        self.logger = Logger(with_time=False)

    def fit(self, X, y):
        """Fit the model according to the given training data."""
        if self.verbose:
            self.logger.info("Training learner L1...")
        self.meta_estimators[0].fit(X, y)

        X_L2, y_L2 = self._prepare_dataset_L2(X, y)
        if self.verbose:
            self.logger.info("Training learner L2...")
        self.meta_estimators[1].fit(X_L2, y_L2)

        X_L3, y_L3 = self._prepare_dataset_L3(X, y)
        if self.verbose:
            self.logger.info("Training learner L3...")
        self.meta_estimators[2].fit(X_L3, y_L3)

        return self

    def predict(self, X):
        """Predict class labels for samples in X."""
        num_classes = self.n_classes
        proba_predictions = []

        for clf in self.meta_estimators:
            proba = clf.predict_proba(X)

            if proba.shape[1] < num_classes:
                # 如果某个模型返回的概率数组中缺少某些类别的概率
                # 我们创建一个新数组，并将缺失的类别概率补为0
                adjusted_proba = np.zeros((proba.shape[0], num_classes))
                adjusted_proba[:, :proba.shape[1]] = proba
                proba_predictions.append(adjusted_proba)
            else:
                proba_predictions.append(proba)

        # 现在可以安全地堆叠和计算平均概率
        proba_array = np.stack(proba_predictions, axis=0)
        avg_proba = np.mean(proba_array, axis=0)
        final_prediction = np.argmax(avg_proba, axis=1)

        return final_prediction

    def _prepare_dataset_L2(self, X, y):
        """Prepare the training data for the second meta-estimator."""
        class_counts = np.bincount(y, minlength=self.n_classes)
        top_classes = np.argsort(class_counts)[-int(self.n_classes * self.top_classes_ratio):]
        y_pred_L1 = self.meta_estimators[0].predict(X)

        top_class_indices = np.isin(y, top_classes)
        X_top, y_top = X[top_class_indices], y[top_class_indices]
        correct_indices_top = np.where(y_pred_L1[top_class_indices] == y_top)[0]
        incorrect_indices_top = np.where(y_pred_L1[top_class_indices] != y_top)[0]
        selected_indices_top = np.concatenate([correct_indices_top[:len(incorrect_indices_top)], incorrect_indices_top])

        bottom_class_indices = ~np.isin(y, top_classes)
        X_bottom, y_bottom = X[bottom_class_indices], y[bottom_class_indices]

        X_combined = np.concatenate([X_top[selected_indices_top], X_bottom], axis=0)
        y_combined = np.concatenate([y_top[selected_indices_top], y_bottom], axis=0)

        # 确保每个类别都有足够的样本
        for class_label in range(self.n_classes):
            if class_label not in np.unique(y_combined):
                # 随机从 B 中选择与 B 相同数量的样本
                class_indices = np.where(y == class_label)[0]
                sampled_indices = np.random.choice(class_indices, len(y_bottom), replace=True)
                X_sampled, y_sampled = X[sampled_indices], y[sampled_indices]
                X_combined = np.concatenate([X_combined, X_sampled], axis=0)
                y_combined = np.concatenate([y_combined, y_sampled], axis=0)

        return X_combined, y_combined

    def _prepare_dataset_L3(self, X, y):
        """Prepare the training data for the third meta-estimator."""
        y_pred_L1 = self.meta_estimators[0].predict(X)
        y_pred_L2 = self.meta_estimators[1].predict(X)
        inconsistent_indices = np.where(y_pred_L1 != y_pred_L2)[0]

        X_L3, y_L3 = X[inconsistent_indices], y[inconsistent_indices]
        # 确保每个类别都有足够的样本
        class_counts_L3 = np.bincount(y_L3, minlength=self.n_classes)
        min_class_count = np.min(class_counts_L3[np.nonzero(class_counts_L3)])
        for class_label in range(self.n_classes):
            if class_label not in np.unique(y_L3):
                # 随机选择与最少类别相同数量的样本
                class_indices = np.where(y == class_label)[0]
                sampled_indices = np.random.choice(class_indices, min_class_count, replace=True)
                X_sampled, y_sampled = X[sampled_indices], y[sampled_indices]
                X_L3 = np.concatenate([X_L3, X_sampled], axis=0)
                y_L3 = np.concatenate([y_L3, y_sampled], axis=0)

        return X_L3, y_L3
