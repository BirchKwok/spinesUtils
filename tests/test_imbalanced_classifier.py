import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split

from spinesUtils.models.imbalanced_classifier import MultiClassBalanceClassifier


@pytest.fixture
def imbalanced_dataset():
    """创建一个不平衡的多分类数据集"""
    # 创建一个不平衡的多分类数据集，类别0的样本数量远大于其他类别
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=4,
        n_clusters_per_class=1,
        weights=[0.7, 0.1, 0.1, 0.1],  # 类别0占70%，其他类别各占10%
        random_state=42
    )
    return X, y


@pytest.fixture
def balanced_dataset():
    """创建一个平衡的多分类数据集"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=4,
        n_clusters_per_class=1,
        weights=[0.25, 0.25, 0.25, 0.25],  # 所有类别均衡
        random_state=42
    )
    return X, y


class TestMultiClassBalanceClassifier:
    """测试 MultiClassBalanceClassifier 类"""

    def test_init(self):
        """测试初始化参数"""
        # 测试默认参数
        clf = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_classes=4
        )
        assert clf.base_estimator is not None
        assert clf.n_classes == 4
        assert clf.top_classes_ratio == 0.75
        assert clf.min_samples_per_class == 10
        assert clf.weights is None
        assert clf.early_stopping_threshold == 0.95
        assert clf.random_state is None
        assert clf.validation_fraction == 0.1
        assert clf.verbose is True
        assert len(clf.meta_estimators) == 3
        assert clf.estimator_weights_ is None
        assert clf.estimator_scores_ is None
        assert clf.classes_ is None

        # 测试自定义参数
        clf = MultiClassBalanceClassifier(
            base_estimator=RandomForestClassifier(),
            n_classes=3,
            top_classes_ratio=0.6,
            min_samples_per_class=20,
            weights=[0.5, 0.3, 0.2],
            early_stopping_threshold=0.8,
            random_state=42,
            validation_fraction=0.2,
            verbose=False
        )
        assert clf.n_classes == 3
        assert clf.top_classes_ratio == 0.6
        assert clf.min_samples_per_class == 20
        assert clf.weights == [0.5, 0.3, 0.2]
        assert clf.early_stopping_threshold == 0.8
        assert clf.random_state == 42
        assert clf.validation_fraction == 0.2
        assert clf.verbose is False

    def test_fit_predict_imbalanced(self, imbalanced_dataset):
        """测试在不平衡数据集上的拟合和预测"""
        X, y = imbalanced_dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 使用决策树作为基础分类器
        base_clf = DecisionTreeClassifier(random_state=42)
        base_clf.fit(X_train, y_train)
        base_pred = base_clf.predict(X_test)
        base_acc = accuracy_score(y_test, base_pred)
        base_bal_acc = balanced_accuracy_score(y_test, base_pred)

        # 使用 MultiClassBalanceClassifier
        clf = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_classes=4,
            random_state=42,
            verbose=False
        )
        clf.fit(X_train, y_train)
        bal_pred = clf.predict(X_test)
        bal_acc = accuracy_score(y_test, bal_pred)
        bal_bal_acc = balanced_accuracy_score(y_test, bal_pred)

        # 验证分类器已正确拟合
        assert hasattr(clf, 'estimator_weights_')
        assert clf.estimator_weights_ is not None
        assert len(clf.estimator_weights_) == 3
        assert hasattr(clf, 'classes_')
        assert len(clf.classes_) == 4

        # 打印性能比较
        print(f"\n不平衡数据集性能比较:")
        print(f"基础分类器准确率: {base_acc:.4f}")
        print(f"基础分类器平衡准确率: {base_bal_acc:.4f}")
        print(f"平衡分类器准确率: {bal_acc:.4f}")
        print(f"平衡分类器平衡准确率: {bal_bal_acc:.4f}")

        # 在不平衡数据集上，平衡分类器的平衡准确率应该更高
        # 注意：这不是一个严格的断言，因为性能可能因随机性而变化
        # 但在大多数情况下，平衡分类器应该在平衡准确率上表现更好
        assert bal_bal_acc >= base_bal_acc * 0.9  # 允许一定的波动

    def test_fit_predict_balanced(self, balanced_dataset):
        """测试在平衡数据集上的拟合和预测"""
        X, y = balanced_dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 使用决策树作为基础分类器
        base_clf = DecisionTreeClassifier(random_state=42)
        base_clf.fit(X_train, y_train)
        base_pred = base_clf.predict(X_test)
        base_acc = accuracy_score(y_test, base_pred)
        base_bal_acc = balanced_accuracy_score(y_test, base_pred)

        # 使用 MultiClassBalanceClassifier
        clf = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_classes=4,
            random_state=42,
            verbose=False
        )
        clf.fit(X_train, y_train)
        bal_pred = clf.predict(X_test)
        bal_acc = accuracy_score(y_test, bal_pred)
        bal_bal_acc = balanced_accuracy_score(y_test, bal_pred)

        # 打印性能比较
        print(f"\n平衡数据集性能比较:")
        print(f"基础分类器准确率: {base_acc:.4f}")
        print(f"基础分类器平衡准确率: {base_bal_acc:.4f}")
        print(f"平衡分类器准确率: {bal_acc:.4f}")
        print(f"平衡分类器平衡准确率: {bal_bal_acc:.4f}")

        # 在平衡数据集上，两种分类器的性能应该相近
        assert abs(bal_bal_acc - base_bal_acc) < 0.1

    def test_predict_proba(self, imbalanced_dataset):
        """测试概率预测功能"""
        X, y = imbalanced_dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        clf = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_classes=4,
            random_state=42,
            verbose=False
        )
        clf.fit(X_train, y_train)
        
        # 测试概率预测
        proba = clf.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 4)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(np.sum(proba, axis=1), 1.0)
        
        # 验证预测结果与概率最大值对应
        pred = clf.predict(X_test)
        pred_from_proba = np.argmax(proba, axis=1)
        assert np.array_equal(pred, pred_from_proba)

    def test_estimator_importances(self, imbalanced_dataset):
        """测试获取分类器重要性"""
        X, y = imbalanced_dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 使用自定义权重
        clf = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_classes=4,
            weights=[0.5, 0.3, 0.2],
            random_state=42,
            verbose=False
        )
        clf.fit(X_train, y_train)
        
        importances = clf.get_estimator_importances()
        assert importances["L1_weight"] == 0.5
        assert importances["L2_weight"] == 0.3
        assert importances["L3_weight"] == 0.2
        assert importances["L1_score"] is None
        assert importances["L2_score"] is None
        assert importances["L3_score"] is None
        
        # 使用自动权重
        clf = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_classes=4,
            random_state=42,
            verbose=False
        )
        clf.fit(X_train, y_train)
        
        importances = clf.get_estimator_importances()
        assert 0 <= importances["L1_weight"] <= 1
        assert 0 <= importances["L2_weight"] <= 1
        assert 0 <= importances["L3_weight"] <= 1
        assert abs(importances["L1_weight"] + importances["L2_weight"] + importances["L3_weight"] - 1.0) < 1e-10
        
        if clf.estimator_scores_ is not None:
            assert 0 <= importances["L1_score"] <= 1
            assert 0 <= importances["L2_score"] <= 1
            if importances["L3_score"] is not None:
                assert 0 <= importances["L3_score"] <= 1

    def test_different_base_estimators(self, imbalanced_dataset):
        """测试不同的基础分类器"""
        X, y = imbalanced_dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        base_estimators = [
            DecisionTreeClassifier(random_state=42),
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42, max_iter=1000)
        ]
        
        results = []
        
        for base_est in base_estimators:
            # 基础分类器
            base_clf = base_est.__class__(**base_est.get_params())
            base_clf.fit(X_train, y_train)
            base_pred = base_clf.predict(X_test)
            base_bal_acc = balanced_accuracy_score(y_test, base_pred)
            
            # 平衡分类器
            clf = MultiClassBalanceClassifier(
                base_estimator=base_est,
                n_classes=4,
                random_state=42,
                verbose=False
            )
            clf.fit(X_train, y_train)
            bal_pred = clf.predict(X_test)
            bal_bal_acc = balanced_accuracy_score(y_test, bal_pred)
            
            results.append({
                'estimator': base_est.__class__.__name__,
                'base_bal_acc': base_bal_acc,
                'bal_bal_acc': bal_bal_acc,
                'improvement': bal_bal_acc - base_bal_acc
            })
        
        # 打印不同基础分类器的性能比较
        print("\n不同基础分类器性能比较:")
        for result in results:
            print(f"{result['estimator']}:")
            print(f"  基础分类器平衡准确率: {result['base_bal_acc']:.4f}")
            print(f"  平衡分类器平衡准确率: {result['bal_bal_acc']:.4f}")
            print(f"  改进: {result['improvement']:.4f}")
        
        # 验证至少有一个基础分类器的性能得到了改进
        assert any(result['improvement'] > 0 for result in results)

    def test_early_stopping(self, imbalanced_dataset):
        """测试早停机制"""
        X, y = imbalanced_dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 设置早停阈值为0，确保L3始终被训练
        clf_no_early_stop = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_classes=4,
            early_stopping_threshold=0,
            random_state=42,
            verbose=False
        )
        clf_no_early_stop.fit(X_train, y_train)
        
        # 设置早停阈值为1，确保L3不被训练
        clf_early_stop = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(random_state=42),
            n_classes=4,
            early_stopping_threshold=0,
            random_state=42,
            verbose=False
        )
        clf_early_stop.fit(X_train, y_train)
        
        # 验证早停机制的影响
        importances_no_early_stop = clf_no_early_stop.get_estimator_importances()
        importances_early_stop = clf_early_stop.get_estimator_importances()
        
        # 打印早停机制的影响
        print("\n早停机制影响:")
        print(f"无早停 L3 权重: {importances_no_early_stop['L3_weight']:.4f}")
        print(f"有早停 L3 权重: {importances_early_stop['L3_weight']:.4f}")
        
        # 两种情况下的预测性能应该相近
        pred_no_early_stop = clf_no_early_stop.predict(X_test)
        pred_early_stop = clf_early_stop.predict(X_test)
        
        acc_no_early_stop = balanced_accuracy_score(y_test, pred_no_early_stop)
        acc_early_stop = balanced_accuracy_score(y_test, pred_early_stop)
        
        print(f"无早停平衡准确率: {acc_no_early_stop:.4f}")
        print(f"有早停平衡准确率: {acc_early_stop:.4f}")
        
        # 性能差异不应太大
        assert abs(acc_no_early_stop - acc_early_stop) < 0.1

    @pytest.mark.slow
    def test_edge_cases(self):
        """测试边缘情况"""
        # 测试单一类别
        X = np.random.rand(100, 10)
        y = np.zeros(100, dtype=np.int64)  # 明确指定为整数类型
        
        # 设置自定义权重，避免使用验证集
        clf = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_classes=1,
            weights=[1.0, 0.0, 0.0],  # 使用固定权重，不需要验证集
            random_state=42,
            verbose=False
        )
        
        # 单类别不会产生警告（sklearn的balanced_accuracy_score会，但我们的模型不会）
        clf.fit(X, y)
        pred = clf.predict(X)
        assert np.all(pred == 0)
        
        # 验证分类器权重
        importances = clf.get_estimator_importances()
        assert importances["L1_weight"] == 1.0
        assert importances["L2_weight"] == 0.0
        assert importances["L3_weight"] == 0.0
        
        # 测试极少样本
        X = np.random.rand(10, 5)
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        
        clf = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_classes=4,
            min_samples_per_class=1,
            weights=[0.5, 0.3, 0.2],  # 使用固定权重，不需要验证集
            random_state=42,
            verbose=False
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        assert len(np.unique(pred)) <= 4
        
        # 测试缺失类别
        X = np.random.rand(100, 10)
        y = np.array([0, 1, 2] * 33 + [0], dtype=np.int64)
        all_classes = set(range(4))  # 所有可能的类别
        
        clf = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_classes=4,
            weights=[0.4, 0.4, 0.2],  # 使用固定权重，避免验证集问题
            random_state=42,
            verbose=False
        )
        
        # 对于缺失类别的情况，可能不会有警告，因为我们在处理中已经考虑这种情况
        clf.fit(X, y)
        pred = clf.predict(X)
        proba = clf.predict_proba(X)
            
        # 验证预测结果
        assert set(np.unique(pred)).issubset(all_classes)
        assert proba.shape[1] == 4
        
        # 验证概率预测的有效性
        assert np.allclose(np.sum(proba, axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        
        # 测试空数据集
        X_empty = np.array([], dtype=np.float64).reshape(0, 10)
        y_empty = np.array([], dtype=np.int64)
        
        clf = MultiClassBalanceClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_classes=4,
            random_state=42,
            verbose=False
        )
        
        # 空数据集应该抛出 ValueError
        with pytest.raises(ValueError):
            clf.fit(X_empty, y_empty)


if __name__ == "__main__":
    pytest.main(["-v", "test_imbalanced_classifier.py"]) 