import numpy as np
import pandas as pd
from tqdm import tqdm

from spinesUtils.utils import Printer


def pos_pred_sample(tree_model, samples, threshold=0.5, use_prob=False, verbose=0):
    """获取预测为正例的样本"""
    from ._split_tools import df_block_split

    # 分割样本
    samples_list = df_block_split(samples, rows_limit=10000)
    total_length = int(np.ceil(samples.shape[0] / 10000))
    pred_res = []
    if verbose:
        iter = tqdm(samples_list, desc=f"filtering positive samples...", total=total_length)
    else:
        iter = samples_list

    for row_idx in iter:
        ss = samples.iloc[row_idx, :]

        if use_prob:
            pred_res.append(
                (
                        tree_model.predict_proba(ss)[:, -1] >= threshold
                ).astype(int)
            )
        else:
            pred_res.append(tree_model.predict(ss).squeeze())

    yp = np.concatenate(pred_res, axis=0)

    return samples.iloc[yp == 1, :].index.tolist()


def get_samples_shap_val(
        tree_model, samples, columns=None, target=None, pos_label=1, threshold=0.5,
        use_prob=False, method='shap', check_additivity=True, use_v2=True,
        approximate=False, tree_limit=None, verbose=0
):
    """获取正例样本的 shap 值
    tree_model: 已完成训练的树模型实例
    samples: 用以解释的样本集
    columns: 解释结果的列名，如果samples是pandas DataFrame并且此参数为None，将默认使用DataFrame全部列名
    pos_model: 正例的索引
    """

    assert isinstance(use_prob, bool) and isinstance(check_additivity, bool) and \
           isinstance(use_v2, bool) and isinstance(approximate, bool)

    assert check_additivity != approximate

    logger = Printer(verbose=verbose)


    if method == 'fast':
        try:
            import fasttreeshap as shap
            shap_name = 'fast'

        except ImportError:
            import shap
            shap_name = 'shap'
    elif method == 'lgb':
        shap_name = 'lgb'
    else:
        import shap
        shap_name = 'shap'

    logger.print(f"Use the `{shap_name}` method.")

    pos_index = pos_pred_sample(tree_model, samples, threshold=threshold, use_prob=use_prob)
    if target is not None:
        pos_samples = samples.iloc[pos_index, :].reset_index(drop=True)
        if isinstance(target, pd.Series):
            target = target.values
        pos_samples = pos_samples.loc[target[pos_index] == pos_label].reset_index(drop=True)
    else:
        pos_samples = samples.iloc[pos_index, :].reset_index(drop=True)
    logger.print(f"预测正类数量为: {pos_samples.shape[0]}")

    # 分割样本
    if pos_samples.shape[0] > 10000:
        split_num = int(np.ceil(pos_samples.shape[0] / 10000))

        pos_sample_list = [pos_samples.iloc[i * 10000:(i + 1) * 10000, :] for i in range(split_num)]
    else:
        pos_sample_list = [pos_samples]

    def check_shap_val(shap_v, p):
        if len(shap_v) == 2 and len(p) != 2:
            shap_v = np.squeeze(shap_v[pos_label])
        else:
            shap_v = np.squeeze(shap_v)
        return shap_v

    shap_res = []

    if verbose > 0:
        iter = tqdm(pos_sample_list, desc=f"{shap_name} method to compute shap value")
    else:
        iter = pos_sample_list

    if shap_name == 'fast':
        explainer = shap.TreeExplainer(tree_model, algorithm='v2' if use_v2 else 'auto',
                                       n_jobs=-1, shortcut=False)

        for ps in iter:
            shap_res.append(
                check_shap_val(
                    explainer.shap_values(ps, check_additivity=check_additivity,
                                          approximate=approximate, tree_limit=tree_limit),
                    pos_samples
                )
            )
    elif shap_name == 'lgb':
        import lightgbm as lgb
        assert isinstance(tree_model, (lgb.basic.Booster, lgb.LGBMModel))
        for ps in iter:
            shap_res.append(
                check_shap_val(
                    tree_model.predict(ps, pred_contrib=True)[:, :-1],
                    pos_samples
                )
            )
    else:
        explainer = shap.TreeExplainer(tree_model)
        for ps in iter:
            shap_res.append(
                check_shap_val(
                    explainer.shap_values(ps, check_additivity=check_additivity,
                                          tree_limit=tree_limit),
                    pos_samples
                )
            )

    logger.print("positive samples shapley values concatenating...")
    shap_values = np.concatenate(shap_res, axis=0)
    logger.print(f"shap vals shape: {shap_values.shape}")

    if isinstance(samples, pd.DataFrame):
        return pd.DataFrame(shap_values, columns=samples.columns if columns is None else columns), pos_index
    else:
        if columns:
            return pd.DataFrame(shap_values, columns=columns), pos_index

    return shap_values, pos_index


def sorted_shap_val(
        tree_model, samples,
        ascending=False,
        columns=None,
        pos_label=1,
        target=None,
        threshold=0.5,
        use_prob=False
):
    """根据shapley value绝对值排序"""
    shap_val, _ = get_samples_shap_val(tree_model=tree_model, samples=samples, columns=columns,
                                       pos_label=pos_label, threshold=threshold, use_prob=use_prob, target=target,
                                       verbose=0)
    shap_sum = shap_val.abs().mean()
    if isinstance(shap_sum, pd.Series):
        return shap_sum.sort_values(ascending=ascending)
    else:
        return {'idx': np.argsort(shap_sum)[::-1], 'values': shap_sum}


def make_metric(metrics_name):
    if metrics_name == 'f1':
        from sklearn.metrics import f1_score as score_func
    elif metrics_name == 'recall':
        from sklearn.metrics import recall_score as score_func
    elif metrics_name == 'precision':
        from sklearn.metrics import precision_score as score_func
    elif metrics_name == 'accuracy':
        from sklearn.metrics import accuracy_score as score_func
    elif metrics_name == 'mse':
        from sklearn.metrics import mean_squared_error as score_func
    elif metrics_name == 'mae':
        from sklearn.metrics import mean_absolute_error as score_func
    elif metrics_name == 'r2':
        from sklearn.metrics import r2_score as score_func
    else:
        raise ValueError(f"{metrics_name} is invalid.")

    return score_func