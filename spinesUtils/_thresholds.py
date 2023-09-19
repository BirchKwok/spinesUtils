import numpy as np
import pandas as pd

from spinesUtils.asserts import TypeAssert
from spinesUtils.metrics import make_metric


@TypeAssert({'y': np.ndarray, 'threshold': float})
def threshold_chosen(y, threshold=0.2):
    """二分类阈值选择法"""
    return (y > threshold).astype(int)


@TypeAssert({'df': pd.DataFrame, 'target_col': str, 'decay': int})
def get_sample_weights(df, target_col, decay=1):
    """获取样本权重"""
    _ = {k: v for k, v in df[target_col].value_counts().items()}
    _dict = {k: v for k, v in df[target_col].value_counts().items()}
    _ = sorted(_dict.items(), key=lambda s: s[-1], reverse=False)

    max_class = _[-1][0]
    max_class_nums = _[-1][1]

    return [1 if i == max_class else max_class_nums / _dict[i] * decay for i in df[target_col]]


@TypeAssert({
    'x': pd.DataFrame,
    'y': np.ndarray,
    'metric_name': str,
    'maximize': bool,
    'early_stopping': bool,
    'floating': bool,
    'skip_steps': int,
    'floating_search_loop': int,
    'verbose': bool
})
def auto_search_threshold(x, y, model,
                          metric_name='f1',
                          maximize=True,
                          early_stopping=True,
                          floating=True,
                          skip_steps=2,
                          floating_search_loop=2,
                          verbose=True,
                          **predict_params):
    import numpy as np
    metric = make_metric(metric_name)

    from spinesUtils.utils import Printer

    logger = Printer(verbose=verbose)

    logger.print("Automatically searching...")
    lr = 0.01

    yp_prob = model.predict_proba(x, **predict_params)[:, 1].squeeze()
    assert skip_steps >= 1
    assert floating_search_loop >= 0

    def _loops_chosen(loops, yp_pb=yp_prob, early_stopping=early_stopping):
        ms = np.finfo(np.float32).min
        bt = 0
        es = 0
        fast_signal = False
        init_skip_steps = 0

        for ts in loops:
            if fast_signal and init_skip_steps <= skip_steps:
                init_skip_steps += 1
                continue
            else:
                init_skip_steps = 0
                fast_signal = False

            yp = threshold_chosen(
                yp_pb,
                threshold=ts
            )

            score_epoch = metric(y, yp) if maximize else -metric(y, yp)

            if score_epoch > ms:
                bt, ms = ts, score_epoch
                fast_signal = True

                es = 0
            else:
                if early_stopping:
                    es += 1
                    if es >= 5:
                        logger.print(f"[early stopping]  max {metric_name} score: {ms},  best threshold: {bt}")
                        break

            logger.print(f"[current loop] try threshold: {ts}, max {metric_name} score: {ms},  best threshold: {bt}")

        return bt, ms

    max_score_group = _loops_chosen(np.arange(0, 1, lr))

    if floating:
        assert floating_search_loop >= 1
        for _ in range(floating_search_loop):
            lr /= 10
            logger.print(f"[floating loop {_ + 1}] Positive floating searching...")
            pos_res = _loops_chosen(np.arange(max_score_group[0], 1, lr))
            logger.print(f"[floating loop {_ + 1}] Negative floating searching...")
            neg_res = _loops_chosen(np.arange(max_score_group[0], 0, -lr))

            res = [max_score_group, pos_res, neg_res]

            max_score_idx = np.argmax([i[1] for i in res])

            if max_score_group != res[max_score_idx]:
                max_score_group = res[max_score_idx]
            else:
                logger.print("[Global Stopping]  tried to improve the {metric_name} score, "
                             "but it had no effect, stopped prematurely.")
                break

    best_threshold, max_score = max_score_group

    logger.print(f"[Global Stopping]  max {metric_name} score: {max_score},  best threshold: {best_threshold}")

    return best_threshold
