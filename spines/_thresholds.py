def threshold_chosen(y, threshold=0.2):
    """二分类阈值选择法"""
    return (y > threshold).astype(int)


def get_sample_weights(df, target_col='is_losing_user', decay=1):
    """获取样本权重"""
    _ = {k: v for k, v in df[target_col].value_counts().items()}
    _dict = {k: v for k, v in df[target_col].value_counts().items()}
    _ = sorted(_dict.items(), key=lambda s: s[-1], reverse=False)

    max_class = _[-1][0]
    max_class_nums = _[-1][1]

    return [1 if i == max_class else max_class_nums / _dict[i] * decay for i in df[target_col]]


def auto_search_threshold(x, y, model,
                          early_stopping=True,
                          floating=True,
                          fast=True,
                          skip_steps=2,
                          floating_search_loop=2,
                          **predict_params):
    import numpy as np
    from sklearn.metrics import f1_score

    print("Automatically searching...")
    lr = 0.01

    yp_prob = model.predict_proba(x, **predict_params)[:, 1].squeeze()
    assert skip_steps >= 1
    assert isinstance(floating_search_loop, int) and floating_search_loop >= 0

    def _loops_chosen(loops, yp_pb=yp_prob, early_stopping=early_stopping):
        max_f1 = 0
        best_threshold = 0
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

            f1_epoch = f1_score(y, yp)

            if f1_epoch > max_f1:
                best_threshold, max_f1 = ts, f1_epoch
                fast_signal = True

                es = 0
            else:
                if early_stopping:
                    es += 1
                    if es >= 5:
                        print(f"[early stopping]  max f1 score: {max_f1},  best threshold: {best_threshold}")
                        break

            print(f"[current loop] try threshold: {ts}, max f1 score: {max_f1},  best threshold: {best_threshold}")

        return best_threshold, max_f1

    max_f1_group = _loops_chosen(np.arange(0, 1, lr))

    if floating:
        assert floating_search_loop >= 1
        for _ in range(floating_search_loop):
            lr /= 10
            print(f"[floating loop {_ + 1}] Positive floating searching...")
            pos_res = _loops_chosen(np.arange(max_f1_group[0], 1, lr))
            print(f"[floating loop {_ + 1}] Negative floating searching...")
            neg_res = _loops_chosen(np.arange(max_f1_group[0], 0, -lr))

            res = [max_f1_group, pos_res, neg_res]

            max_f1_idx = np.argmax([i[1] for i in res])

            if max_f1_group != res[max_f1_idx]:
                max_f1_group = res[max_f1_idx]
            else:
                print("[Global Stopping]  tried to improve the f1 score, but it had no effect, stopped prematurely.")
                break

    best_threshold, max_f1 = max_f1_group

    print(f"[Global Stopping]  max f1 score: {max_f1},  best threshold: {best_threshold}")

    return best_threshold
