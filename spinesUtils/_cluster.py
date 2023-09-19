import pandas as pd

from spinesUtils.asserts import TypeAssert


@TypeAssert({'df': pd.DataFrame, 'n_cluster': int, 'dist_algo': str, 'batch_size': int, 'random_state': (None, int)})
def make_cluster(df, n_cluster, dist_algo='mini', batch_size=10000, random_state=42):
    import pandas as pd

    assert dist_algo in ('kmeans', 'mini', 'bisecting')

    if dist_algo == 'kmeans':
        from sklearn.cluster import KMeans as algo
    elif dist_algo == 'mini':
        from sklearn.cluster import MiniBatchKMeans as algo
    elif dist_algo == 'bisecting':
        from sklearn.cluster import BisectingKMeans as algo
    else:
        raise ValueError

    ag = algo(n_clusters=n_cluster, random_state=random_state)

    if dist_algo == 'mini':
        from tqdm.auto import trange
        for i in trange((df.shape[0] // batch_size) + 1, desc='clustering...'):
            if isinstance(df, pd.DataFrame):
                ag.partial_fit(df.iloc[i * batch_size: (i + 1) * batch_size, :])
            else:
                ag.partial_fit(df[i * batch_size: (i + 1) * batch_size, :])
    else:
        ag.fit(df)

    return ag


@TypeAssert({
    'df': pd.DataFrame,
    'n_cluster': int,
    'dist_algo': str,
    'batch_size': int,
    'marker_mapping': (None, list, tuple),
    'figsize': tuple,
    'random_state': (None, int),
})
def plot_cluster_res(df, n_cluster, dist_algo='mini', batch_size=10000,
                     marker_mapping=None, figsize=(10, 8), random_state=42):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    reduced_data = PCA(2, random_state=random_state).fit_transform(df)

    ag = make_cluster(df=reduced_data, n_cluster=n_cluster, dist_algo=dist_algo, batch_size=batch_size)

    label = ag.predict(reduced_data)

    # 根据簇标签生成独立的数组
    clusters = [reduced_data[label == i, :] for i in range(n_cluster)]

    # 绘制散点图
    fig, ax = plt.subplots(figsize=figsize)
    for i, cluster in enumerate(clusters):
        ax.scatter(
            cluster[:, 0], cluster[:, 1], c=f'C{i}', label=f'Cluster {i + 1}'
            if marker_mapping is None else marker_mapping[i + 1]
        )

    centroids = ag.cluster_centers_

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="*",
        s=50,
        linewidths=3,
        color="gold",
        zorder=10,
    )

    plt.legend()
    plt.show()

    return ag
