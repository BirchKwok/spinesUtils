# spinesUtils -- A Machine-Learning Toolsets

模型训练工具集 model training toolsets.

使用pip安装 `use pip install spinesUtils`

```bash
pip install spinesUtils
```

## better csv dataloader
```python
from spinesUtils import dataloader

your_df = dataloader(
    fp='/path/to/your/file.csv',
    sep=',',  # equal to pandas read_csv.sep
    turbo_method='pyarrow', # use turbo_method to speed up load time
    chunk_size=None, # it can be integer if you want to use pandas backend
    save_as_pkl=False, # if you want to save the file as pickle, it can speed up next load time
    transform2low_mem=True, # it can compresses file to save more memory
    verbose=False
)

```

## better pandas dataframe insight tools
```python
from spinesUtils import df_preview, classify_samples_dist

df_insight = df_preview(your_df)

df_target_distribution = classify_samples_dist(your_df, target_col=your_df[y_col])

print(df_insight)
print(df_target_distribution)
```

## better dataframe compresses/uncompress tools

```python
# single dataframe
from spinesUtils import transform_dtypes_low_mem, inverse_transform_dtypes

# compresses file to save memory
transform_dtypes_low_mem(your_df, verbose=True)

# uncompress file to python type
inverse_transform_dtypes(your_df, verbose=True, int_dtypes=int, float_dtypes=float)
```

```python
# dataframes
import numpy as np
from spinesUtils import transform_batch_dtypes_low_mem, inverse_transform_batch_dtypes

your_dfs = [your_df1, your_df2, your_df3] # it can be unlimited

# compresses files to save memory
transform_batch_dtypes_low_mem(your_dfs, verbose=True)

# uncompress file to numpy type
inverse_transform_batch_dtypes(your_dfs, verbose=True, int_dtypes=np.int32, float_dtypes=np.float32)
```

## better features selector
```python
from spinesUtils import TreeSequentialFeatureSelector
from lightgbm import LGBMClassifier

estimator = LGBMClassifier(random_state=0)
fe = TreeSequentialFeatureSelector(estimator, metrics_name='f1',
    forward=True,
    floating=True,
    log_file_path='feature_selection.log',
    best_features_save_path='best_feature.txt', verbose=True)

fe.fit(your_df[x_cols], your_df[y_col])
print(fe.best_cols_, fe.best_score_)
```

## better train_test_split function
```python
# return numpy.ndarray
from spinesUtils import train_test_split_bigdata

X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_split_bigdata(
    df=your_df, 
    x_cols=x_cols,
    y_col=y_col, 
    shuffle=True,
    return_valid=True,
    train_size=0.8,
    valid_size=0.5
)
```
```python
# return pandas.dataframe
from spinesUtils import train_test_split_bigdata_df

train, valid, test = train_test_split_bigdata_df(
    df=your_df, 
    x_cols=x_cols,
    y_col=y_col, 
    shuffle=True,
    return_valid=True,
    train_size=0.8,
    valid_size=0.5,
    reset_index=True
)
```

## better imbalanced-data model
```python
from spinesUtils import BinaryBalanceClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

classifier = BinaryBalanceClassifier(meta_estimators=[LGBMClassifier(), LGBMClassifier()])

classifier.fit(your_df[x_cols], your_df[y_col], threshold_search_set=(your_df[x_cols], your_df[y_col]))

print('threshold: ', classifier.auto_threshold)

print(
    'f1:', f1_score(your_df[y_col], classifier.predict(your_df[x_cols])), 
    'recall:', recall_score(your_df[y_col], classifier.predict(your_df[x_cols])), 
    'precision:', precision_score(your_df[y_col], classifier.predict(your_df[x_cols]))
)
```

## log for human
```python
from spinesUtils import Printer

your_logger = Printer(name='your_logger', verbose=True, 
        truncate_file=True, with_time=True)

your_logger.insert2file("test") 
your_logger.print('test')

# Or you can do it both
your_logger.insert_and_throwout('test')
```