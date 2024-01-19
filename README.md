# spinesUtils 
*Dedicated to helping users do more in less time.*

<big><i><b>spinesUtils</b></i></big>
 is a user-friendly toolkit for the machine learning ecosystem, offering ready-to-use features such as

- [x] Logging functionality
- [x] Type checking and parameter generation
- [x] CSV file reading acceleration
- [x] Classifiers for imbalanced data
- [x] Pandas Dataframe data compression
- [x] Pandas DataFrame insight tools
- [x] Large data training and testing set splitting functions
- [x] An intuitive timer.

It is currently undergoing rapid iteration. If you encounter any issues with its functionalities, feel free to raise an issue.

# Installation
You can install spinesUtils from PyPI:
```bash
pip install spinesUtils
```

# Logger

You can use the Logger class to print your logs without worrying about handler conflicts with the native Python logging module. 

This class provides log/debug/info/warning/error/critical methods, where debug/info/warning/error/critical are partial versions of the log method, available for use as needed.


```python
# load spinesUtils module
from spinesUtils.logging import Logger

# create a logger instance, with name "MyLogger", and no file handler, the default level is "INFO"
# You can specify a file path `fp` during instantiation. If not specified, logs will not be written to a file.
logger = Logger(name="MyLogger", fp=None, level="DEBUG")

logger.log("This is an info log emitted by the log function.", level='INFO')
logger.debug("This is an debug message")
logger.info("This is an info message.")
logger.warning("This is an warning message.")
logger.error("This is an error message.")
logger.critical("This is an critical message.")
```

    2024-01-19 15:02:51 - MyLogger - INFO - This is an info log emitted by the log function.
    2024-01-19 15:02:51 - MyLogger - DEBUG - This is an debug message
    2024-01-19 15:02:51 - MyLogger - INFO - This is an info message.
    2024-01-19 15:02:51 - MyLogger - WARNING - This is an warning message.
    2024-01-19 15:02:51 - MyLogger - ERROR - This is an error message.
    2024-01-19 15:02:51 - MyLogger - CRITICAL - This is an critical message.


## Type checking and parameter generation 


```python
from spinesUtils.asserts import *

# check parameter type
@ParameterTypeAssert({
    'a': (int, float),
    'b': (int, float)
})
def add(a, b):
    pass

# try to pass a string to the function, and it will raise an ParametersTypeError error
add(a=1, b='2')
```


    ---------------------------------------------------------------------------

    ParametersTypeError                       Traceback (most recent call last)

    Cell In[2], line 12
          9     pass
         11 # try to pass a string to the function, and it will raise an ParametersTypeError error
    ---> 12 add(a=1, b='2')


    File ~/projects/spinesUtils/spinesUtils/asserts/_inspect.py:196, in ParameterTypeAssert.__call__.<locals>.wrapper(*args, **kwargs)
        194 if mismatched_params:
        195     error_msg = self.build_type_error_msg(mismatched_params)
    --> 196     raise ParametersTypeError(error_msg)
        198 return func(**kwargs)


    ParametersTypeError: Function 'add' parameter(s) type mismatch: b only accept '['int', 'float']' type.



```python
# check parameter value
@ParameterValuesAssert({
    'a': lambda x: x > 0,
    'b': lambda x: x > 0
})
def add(a, b):
    pass

# try to pass a negative number to the function, and it will raise an ParametersValueError error
add(a=1, b=-2)
```


    ---------------------------------------------------------------------------

    ParametersValueError                      Traceback (most recent call last)

    Cell In[3], line 10
          7     pass
          9 # try to pass a negative number to the function, and it will raise an ParametersValueError error
    ---> 10 add(a=1, b=-2)


    File ~/projects/spinesUtils/spinesUtils/asserts/_inspect.py:258, in ParameterValuesAssert.__call__.<locals>.wrapper(*args, **kwargs)
        256 if mismatched_params:
        257     error_msg = self.build_values_error_msg(mismatched_params)
    --> 258     raise ParametersValueError(error_msg)
        260 return func(**kwargs)


    ParametersValueError: Function 'add' parameter(s) values mismatch: `b` must in or satisfy ''b': lambda x: x > 0' condition(s).



```python
# generate a dictionary of keyword arguments for a given function using provided arguments
generate_function_kwargs(add, a=1, b=2)
```




    {'a': 1, 'b': 2}




```python
# isinstance function with support for None
augmented_isinstance(1, (int, float, None))
```




    True




```python
# raise_if and raise_if_not functions
raise_if(ValueError, 1 == 1, "test raise_if")
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[6], line 2
          1 # raise_if and raise_if_not functions
    ----> 2 raise_if(ValueError, 1 == 1, "test raise_if")


    File ~/projects/spinesUtils/spinesUtils/asserts/_type_and_exceptions.py:115, in raise_if(exception, condition, error_msg)
        112 assert issubclass(exception, BaseException), "Exception must be a subclass of BaseException."
        114 if condition:
    --> 115     raise exception(error_msg)


    ValueError: test raise_if



```python
raise_if_not(ZeroDivisionError, 1 != 1, "test raise_if_not")
```


    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    Cell In[7], line 1
    ----> 1 raise_if_not(ZeroDivisionError, 1 != 1, "test raise_if_not")


    File ~/projects/spinesUtils/spinesUtils/asserts/_type_and_exceptions.py:144, in raise_if_not(exception, condition, error_msg)
        141 assert issubclass(exception, BaseException), "Exception must be a subclass of BaseException."
        143 if not condition:
    --> 144     raise exception(error_msg)


    ZeroDivisionError: test raise_if_not


## Faster csv reader


```python
from spinesUtils import read_csv

your_df = read_csv(
    fp='/path/to/your/file.csv',
    sep=',',  # equal to pandas read_csv.sep
    turbo_method='polars',  # use turbo_method to speed up load time
    chunk_size=None,  # it can be integer if you want to use pandas backend
    transform2low_mem=True,  # it can compresses file to save more memory
    verbose=False
)
```

## Classifiers for imbalanced data


```python
from spinesUtils.models import MultiClassBalanceClassifier
```


```python
# make a toy dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset = make_classification(
    n_samples=10000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=1,
    weights=[0.01, 0.05, 0.94],
    class_sep=0.8,
    random_state=0
)

X, y = dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```


```python
from sklearn.ensemble import RandomForestClassifier

classifier = MultiClassBalanceClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100),
    n_classes=3,
    random_state=0,
    verbose=0
)

# fit the classifier
classifier.fit(X_train, y_train)

# predict
y_pred = classifier.predict(X_test)

# print classification report
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.74      0.72      0.73        32
               1       0.91      0.71      0.80       111
               2       0.98      1.00      0.99      1857
    
        accuracy                           0.98      2000
       macro avg       0.88      0.81      0.84      2000
    weighted avg       0.98      0.98      0.98      2000


## Pandas dataframe data compression


```python
# make a toy dataset
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'a': np.random.randint(0, 100, 100000),
    'b': np.random.randint(0, 100, 100000),
    'c': np.random.randint(0, 100, 100000),
    'd': np.random.randint(0, 100, 100000),
    'e': np.random.randint(0, 100, 100000),
    'f': np.random.randint(0, 100, 100000),
    'g': np.random.randint(0, 100, 100000),
    'h': np.random.randint(0, 100, 100000),
    'i': np.random.randint(0, 100, 100000),
    'j': np.random.randint(0, 100, 100000),
    'k': np.random.randint(0, 100, 100000),
    'l': np.random.randint(0, 100, 100000),
    'm': np.random.randint(0, 100, 100000),
    'n': np.random.randint(0, 100, 100000),
    'o': np.random.randint(0, 100, 100000),
    'p': np.random.randint(0, 100, 100000),
    'q': np.random.randint(0, 100, 100000),
    'r': np.random.randint(0, 100, 100000),
    's': np.random.randint(0, 100, 100000),
    't': np.random.randint(0, 100, 100000),
    'u': np.random.randint(0, 100, 100000),
    'v': np.random.randint(0, 100, 100000),
    'w': np.random.randint(0, 100, 100000),
    'x': np.random.randint(0, 100, 100000),
    'y': np.random.randint(0, 100, 100000),
    'z': np.random.randint(0, 100, 100000),
})

# compress dataframe
from spinesUtils import transform_dtypes_low_mem

transform_dtypes_low_mem(df, verbose=True, inplace=True)
```


    Converting ...:   0%|          | 0/26 [00:00<?, ?it/s]


    [log] INFO - Memory usage before conversion is: 19.84 MB  
    [log] INFO - Memory usage after conversion is: 2.48 MB  
    [log] INFO - After conversion, the percentage of memory fluctuation is 87.5 %



```python
# batch compress dataframes
from spinesUtils import transform_batch_dtypes_low_mem

# make some toy datasets
df1 = pd.DataFrame({
    'a': np.random.randint(0, 100, 100000),
    'b': np.random.randint(0, 100, 100000),
    'c': np.random.randint(0, 100, 100000),
    'd': np.random.randint(0, 100, 100000),
    'e': np.random.randint(0, 100, 100000),
    'f': np.random.randint(0, 100, 100000),
    'g': np.random.randint(0, 100, 100000),
    'h': np.random.randint(0, 100, 100000),
    'i': np.random.randint(0, 100, 100000),
    'j': np.random.randint(0, 100, 100000),
    'k': np.random.randint(0, 100, 100000),
    'l': np.random.randint(0, 100, 100000),
    'm': np.random.randint(0, 100, 100000),
    'n': np.random.randint(0, 100, 100000),
    'o': np.random.randint(0, 100, 100000),
    'p': np.random.randint(0, 100, 100000),
    'q': np.random.randint(0, 100, 100000),
    'r': np.random.randint(0, 100, 100000),
    's': np.random.randint(0, 100, 100000),
    't': np.random.randint(0, 100, 100000),
    'u': np.random.randint(0, 100, 100000),
    'v': np.random.randint(0, 100, 100000),
    'w': np.random.randint(0, 100, 100000),
    'x': np.random.randint(0, 100, 100000),
    'y': np.random.randint(0, 100, 100000),
    'z': np.random.randint(0, 100, 100000),
})

df2 = df1.copy()
df3 = df1.copy()
df4 = df1.copy()

# batch compress dataframes
transform_batch_dtypes_low_mem([df1, df2, df3, df4], verbose=True, inplace=True)
```


    Batch converting ...:   0%|          | 0/4 [00:00<?, ?it/s]


    [log] INFO - Memory usage before conversion is: 79.35 MB  
    [log] INFO - Memory usage after conversion is: 9.92 MB  
    [log] INFO - After conversion, the percentage of memory fluctuation is 87.5 %


## Pandas DataFrame insight tools


```python
from spinesUtils import df_preview, classify_samples_dist

# make a toy dataset
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'a': np.random.randint(0, 100, 100000),
    'b': np.random.randint(0, 100, 100000),
    'c': np.random.randint(0, 100, 100000),
    'd': np.random.randint(0, 100, 100000),
    'e': np.random.randint(0, 100, 100000),
    'f': np.random.randint(0, 100, 100000),
    'g': np.random.randint(0, 100, 100000),
    'h': np.random.randint(0, 100, 100000),
    'i': np.random.randint(0, 100, 100000),
    'j': np.random.randint(0, 100, 100000),
    'k': np.random.randint(0, 100, 100000),
    'l': np.random.randint(0, 100, 100000),
    'm': np.random.randint(0, 100, 100000),
    'n': np.random.randint(0, 100, 100000),
    'o': np.random.randint(0, 100, 100000),
    'p': np.random.randint(0, 100, 100000),
    'q': np.random.randint(0, 100, 100000),
    'r': np.random.randint(0, 100, 100000),
    's': np.random.randint(0, 100, 100000),
    't': np.random.randint(0, 100, 100000),
    'u': np.random.randint(0, 100, 100000),
    'v': np.random.randint(0, 100, 100000),
    'w': np.random.randint(0, 100, 100000),
    'x': np.random.randint(0, 100, 100000),
    'y': np.random.randint(0, 100, 100000),
    'z': np.random.randint(0, 100, 100000),
})

df_insight = df_preview(df)

df_insight
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total</th>
      <th>na</th>
      <th>naPercent</th>
      <th>nunique</th>
      <th>dtype</th>
      <th>max</th>
      <th>75%</th>
      <th>median</th>
      <th>25%</th>
      <th>min</th>
      <th>mean</th>
      <th>mode</th>
      <th>variation</th>
      <th>std</th>
      <th>skew</th>
      <th>kurt</th>
      <th>samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.53968</td>
      <td>36</td>
      <td>0.9892</td>
      <td>28.848392</td>
      <td>-0.000158</td>
      <td>-1.196434</td>
      <td>(32, 81)</td>
    </tr>
    <tr>
      <th>b</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>49.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.41822</td>
      <td>40</td>
      <td>0.98928</td>
      <td>28.937601</td>
      <td>0.005974</td>
      <td>-1.206987</td>
      <td>(76, 28)</td>
    </tr>
    <tr>
      <th>c</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.58261</td>
      <td>82</td>
      <td>0.98923</td>
      <td>28.928019</td>
      <td>-0.003537</td>
      <td>-1.202994</td>
      <td>(21, 68)</td>
    </tr>
    <tr>
      <th>d</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>49.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.46308</td>
      <td>9</td>
      <td>0.98906</td>
      <td>28.886459</td>
      <td>0.003344</td>
      <td>-1.200654</td>
      <td>(42, 90)</td>
    </tr>
    <tr>
      <th>e</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>49.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.55014</td>
      <td>37</td>
      <td>0.98911</td>
      <td>28.834041</td>
      <td>0.003987</td>
      <td>-1.196103</td>
      <td>(15, 59)</td>
    </tr>
    <tr>
      <th>f</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>49.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.20195</td>
      <td>4</td>
      <td>0.98926</td>
      <td>28.886463</td>
      <td>0.009183</td>
      <td>-1.203297</td>
      <td>(72, 9)</td>
    </tr>
    <tr>
      <th>g</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.62199</td>
      <td>4</td>
      <td>0.98919</td>
      <td>28.849264</td>
      <td>-0.012746</td>
      <td>-1.199283</td>
      <td>(69, 64)</td>
    </tr>
    <tr>
      <th>h</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.58739</td>
      <td>40</td>
      <td>0.98917</td>
      <td>28.83744</td>
      <td>-0.004719</td>
      <td>-1.193858</td>
      <td>(30, 79)</td>
    </tr>
    <tr>
      <th>i</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>49.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.41076</td>
      <td>10</td>
      <td>0.98939</td>
      <td>28.910095</td>
      <td>0.005218</td>
      <td>-1.207459</td>
      <td>(36, 54)</td>
    </tr>
    <tr>
      <th>j</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>49.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.45686</td>
      <td>46</td>
      <td>0.98909</td>
      <td>28.816681</td>
      <td>0.004751</td>
      <td>-1.190756</td>
      <td>(29, 95)</td>
    </tr>
    <tr>
      <th>k</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.54948</td>
      <td>46</td>
      <td>0.98914</td>
      <td>28.806187</td>
      <td>-0.003731</td>
      <td>-1.196876</td>
      <td>(32, 94)</td>
    </tr>
    <tr>
      <th>l</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>49.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.45631</td>
      <td>20</td>
      <td>0.98923</td>
      <td>28.921314</td>
      <td>0.002344</td>
      <td>-1.205342</td>
      <td>(22, 91)</td>
    </tr>
    <tr>
      <th>m</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>49.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.43142</td>
      <td>49</td>
      <td>0.98901</td>
      <td>28.852962</td>
      <td>0.002507</td>
      <td>-1.198267</td>
      <td>(94, 26)</td>
    </tr>
    <tr>
      <th>n</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>50.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.49325</td>
      <td>8</td>
      <td>0.98931</td>
      <td>28.899022</td>
      <td>0.000698</td>
      <td>-1.200786</td>
      <td>(46, 50)</td>
    </tr>
    <tr>
      <th>o</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.52091</td>
      <td>4</td>
      <td>0.98923</td>
      <td>28.869563</td>
      <td>-0.003987</td>
      <td>-1.202426</td>
      <td>(33, 13)</td>
    </tr>
    <tr>
      <th>p</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>49.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.40997</td>
      <td>61</td>
      <td>0.98918</td>
      <td>28.900207</td>
      <td>0.007921</td>
      <td>-1.204621</td>
      <td>(58, 93)</td>
    </tr>
    <tr>
      <th>q</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.62826</td>
      <td>33</td>
      <td>0.98936</td>
      <td>28.831896</td>
      <td>-0.003291</td>
      <td>-1.201172</td>
      <td>(82, 31)</td>
    </tr>
    <tr>
      <th>r</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>50.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.47208</td>
      <td>60</td>
      <td>0.98925</td>
      <td>28.873943</td>
      <td>0.000515</td>
      <td>-1.202925</td>
      <td>(0, 26)</td>
    </tr>
    <tr>
      <th>s</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.64847</td>
      <td>48</td>
      <td>0.9893</td>
      <td>28.853741</td>
      <td>-0.010258</td>
      <td>-1.202701</td>
      <td>(94, 37)</td>
    </tr>
    <tr>
      <th>t</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.55305</td>
      <td>32</td>
      <td>0.98898</td>
      <td>28.801028</td>
      <td>-0.001721</td>
      <td>-1.193403</td>
      <td>(85, 10)</td>
    </tr>
    <tr>
      <th>u</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>49.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.45428</td>
      <td>80</td>
      <td>0.98928</td>
      <td>28.876812</td>
      <td>0.002018</td>
      <td>-1.201612</td>
      <td>(56, 16)</td>
    </tr>
    <tr>
      <th>v</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>75.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.59953</td>
      <td>16</td>
      <td>0.98945</td>
      <td>28.891313</td>
      <td>-0.006261</td>
      <td>-1.199011</td>
      <td>(60, 39)</td>
    </tr>
    <tr>
      <th>w</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>49.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.34131</td>
      <td>4</td>
      <td>0.98915</td>
      <td>28.925175</td>
      <td>0.009523</td>
      <td>-1.203308</td>
      <td>(78, 96)</td>
    </tr>
    <tr>
      <th>x</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>49.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.45791</td>
      <td>95</td>
      <td>0.98933</td>
      <td>28.860322</td>
      <td>0.007199</td>
      <td>-1.198962</td>
      <td>(93, 79)</td>
    </tr>
    <tr>
      <th>y</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>50.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>49.58517</td>
      <td>34</td>
      <td>0.98929</td>
      <td>28.765474</td>
      <td>-0.000497</td>
      <td>-1.193016</td>
      <td>(80, 42)</td>
    </tr>
    <tr>
      <th>z</th>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>100</td>
      <td>int64</td>
      <td>99.0</td>
      <td>74.0</td>
      <td>50.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>49.44355</td>
      <td>21</td>
      <td>0.98876</td>
      <td>28.85751</td>
      <td>0.000819</td>
      <td>-1.201063</td>
      <td>(25, 25)</td>
    </tr>
  </tbody>
</table>
</div>



## Large data training and testing set splitting functions


```python
# make a toy dataset
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'a': np.random.randint(0, 100, 100000),
    'b': np.random.randint(0, 100, 100000),
    'c': np.random.randint(0, 100, 100000),
    'd': np.random.randint(0, 100, 100000),
    'e': np.random.randint(0, 100, 100000),
    'f': np.random.randint(0, 100, 100000),
    'g': np.random.randint(0, 100, 100000),
    'h': np.random.randint(0, 100, 100000),
    'i': np.random.randint(0, 100, 100000),
    'j': np.random.randint(0, 100, 100000),
    'k': np.random.randint(0, 100, 100000),
    'l': np.random.randint(0, 100, 100000),
    'm': np.random.randint(0, 100, 100000),
    'n': np.random.randint(0, 100, 100000),
    'o': np.random.randint(0, 100, 100000),
    'p': np.random.randint(0, 100, 100000),
    'q': np.random.randint(0, 100, 100000),
    'r': np.random.randint(0, 100, 100000),
    's': np.random.randint(0, 100, 100000),
    't': np.random.randint(0, 100, 100000),
    'u': np.random.randint(0, 100, 100000),
    'v': np.random.randint(0, 100, 100000),
    'w': np.random.randint(0, 100, 100000),
    'x': np.random.randint(0, 100, 100000),
    'y': np.random.randint(0, 100, 100000),
    'z': np.random.randint(0, 100, 100000),
})

# split dataframe into training and testing sets

# return numpy.ndarray
from spinesUtils import train_test_split_bigdata
from spinesUtils.feature_tools import get_x_cols

X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_split_bigdata(
    df=df, 
    x_cols=get_x_cols(df, y_col='a'),
    y_col='a', 
    shuffle=True,
    return_valid=True,
    train_size=0.8,
    valid_size=0.5
)

print(X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape)
X_train[:5]
```

    (80000, 25) (80000,) (10000, 25) (10000,) (10000, 25) (10000,)





    array([[45, 83, 43, 94,  1, 86, 56,  0, 78, 60, 79, 42, 24, 43, 94, 83,
            45, 50, 59, 50, 17, 99, 40, 95, 70],
           [ 4, 81,  9, 25, 54, 18, 14,  6, 17, 39,  0, 36, 82, 33, 11, 76,
            92, 29, 33, 50, 44, 11, 87, 86, 31],
           [72, 82, 52, 96, 55, 89, 35, 71, 48, 73, 34, 19, 53, 89, 46, 57,
            84, 67, 10, 40, 50, 61, 10, 76, 84],
           [46, 45, 79, 53, 80, 85, 58, 65, 26, 49, 46, 97, 83, 47, 77, 97,
            26,  4, 33, 79, 36, 65, 50, 94, 87],
           [36,  7, 46, 10, 11, 33,  3,  7, 82, 29, 28,  2, 42, 89, 42, 66,
            79, 51, 49, 43, 63, 14, 13, 74, 26]])




```python
# return pandas.DataFrame
from spinesUtils import train_test_split_bigdata_df
from spinesUtils.feature_tools import get_x_cols

train_df, valid_df, test_df = train_test_split_bigdata_df(
    df=df, 
    x_cols=get_x_cols(df, y_col='a'),
    y_col='a', 
    shuffle=True,
    return_valid=True,
    train_size=0.8,
    valid_size=0.5
)

print(train_df.shape, valid_df.shape, test_df.shape)
train_df.head()
```

    (8000000, 26) (1000000, 26) (1000000, 26)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
      <th>g</th>
      <th>h</th>
      <th>i</th>
      <th>j</th>
      <th>k</th>
      <th>...</th>
      <th>r</th>
      <th>s</th>
      <th>t</th>
      <th>u</th>
      <th>v</th>
      <th>w</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14</td>
      <td>67</td>
      <td>41</td>
      <td>87</td>
      <td>68</td>
      <td>87</td>
      <td>27</td>
      <td>67</td>
      <td>26</td>
      <td>62</td>
      <td>...</td>
      <td>63</td>
      <td>43</td>
      <td>77</td>
      <td>4</td>
      <td>6</td>
      <td>72</td>
      <td>5</td>
      <td>63</td>
      <td>73</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47</td>
      <td>37</td>
      <td>43</td>
      <td>98</td>
      <td>55</td>
      <td>68</td>
      <td>82</td>
      <td>48</td>
      <td>37</td>
      <td>35</td>
      <td>...</td>
      <td>99</td>
      <td>92</td>
      <td>23</td>
      <td>44</td>
      <td>92</td>
      <td>14</td>
      <td>54</td>
      <td>95</td>
      <td>58</td>
      <td>59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52</td>
      <td>97</td>
      <td>71</td>
      <td>62</td>
      <td>18</td>
      <td>54</td>
      <td>22</td>
      <td>2</td>
      <td>57</td>
      <td>93</td>
      <td>...</td>
      <td>82</td>
      <td>6</td>
      <td>61</td>
      <td>41</td>
      <td>24</td>
      <td>40</td>
      <td>54</td>
      <td>11</td>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>45</td>
      <td>22</td>
      <td>46</td>
      <td>32</td>
      <td>37</td>
      <td>6</td>
      <td>13</td>
      <td>42</td>
      <td>67</td>
      <td>...</td>
      <td>9</td>
      <td>1</td>
      <td>65</td>
      <td>84</td>
      <td>11</td>
      <td>86</td>
      <td>54</td>
      <td>22</td>
      <td>89</td>
      <td>85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26</td>
      <td>23</td>
      <td>55</td>
      <td>31</td>
      <td>61</td>
      <td>72</td>
      <td>68</td>
      <td>82</td>
      <td>6</td>
      <td>19</td>
      <td>...</td>
      <td>13</td>
      <td>44</td>
      <td>3</td>
      <td>93</td>
      <td>66</td>
      <td>53</td>
      <td>75</td>
      <td>93</td>
      <td>53</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
# performances comparison
from sklearn.model_selection import train_test_split
from spinesUtils import train_test_split_bigdata, train_test_split_bigdata_df
from spinesUtils.feature_tools import get_x_cols

# make a toy dataset
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'a': np.random.randint(0, 100, 10000),
    'b': np.random.randint(0, 100, 10000),
    'c': np.random.randint(0, 100, 10000),
    'd': np.random.randint(0, 100, 10000),
    'e': np.random.randint(0, 100, 10000),
    'f': np.random.randint(0, 100, 10000),
    'g': np.random.randint(0, 100, 10000),
    'h': np.random.randint(0, 100, 10000),
    'i': np.random.randint(0, 100, 10000),
    'j': np.random.randint(0, 100, 10000),
    'k': np.random.randint(0, 100, 10000),
    'l': np.random.randint(0, 100, 10000),
    'm': np.random.randint(0, 100, 10000),
    'n': np.random.randint(0, 100, 10000),
    'o': np.random.randint(0, 100, 10000),
    'p': np.random.randint(0, 100, 10000),
    'q': np.random.randint(0, 100, 10000),
    'r': np.random.randint(0, 100, 10000),
    's': np.random.randint(0, 100, 10000),
    't': np.random.randint(0, 100, 10000),
    'u': np.random.randint(0, 100, 10000),
    'v': np.random.randint(0, 100, 10000),
    'w': np.random.randint(0, 100, 10000),
    'x': np.random.randint(0, 100, 10000),
    'y': np.random.randint(0, 100, 10000),
    'z': np.random.randint(0, 100, 10000),
})

# define a function to split a valid set for sklearn train_test_split
def train_test_split_sklearn(df, x_cols, y_col, shuffle, train_size, valid_size):
    X_train, X_test, y_train, y_test = train_test_split(df[x_cols], df[y_col], test_size=1-train_size, random_state=0, shuffle=shuffle)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=valid_size, random_state=0, shuffle=shuffle)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

%timeit X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_split_sklearn(df=df, x_cols=get_x_cols(df, y_col='a'), y_col='a', shuffle=True, train_size=0.8, valid_size=0.5)
%timeit X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_split_bigdata(df=df, x_cols=get_x_cols(df, y_col='a'), y_col='a', shuffle=True, return_valid=True, train_size=0.8, valid_size=0.5)
%timeit train_df, valid_df, test_df = train_test_split_bigdata_df(df=df, x_cols=get_x_cols(df, y_col='a'), y_col='a', shuffle=True, return_valid=True, train_size=0.8, valid_size=0.5)
```

    1.28 ms ± 20.5 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    1.05 ms ± 14.1 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    1.36 ms ± 11.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


## An intuitive timer


```python
from spinesUtils.timer import Timer

# create a timer instance
timer = Timer()

# start the timer
timer.start()

# do something
for i in range(10):
    # timer sleep for 1 second
    timer.sleep(1)
    # print the elapsed time from last sleep
    print("Elapsed time: {} seconds".format(timer.last_timestamp_diff()))

# print the elapsed time
print("Total elapsed time: {} seconds".format(timer.total_elapsed_time()))

# stop the timer
timer.end()
```

    Elapsed time: 1.0117900371551514 seconds
    Elapsed time: 2.016140937805176 seconds
    Elapsed time: 3.0169479846954346 seconds
    Elapsed time: 4.0224690437316895 seconds
    Elapsed time: 5.027086019515991 seconds
    Elapsed time: 6.0309507846832275 seconds
    Elapsed time: 7.035104036331177 seconds
    Elapsed time: 8.040709972381592 seconds
    Elapsed time: 9.042311906814575 seconds
    Elapsed time: 10.046867847442627 seconds
    Total elapsed time: 10.047839879989624 seconds





    10.047943830490112




```python
from spinesUtils.timer import Timer

# you can also use the timer as a context manager
t = Timer()
with t.session():
    t.sleep(1)
    print("Last step elapsed time:", round(t.last_timestamp_diff(), 2), 'seconds')
    t.middle_point()
    t.sleep(2)
    print("Last step elapsed time:", round(t.last_timestamp_diff(), 2), 'seconds')
    
    total_elapsed_time = t.total_elapsed_time()
    
print("Total Time:", round(total_elapsed_time, 2), 'seconds')
```

    Last step elapsed time: 1.01 seconds
    Last step elapsed time: 2.01 seconds
    Total Time: 3.01 seconds

