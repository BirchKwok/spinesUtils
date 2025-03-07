<div align="center">
  <h3><i><b>spinesUtils</b></i></h3>
  <p><i>Accelerate your Python development workflow</i></p>
</div>

## Overview

**spinesUtils** is a powerful library that provides ready-to-use features and utilities for Python development to shorten the project development cycle. Our goal is to help developers focus on solving their core problems instead of reimplementing common functionality.

## Features

- [x] **Logging functionality** - Simplified logging without handler conflicts
- [x] **Type checking and parameter validation** - Robust validation decorators
- [x] **CSV file reading acceleration** - Performance-optimized data loading
- [x] **Imbalanced data classifiers** - Specialized ML tools for imbalanced datasets
- [x] **Pandas DataFrame data compression** - Memory optimization for large datasets
- [x] **DataFrame insight tools** - Quick data analysis and visualization
- [x] **Large data train-test splitting** - Efficient data partitioning for ML pipelines
- [x] **Intuitive timer** - Simple timing and benchmarking tools

This library is currently undergoing rapid iteration. If you encounter any issues with its functionalities, feel free to [raise an issue](https://github.com/yourusername/spinesUtils/issues).

## Installation

You can install spinesUtils from PyPI:

```bash
pip install spinesUtils
```

## Usage Examples

### Logger

The Logger class provides convenient logging without worrying about handler conflicts with the native Python logging module.

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

### Type Checking and Parameter Validation

Ensure your functions receive the correct input types and values:

```python
from spinesUtils.asserts import *

# Check parameter type
@ParameterTypeAssert({
    'a': (int, float),
    'b': (int, float)
})
def add(a, b):
    return a + b

# Check parameter value
@ParameterValuesAssert({
    'a': lambda x: x > 0,
    'b': lambda x: x > 0
})
def divide(a, b):
    return a / b

# Generate function kwargs
params = generate_function_kwargs(add, a=1, b=2)
```

### CSV Reading Acceleration

Read large CSV files efficiently:

```python
from spinesUtils import read_csv

df = read_csv(
    fp='/path/to/your/file.csv',
    sep=',',  # equal to pandas read_csv.sep
    turbo_method='polars',  # use turbo_method to speed up load time
    chunk_size=None,  # it can be integer if you want to use pandas backend
    transform2low_mem=True,  # compresses file to save memory
    verbose=False
)
```

### Classifiers for Imbalanced Data

Handle imbalanced datasets effectively:

```python
from spinesUtils.models import MultiClassBalanceClassifier
from sklearn.ensemble import RandomForestClassifier

classifier = MultiClassBalanceClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100),
    n_classes=3,
    random_state=0,
    verbose=0
)

# Fit and predict as you would with any scikit-learn estimator
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```

### DataFrame Data Compression

Optimize memory usage for large DataFrames:

```python
from spinesUtils import transform_dtypes_low_mem

# Compress a single DataFrame
transform_dtypes_low_mem(df, verbose=True, inplace=True)

# Batch compress multiple DataFrames
from spinesUtils import transform_batch_dtypes_low_mem
transform_batch_dtypes_low_mem([df1, df2, df3, df4], verbose=True, inplace=True)
```

### DataFrame Insight Tools

Quickly analyze your data:

```python
from spinesUtils import df_preview, classify_samples_dist

# Get comprehensive DataFrame insights
df_insight = df_preview(df)
```

### Data Splitting Utilities

Efficiently split large datasets:

```python
from spinesUtils import train_test_split_bigdata, train_test_split_bigdata_df
from spinesUtils.feature_tools import get_x_cols

# Return numpy arrays
X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_split_bigdata(
    df=df, 
    x_cols=get_x_cols(df, y_col='target_column'),
    y_col='target_column', 
    shuffle=True,
    return_valid=True,
    train_size=0.8,
    valid_size=0.5
)

# Return pandas DataFrames
train_df, valid_df, test_df = train_test_split_bigdata_df(
    df=df, 
    x_cols=get_x_cols(df, y_col='target_column'),
    y_col='target_column', 
    shuffle=True,
    return_valid=True,
    train_size=0.8,
    valid_size=0.5
)
```

### Timer Utility

Time your code execution simply:

```python
from spinesUtils.timer import Timer

# As a context manager
with Timer().session() as t:
    # Your code here
    t.sleep(1)
    print(f"Step 1 time: {t.last_timestamp_diff():.2f}s")
    
    # Mark a middle point
    t.middle_point()
    
    # More code
    t.sleep(2)
    print(f"Step 2 time: {t.last_timestamp_diff():.2f}s")
    
print(f"Total time: {t.total_elapsed_time():.2f}s")

# Or use it manually
timer = Timer()
timer.start()
# Your code here
timer.end()
print(f"Elapsed: {timer.total_elapsed_time():.2f}s")
```
