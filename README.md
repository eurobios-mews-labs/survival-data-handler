# Survival data handler
![code coverage](https://raw.githubusercontent.com/eurobios-mews-labs/survival-data-handler/coverage-badge/coverage.svg?raw=true)
[![PyPI version](https://badge.fury.io/py/palma.svg)](https://badge.fury.io/py/palma)

The aim of this package is to facilitate the use of survival data by switching from temporal data in the form of a
collection of survival functions to temporal matrices calculating other functions derived from survival analysis, 
such as residual life, hazard function, etc.  analysis, such as residual life expectancy, hazard function, etc.

```python
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

rossi = load_rossi()
cph = CoxPHFitter()

cph.fit(rossi, duration_col='week', event_col='arrest')
curves = cph.predict_survival_function(rossi).T
curves.columns = pd.to_timedelta(curves.columns.to_numpy() * 7, unit="D")
print(curves.head())
```

|    |   7 days 00:00:00 |   14 days 00:00:00 |   21 days 00:00:00 |   28 days 00:00:00 |   35 days 00:00:00 |   42 days 00:00:00 |   49 days 00:00:00 |   56 days 00:00:00 |   63 days 00:00:00 |   70 days 00:00:00 |
|---:|------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
|  0 |          0.997616 |           0.99523  |           0.992848 |           0.990468 |           0.988085 |           0.985699 |           0.983305 |           0.971402 |           0.966614 |           0.964223 |
|  1 |          0.993695 |           0.987411 |           0.981162 |           0.974941 |           0.968739 |           0.962552 |           0.95637  |           0.926001 |           0.913958 |           0.907978 |
|  2 |          0.994083 |           0.988183 |           0.982314 |           0.976468 |           0.970639 |           0.96482  |           0.959004 |           0.930402 |           0.919043 |           0.913399 |
|  3 |          0.999045 |           0.998089 |           0.997133 |           0.996176 |           0.995216 |           0.994254 |           0.993287 |           0.98846  |           0.986508 |           0.985531 |
|  4 |          0.997626 |           0.99525  |           0.992878 |           0.990507 |           0.988135 |           0.985758 |           0.983374 |           0.97152  |           0.966752 |           0.96437  |


```python
from survival_data_handler import Lifespan


age = pd.to_timedelta(rossi["age"] * 365.25, unit="D")
birth = pd.to_datetime('2000')
rossi["index"] = rossi.index
    
birth = pd.to_datetime('2000')

lifespan = Lifespan(
    curves,
    index=rossi["index"],
    birth=birth,
    age=age,
    window=(pd.to_datetime("2000"), pd.to_datetime("2001"))
)
```


We now add the supervision data (in the form of duration)

```python    
lifespan.add_supervision(
    event=rossi["arrest"],                                      # True if the data is observed False, when censored
    durations=rossi["duration"] + birth      # The duration
)
```
Let's calculate the associated performance

```python

lifespan.assess_metric("survival_function")
```

| Date       | Performance (1 - AUC) |
|------------|-----------------------|
| 2000-01-31 | 0.468458              |
| 2000-03-01 | 0.384425              |
| 2000-03-31 | 0.432012              |
| 2000-04-30 | 0.357338              |
| 2000-05-30 | 0.365263              |
| 2000-06-29 | 0.365190              |
| 2000-07-29 | 0.371438              |
| 2000-08-28 | 0.343447              |
| 2000-09-27 | 0.340607              |
| 2000-10-27 | 0.344628              |
| 2000-11-26 | 0.334398              |
| 2000-12-26 | 0.334444              |
