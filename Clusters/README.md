# Clusters Module

The **Clusters** module is designed to simulate multivariate normal distributions in *n* dimensions. This simulation is particularly useful for training and evaluating unsupervised clustering models, allowing users to understand how these models perform under optimal conditions.

## Requirements

To use this module, you'll need the following Python libraries:

```python
  import numpy as np
  import matplotlib.pyplot as plt
  import plotly.graph_objs as go
  import plotly.express as px
  import pandas as pd
  ```

## Changes Log
#### V0.1 Realese 
- Generate Samples: Simulates random multivariate normal distributions in n dimensions.
- 2D & 3D Plotting: Includes methods for visualizing the generated distributions in both 2D and 3D.
- Data Retrieval: Provides a method to retrieve the generated data as a Pandas DataFrame.
