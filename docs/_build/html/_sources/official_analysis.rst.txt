This is a time series cluster analysis for train ridership over seven
years. My goal was to compare over 1000 station pairs, check their
patterns over time (42 representative days), cluster them according to
their patterns, and visualize each cluster. I was able to accomplish
this using Python’s Pandas, NumPy, and Scikit-learn.

Installation
------------

Here’s a list of all of the libraries needed to run this Jupyter
Notebook:

.. code:: ipython3

    !pip install pandas
    !pip install numpy
    !pip install holiday
    !pip install datetime
    !pip install matlablib
    !pip install scipy
    !pip install seaborn
    !pip install matplotlib
    !pip install scikit-learn

Imports
~~~~~~~

Here are the imports needed in order to carry out all of the functions
in this notebook:

.. code:: ipython3

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from pandas import DataFrame
    import random
    from sklearn.datasets import make_blobs #for testing
    from scipy.stats import skewnorm
    from sklearn.cluster import KMeans

Data Importing and Cleaning
---------------------------

This section of the notebook imports, cleans, and formats the data into
useable inputs for our K-Means Clustering Algorithm.

.. code:: ipython3

    # imports the data
    cluster_data = pd.read_csv('cluster_data.csv')

.. code:: ipython3

    # displays the imported data as a DataFrame
    cluster_data




.. raw:: html

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
          <th>stnpair</th>
          <th>year</th>
          <th>season</th>
          <th>dow</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
          <th>cluster_input</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Friday</td>
          <td>2.000000</td>
          <td>1320</td>
          <td>0.001515</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Monday</td>
          <td>3.000000</td>
          <td>1320</td>
          <td>0.002273</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Saturday</td>
          <td>9.500000</td>
          <td>1320</td>
          <td>0.007197</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Sunday</td>
          <td>6.000000</td>
          <td>1320</td>
          <td>0.004545</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Thursday</td>
          <td>8.000000</td>
          <td>1320</td>
          <td>0.006061</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>270053</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Saturday</td>
          <td>9.375000</td>
          <td>2150</td>
          <td>0.004360</td>
        </tr>
        <tr>
          <th>270054</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Sunday</td>
          <td>5.714286</td>
          <td>2150</td>
          <td>0.002658</td>
        </tr>
        <tr>
          <th>270055</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Thursday</td>
          <td>5.750000</td>
          <td>2150</td>
          <td>0.002674</td>
        </tr>
        <tr>
          <th>270056</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Tuesday</td>
          <td>5.428571</td>
          <td>2150</td>
          <td>0.002525</td>
        </tr>
        <tr>
          <th>270057</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Wednesday</td>
          <td>5.666667</td>
          <td>2150</td>
          <td>0.002636</td>
        </tr>
      </tbody>
    </table>
    <p>270058 rows × 7 columns</p>
    </div>



.. code:: ipython3

    # Checks which station pairs have over 7 missing days
    df1 = cluster_data[cluster_data['stnpair'].map(cluster_data['stnpair'].value_counts()) < 287]
    df1 = df1['stnpair'].value_counts().reset_index()
    df1.columns = ['stnpair','no. of rep days']

.. code:: ipython3

    # Saves station pairs with missing days to a csv
    df1.to_csv('station_pairs_missing_days', index=False)

.. code:: ipython3

    cluster_data




.. raw:: html

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
          <th>stnpair</th>
          <th>year</th>
          <th>season</th>
          <th>dow</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
          <th>cluster_input</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Friday</td>
          <td>2.000000</td>
          <td>1320</td>
          <td>0.001515</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Monday</td>
          <td>3.000000</td>
          <td>1320</td>
          <td>0.002273</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Saturday</td>
          <td>9.500000</td>
          <td>1320</td>
          <td>0.007197</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Sunday</td>
          <td>6.000000</td>
          <td>1320</td>
          <td>0.004545</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Thursday</td>
          <td>8.000000</td>
          <td>1320</td>
          <td>0.006061</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>270053</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Saturday</td>
          <td>9.375000</td>
          <td>2150</td>
          <td>0.004360</td>
        </tr>
        <tr>
          <th>270054</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Sunday</td>
          <td>5.714286</td>
          <td>2150</td>
          <td>0.002658</td>
        </tr>
        <tr>
          <th>270055</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Thursday</td>
          <td>5.750000</td>
          <td>2150</td>
          <td>0.002674</td>
        </tr>
        <tr>
          <th>270056</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Tuesday</td>
          <td>5.428571</td>
          <td>2150</td>
          <td>0.002525</td>
        </tr>
        <tr>
          <th>270057</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Wednesday</td>
          <td>5.666667</td>
          <td>2150</td>
          <td>0.002636</td>
        </tr>
      </tbody>
    </table>
    <p>270058 rows × 7 columns</p>
    </div>



.. code:: ipython3

    # Creates a df that displays the number of years each station pair is listed for
    number_years_df = cluster_data.groupby("stnpair")["year"].nunique().reset_index()
    number_years_df = number_years_df.reset_index(drop = True)

.. code:: ipython3

    # Displays the station pairs and the number of years they have in the data
    number_years_df




.. raw:: html

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
          <th>stnpair</th>
          <th>year</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>7</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BWI</td>
          <td>2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-MET</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-NWK</td>
          <td>7</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-NYP</td>
          <td>7</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1010</th>
          <td>WAS-YEM</td>
          <td>4</td>
        </tr>
        <tr>
          <th>1011</th>
          <td>WEM-WOB</td>
          <td>7</td>
        </tr>
        <tr>
          <th>1012</th>
          <td>WIL-WLN</td>
          <td>2</td>
        </tr>
        <tr>
          <th>1013</th>
          <td>WPB-WPK</td>
          <td>7</td>
        </tr>
        <tr>
          <th>1014</th>
          <td>WPB-WTH</td>
          <td>7</td>
        </tr>
      </tbody>
    </table>
    <p>1015 rows × 2 columns</p>
    </div>



.. code:: ipython3

    # Finds the average daily ridership and average annual ridership by station pair, season, and dow and stores in a df
    grouped = cluster_data.groupby(['stnpair','season','dow'])['avg_daily_ridership','annual_ridership_at_stnpair'].mean().reset_index()
    grouped 


.. parsed-literal::

    <ipython-input-11-05f35d0dd445>:1: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
      grouped = cluster_data.groupby(['stnpair','season','dow'])['avg_daily_ridership','annual_ridership_at_stnpair'].mean().reset_index()




.. raw:: html

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
          <th>stnpair</th>
          <th>season</th>
          <th>dow</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Friday</td>
          <td>4.071429</td>
          <td>1163.000000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Monday</td>
          <td>2.142857</td>
          <td>1163.000000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Saturday</td>
          <td>8.142857</td>
          <td>1163.000000</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Sunday</td>
          <td>5.142857</td>
          <td>1163.000000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Thursday</td>
          <td>5.285714</td>
          <td>1163.000000</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>42370</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Saturday</td>
          <td>7.144898</td>
          <td>2115.714286</td>
        </tr>
        <tr>
          <th>42371</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Sunday</td>
          <td>6.314909</td>
          <td>2115.714286</td>
        </tr>
        <tr>
          <th>42372</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Thursday</td>
          <td>5.349490</td>
          <td>2115.714286</td>
        </tr>
        <tr>
          <th>42373</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Tuesday</td>
          <td>4.711565</td>
          <td>2115.714286</td>
        </tr>
        <tr>
          <th>42374</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Wednesday</td>
          <td>4.580499</td>
          <td>2115.714286</td>
        </tr>
      </tbody>
    </table>
    <p>42375 rows × 5 columns</p>
    </div>



.. code:: ipython3

    # Creates cluster inputs by dividing daily ridership by annual ridership
    # We want all inputs to be around the same so that we check patterns rather than high vs. low
    
    grouped['avg_cluster_input'] = grouped['avg_daily_ridership']/grouped['annual_ridership_at_stnpair']
    grouped['season_dow'] = grouped['season'] + ' ' + grouped['dow']

.. code:: ipython3

    grouped




.. raw:: html

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
          <th>stnpair</th>
          <th>season</th>
          <th>dow</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
          <th>avg_cluster_input</th>
          <th>season_dow</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Friday</td>
          <td>4.071429</td>
          <td>1163.000000</td>
          <td>0.003501</td>
          <td>Thanksgiving Friday</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Monday</td>
          <td>2.142857</td>
          <td>1163.000000</td>
          <td>0.001843</td>
          <td>Thanksgiving Monday</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Saturday</td>
          <td>8.142857</td>
          <td>1163.000000</td>
          <td>0.007002</td>
          <td>Thanksgiving Saturday</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Sunday</td>
          <td>5.142857</td>
          <td>1163.000000</td>
          <td>0.004422</td>
          <td>Thanksgiving Sunday</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Thursday</td>
          <td>5.285714</td>
          <td>1163.000000</td>
          <td>0.004545</td>
          <td>Thanksgiving Thursday</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>42370</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Saturday</td>
          <td>7.144898</td>
          <td>2115.714286</td>
          <td>0.003377</td>
          <td>winter Saturday</td>
        </tr>
        <tr>
          <th>42371</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Sunday</td>
          <td>6.314909</td>
          <td>2115.714286</td>
          <td>0.002985</td>
          <td>winter Sunday</td>
        </tr>
        <tr>
          <th>42372</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Thursday</td>
          <td>5.349490</td>
          <td>2115.714286</td>
          <td>0.002528</td>
          <td>winter Thursday</td>
        </tr>
        <tr>
          <th>42373</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Tuesday</td>
          <td>4.711565</td>
          <td>2115.714286</td>
          <td>0.002227</td>
          <td>winter Tuesday</td>
        </tr>
        <tr>
          <th>42374</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Wednesday</td>
          <td>4.580499</td>
          <td>2115.714286</td>
          <td>0.002165</td>
          <td>winter Wednesday</td>
        </tr>
      </tbody>
    </table>
    <p>42375 rows × 7 columns</p>
    </div>



.. code:: ipython3

    # Creates df with only the top 5 station city pairs
    top_5 = grouped[grouped["stnpair"].isin(['NYP-WAS','NYG-WAS', 'NYP-PHL','NYG-PHL', 'BOS-NYP','BOS-NYG','BBY-NYP','BBY-NYG', 'PHL-WAS', 'ALB-NYP','ALB-NYG'])]
    top_5




.. raw:: html

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
          <th>stnpair</th>
          <th>season</th>
          <th>dow</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
          <th>avg_cluster_input</th>
          <th>season_dow</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>797</th>
          <td>ALB-NYP</td>
          <td>Thanksgiving</td>
          <td>Friday</td>
          <td>1967.357143</td>
          <td>639312.428571</td>
          <td>0.003077</td>
          <td>Thanksgiving Friday</td>
        </tr>
        <tr>
          <th>798</th>
          <td>ALB-NYP</td>
          <td>Thanksgiving</td>
          <td>Monday</td>
          <td>1896.000000</td>
          <td>639312.428571</td>
          <td>0.002966</td>
          <td>Thanksgiving Monday</td>
        </tr>
        <tr>
          <th>799</th>
          <td>ALB-NYP</td>
          <td>Thanksgiving</td>
          <td>Saturday</td>
          <td>1906.428571</td>
          <td>639312.428571</td>
          <td>0.002982</td>
          <td>Thanksgiving Saturday</td>
        </tr>
        <tr>
          <th>800</th>
          <td>ALB-NYP</td>
          <td>Thanksgiving</td>
          <td>Sunday</td>
          <td>2238.357143</td>
          <td>639312.428571</td>
          <td>0.003501</td>
          <td>Thanksgiving Sunday</td>
        </tr>
        <tr>
          <th>801</th>
          <td>ALB-NYP</td>
          <td>Thanksgiving</td>
          <td>Thursday</td>
          <td>1249.142857</td>
          <td>639312.428571</td>
          <td>0.001954</td>
          <td>Thanksgiving Thursday</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>37865</th>
          <td>PHL-WAS</td>
          <td>winter</td>
          <td>Saturday</td>
          <td>1119.428571</td>
          <td>701509.428571</td>
          <td>0.001596</td>
          <td>winter Saturday</td>
        </tr>
        <tr>
          <th>37866</th>
          <td>PHL-WAS</td>
          <td>winter</td>
          <td>Sunday</td>
          <td>1501.504000</td>
          <td>701509.428571</td>
          <td>0.002140</td>
          <td>winter Sunday</td>
        </tr>
        <tr>
          <th>37867</th>
          <td>PHL-WAS</td>
          <td>winter</td>
          <td>Thursday</td>
          <td>2152.554429</td>
          <td>701509.428571</td>
          <td>0.003068</td>
          <td>winter Thursday</td>
        </tr>
        <tr>
          <th>37868</th>
          <td>PHL-WAS</td>
          <td>winter</td>
          <td>Tuesday</td>
          <td>1799.303571</td>
          <td>701509.428571</td>
          <td>0.002565</td>
          <td>winter Tuesday</td>
        </tr>
        <tr>
          <th>37869</th>
          <td>PHL-WAS</td>
          <td>winter</td>
          <td>Wednesday</td>
          <td>1900.311429</td>
          <td>701509.428571</td>
          <td>0.002709</td>
          <td>winter Wednesday</td>
        </tr>
      </tbody>
    </table>
    <p>210 rows × 7 columns</p>
    </div>



.. code:: ipython3

    # Saves top 5 data to CSV
    top_5.to_csv("top_5_cluster_inputs.csv")

.. code:: ipython3

    # Creates a df with all station pairs except the top 5
    grouped = grouped[~(grouped["stnpair"].isin(['NYP-WAS','NYG-WAS', 'NYP-PHL','NYG-PHL', 'BOS-NYP','BOS-NYG','BBY-NYP','BBY-NYG', 'PHL-WAS', 'ALB-NYP','ALB-NYG']))]
    grouped




.. raw:: html

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
          <th>stnpair</th>
          <th>season</th>
          <th>dow</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
          <th>avg_cluster_input</th>
          <th>season_dow</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Friday</td>
          <td>4.071429</td>
          <td>1163.000000</td>
          <td>0.003501</td>
          <td>Thanksgiving Friday</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Monday</td>
          <td>2.142857</td>
          <td>1163.000000</td>
          <td>0.001843</td>
          <td>Thanksgiving Monday</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Saturday</td>
          <td>8.142857</td>
          <td>1163.000000</td>
          <td>0.007002</td>
          <td>Thanksgiving Saturday</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Sunday</td>
          <td>5.142857</td>
          <td>1163.000000</td>
          <td>0.004422</td>
          <td>Thanksgiving Sunday</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Thursday</td>
          <td>5.285714</td>
          <td>1163.000000</td>
          <td>0.004545</td>
          <td>Thanksgiving Thursday</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>42370</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Saturday</td>
          <td>7.144898</td>
          <td>2115.714286</td>
          <td>0.003377</td>
          <td>winter Saturday</td>
        </tr>
        <tr>
          <th>42371</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Sunday</td>
          <td>6.314909</td>
          <td>2115.714286</td>
          <td>0.002985</td>
          <td>winter Sunday</td>
        </tr>
        <tr>
          <th>42372</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Thursday</td>
          <td>5.349490</td>
          <td>2115.714286</td>
          <td>0.002528</td>
          <td>winter Thursday</td>
        </tr>
        <tr>
          <th>42373</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Tuesday</td>
          <td>4.711565</td>
          <td>2115.714286</td>
          <td>0.002227</td>
          <td>winter Tuesday</td>
        </tr>
        <tr>
          <th>42374</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Wednesday</td>
          <td>4.580499</td>
          <td>2115.714286</td>
          <td>0.002165</td>
          <td>winter Wednesday</td>
        </tr>
      </tbody>
    </table>
    <p>42165 rows × 7 columns</p>
    </div>



.. code:: ipython3

    # Creates a matrix of cluster inputs
    mtx = pd.pivot_table(grouped, values = "avg_cluster_input", columns = "season_dow", index = "stnpair", aggfunc = np.sum )
    mtx = mtx.fillna(0)
    mtx




.. raw:: html

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
          <th>season_dow</th>
          <th>Thanksgiving Friday</th>
          <th>Thanksgiving Monday</th>
          <th>Thanksgiving Saturday</th>
          <th>Thanksgiving Sunday</th>
          <th>Thanksgiving Thursday</th>
          <th>Thanksgiving Tuesday</th>
          <th>Thanksgiving Wednesday</th>
          <th>december Friday</th>
          <th>december Monday</th>
          <th>december Saturday</th>
          <th>...</th>
          <th>summer Thursday</th>
          <th>summer Tuesday</th>
          <th>summer Wednesday</th>
          <th>winter Friday</th>
          <th>winter Monday</th>
          <th>winter Saturday</th>
          <th>winter Sunday</th>
          <th>winter Thursday</th>
          <th>winter Tuesday</th>
          <th>winter Wednesday</th>
        </tr>
        <tr>
          <th>stnpair</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>ABE-BAL</th>
          <td>0.003501</td>
          <td>0.001843</td>
          <td>0.007002</td>
          <td>0.004422</td>
          <td>0.004545</td>
          <td>0.002334</td>
          <td>0.003317</td>
          <td>0.001695</td>
          <td>0.002426</td>
          <td>0.006946</td>
          <td>...</td>
          <td>0.001932</td>
          <td>0.001710</td>
          <td>0.001769</td>
          <td>0.002348</td>
          <td>0.001935</td>
          <td>0.007051</td>
          <td>0.003918</td>
          <td>0.002248</td>
          <td>0.002275</td>
          <td>0.001927</td>
        </tr>
        <tr>
          <th>ABE-BWI</th>
          <td>0.004021</td>
          <td>0.004832</td>
          <td>0.002577</td>
          <td>0.006765</td>
          <td>0.001289</td>
          <td>0.002681</td>
          <td>0.008376</td>
          <td>0.004349</td>
          <td>0.002846</td>
          <td>0.004124</td>
          <td>...</td>
          <td>0.005000</td>
          <td>0.004236</td>
          <td>0.004631</td>
          <td>0.002761</td>
          <td>0.002577</td>
          <td>0.003093</td>
          <td>0.002556</td>
          <td>0.002416</td>
          <td>0.002094</td>
          <td>0.003195</td>
        </tr>
        <tr>
          <th>ABE-MET</th>
          <td>0.005442</td>
          <td>0.002721</td>
          <td>0.002721</td>
          <td>0.001361</td>
          <td>0.009524</td>
          <td>0.001361</td>
          <td>0.005442</td>
          <td>0.005442</td>
          <td>0.004422</td>
          <td>0.003401</td>
          <td>...</td>
          <td>0.004453</td>
          <td>0.002968</td>
          <td>0.002721</td>
          <td>0.004276</td>
          <td>0.005669</td>
          <td>0.004354</td>
          <td>0.002494</td>
          <td>0.003571</td>
          <td>0.003810</td>
          <td>0.002948</td>
        </tr>
        <tr>
          <th>ABE-NWK</th>
          <td>0.002869</td>
          <td>0.004341</td>
          <td>0.002487</td>
          <td>0.003090</td>
          <td>0.005003</td>
          <td>0.005304</td>
          <td>0.006164</td>
          <td>0.003181</td>
          <td>0.004341</td>
          <td>0.002060</td>
          <td>...</td>
          <td>0.004726</td>
          <td>0.003316</td>
          <td>0.004331</td>
          <td>0.003936</td>
          <td>0.003213</td>
          <td>0.002012</td>
          <td>0.002173</td>
          <td>0.004452</td>
          <td>0.004721</td>
          <td>0.004175</td>
        </tr>
        <tr>
          <th>ABE-NYP</th>
          <td>0.003743</td>
          <td>0.003503</td>
          <td>0.003734</td>
          <td>0.003957</td>
          <td>0.004225</td>
          <td>0.004189</td>
          <td>0.006996</td>
          <td>0.004120</td>
          <td>0.003686</td>
          <td>0.003988</td>
          <td>...</td>
          <td>0.003061</td>
          <td>0.002600</td>
          <td>0.002869</td>
          <td>0.002846</td>
          <td>0.002622</td>
          <td>0.001897</td>
          <td>0.002053</td>
          <td>0.002572</td>
          <td>0.002313</td>
          <td>0.002510</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>WAS-YEM</th>
          <td>0.003510</td>
          <td>0.005996</td>
          <td>0.005265</td>
          <td>0.005805</td>
          <td>0.001121</td>
          <td>0.008482</td>
          <td>0.015794</td>
          <td>0.003939</td>
          <td>0.002486</td>
          <td>0.002510</td>
          <td>...</td>
          <td>0.005120</td>
          <td>0.003483</td>
          <td>0.004937</td>
          <td>0.003826</td>
          <td>0.002306</td>
          <td>0.002525</td>
          <td>0.003471</td>
          <td>0.002859</td>
          <td>0.002428</td>
          <td>0.002554</td>
        </tr>
        <tr>
          <th>WEM-WOB</th>
          <td>0.006303</td>
          <td>0.002029</td>
          <td>0.002508</td>
          <td>0.002508</td>
          <td>0.005941</td>
          <td>0.002898</td>
          <td>0.008115</td>
          <td>0.004183</td>
          <td>0.002789</td>
          <td>0.003821</td>
          <td>...</td>
          <td>0.005497</td>
          <td>0.004266</td>
          <td>0.004699</td>
          <td>0.003091</td>
          <td>0.002451</td>
          <td>0.002581</td>
          <td>0.002897</td>
          <td>0.002144</td>
          <td>0.002012</td>
          <td>0.002267</td>
        </tr>
        <tr>
          <th>WIL-WLN</th>
          <td>0.004829</td>
          <td>0.004829</td>
          <td>0.006439</td>
          <td>0.009015</td>
          <td>0.001332</td>
          <td>0.007989</td>
          <td>0.007083</td>
          <td>0.003767</td>
          <td>0.004293</td>
          <td>0.002576</td>
          <td>...</td>
          <td>0.004090</td>
          <td>0.004341</td>
          <td>0.004037</td>
          <td>0.003955</td>
          <td>0.002743</td>
          <td>0.003134</td>
          <td>0.003917</td>
          <td>0.002783</td>
          <td>0.002189</td>
          <td>0.002468</td>
        </tr>
        <tr>
          <th>WPB-WPK</th>
          <td>0.004834</td>
          <td>0.003946</td>
          <td>0.006018</td>
          <td>0.007941</td>
          <td>0.007695</td>
          <td>0.012035</td>
          <td>0.007991</td>
          <td>0.003377</td>
          <td>0.003157</td>
          <td>0.003977</td>
          <td>...</td>
          <td>0.004363</td>
          <td>0.002669</td>
          <td>0.003141</td>
          <td>0.003167</td>
          <td>0.003042</td>
          <td>0.003491</td>
          <td>0.002751</td>
          <td>0.003109</td>
          <td>0.002250</td>
          <td>0.002634</td>
        </tr>
        <tr>
          <th>WPB-WTH</th>
          <td>0.004288</td>
          <td>0.003984</td>
          <td>0.005571</td>
          <td>0.004220</td>
          <td>0.007427</td>
          <td>0.005267</td>
          <td>0.003976</td>
          <td>0.003341</td>
          <td>0.002745</td>
          <td>0.004276</td>
          <td>...</td>
          <td>0.003916</td>
          <td>0.002933</td>
          <td>0.002686</td>
          <td>0.002838</td>
          <td>0.002487</td>
          <td>0.003377</td>
          <td>0.002985</td>
          <td>0.002528</td>
          <td>0.002227</td>
          <td>0.002165</td>
        </tr>
      </tbody>
    </table>
    <p>1010 rows × 42 columns</p>
    </div>



.. code:: ipython3

    # Makes a numerical value into a string value in order to group by later
    grouped['annual_ridership_at_stnpair_str'] = grouped['annual_ridership_at_stnpair'].astype(str)


.. parsed-literal::

    <ipython-input-22-b4a6057d4306>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      grouped['annual_ridership_at_stnpair_str'] = grouped['annual_ridership_at_stnpair'].astype(str)


.. code:: ipython3

    grouped




.. raw:: html

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
          <th>stnpair</th>
          <th>season</th>
          <th>dow</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
          <th>avg_cluster_input</th>
          <th>season_dow</th>
          <th>annual_ridership_at_stnpair_str</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Friday</td>
          <td>4.071429</td>
          <td>1163.000000</td>
          <td>0.003501</td>
          <td>Thanksgiving Friday</td>
          <td>1163.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Monday</td>
          <td>2.142857</td>
          <td>1163.000000</td>
          <td>0.001843</td>
          <td>Thanksgiving Monday</td>
          <td>1163.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Saturday</td>
          <td>8.142857</td>
          <td>1163.000000</td>
          <td>0.007002</td>
          <td>Thanksgiving Saturday</td>
          <td>1163.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Sunday</td>
          <td>5.142857</td>
          <td>1163.000000</td>
          <td>0.004422</td>
          <td>Thanksgiving Sunday</td>
          <td>1163.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-BAL</td>
          <td>Thanksgiving</td>
          <td>Thursday</td>
          <td>5.285714</td>
          <td>1163.000000</td>
          <td>0.004545</td>
          <td>Thanksgiving Thursday</td>
          <td>1163.0</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>42370</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Saturday</td>
          <td>7.144898</td>
          <td>2115.714286</td>
          <td>0.003377</td>
          <td>winter Saturday</td>
          <td>2115.714285714286</td>
        </tr>
        <tr>
          <th>42371</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Sunday</td>
          <td>6.314909</td>
          <td>2115.714286</td>
          <td>0.002985</td>
          <td>winter Sunday</td>
          <td>2115.714285714286</td>
        </tr>
        <tr>
          <th>42372</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Thursday</td>
          <td>5.349490</td>
          <td>2115.714286</td>
          <td>0.002528</td>
          <td>winter Thursday</td>
          <td>2115.714285714286</td>
        </tr>
        <tr>
          <th>42373</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Tuesday</td>
          <td>4.711565</td>
          <td>2115.714286</td>
          <td>0.002227</td>
          <td>winter Tuesday</td>
          <td>2115.714285714286</td>
        </tr>
        <tr>
          <th>42374</th>
          <td>WPB-WTH</td>
          <td>winter</td>
          <td>Wednesday</td>
          <td>4.580499</td>
          <td>2115.714286</td>
          <td>0.002165</td>
          <td>winter Wednesday</td>
          <td>2115.714285714286</td>
        </tr>
      </tbody>
    </table>
    <p>42165 rows × 8 columns</p>
    </div>



.. code:: ipython3

    # Makes a numerical value into a string value in order to group by later
    cluster_data['annual_ridership_at_stnpair_str'] = cluster_data['annual_ridership_at_stnpair'].astype(str)

.. code:: ipython3

    cluster_data




.. raw:: html

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
          <th>stnpair</th>
          <th>year</th>
          <th>season</th>
          <th>dow</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
          <th>cluster_input</th>
          <th>annual_ridership_at_stnpair_str</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Friday</td>
          <td>2.000000</td>
          <td>1320</td>
          <td>0.001515</td>
          <td>1320</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Monday</td>
          <td>3.000000</td>
          <td>1320</td>
          <td>0.002273</td>
          <td>1320</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Saturday</td>
          <td>9.500000</td>
          <td>1320</td>
          <td>0.007197</td>
          <td>1320</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Sunday</td>
          <td>6.000000</td>
          <td>1320</td>
          <td>0.004545</td>
          <td>1320</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>Thanksgiving</td>
          <td>Thursday</td>
          <td>8.000000</td>
          <td>1320</td>
          <td>0.006061</td>
          <td>1320</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>270053</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Saturday</td>
          <td>9.375000</td>
          <td>2150</td>
          <td>0.004360</td>
          <td>2150</td>
        </tr>
        <tr>
          <th>270054</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Sunday</td>
          <td>5.714286</td>
          <td>2150</td>
          <td>0.002658</td>
          <td>2150</td>
        </tr>
        <tr>
          <th>270055</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Thursday</td>
          <td>5.750000</td>
          <td>2150</td>
          <td>0.002674</td>
          <td>2150</td>
        </tr>
        <tr>
          <th>270056</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Tuesday</td>
          <td>5.428571</td>
          <td>2150</td>
          <td>0.002525</td>
          <td>2150</td>
        </tr>
        <tr>
          <th>270057</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>winter</td>
          <td>Wednesday</td>
          <td>5.666667</td>
          <td>2150</td>
          <td>0.002636</td>
          <td>2150</td>
        </tr>
      </tbody>
    </table>
    <p>270058 rows × 8 columns</p>
    </div>



.. code:: ipython3

    # Groups by station pair, year, and annual ridership to create weighted df
    for_weights = cluster_data.groupby(['stnpair','year','annual_ridership_at_stnpair_str']).sum().reset_index()

.. code:: ipython3

    for_weights




.. raw:: html

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
          <th>stnpair</th>
          <th>year</th>
          <th>annual_ridership_at_stnpair_str</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
          <th>cluster_input</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>1320</td>
          <td>177.802453</td>
          <td>55440</td>
          <td>0.134699</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BAL</td>
          <td>2014</td>
          <td>1305</td>
          <td>174.763906</td>
          <td>54810</td>
          <td>0.133919</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-BAL</td>
          <td>2015</td>
          <td>1179</td>
          <td>172.376227</td>
          <td>49518</td>
          <td>0.146205</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-BAL</td>
          <td>2016</td>
          <td>1056</td>
          <td>151.674192</td>
          <td>44352</td>
          <td>0.143631</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-BAL</td>
          <td>2017</td>
          <td>1130</td>
          <td>158.849459</td>
          <td>47460</td>
          <td>0.140575</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>6512</th>
          <td>WPB-WTH</td>
          <td>2015</td>
          <td>2146</td>
          <td>287.479564</td>
          <td>90132</td>
          <td>0.133961</td>
        </tr>
        <tr>
          <th>6513</th>
          <td>WPB-WTH</td>
          <td>2016</td>
          <td>1984</td>
          <td>288.904095</td>
          <td>81344</td>
          <td>0.145617</td>
        </tr>
        <tr>
          <th>6514</th>
          <td>WPB-WTH</td>
          <td>2017</td>
          <td>2161</td>
          <td>312.784246</td>
          <td>90762</td>
          <td>0.144741</td>
        </tr>
        <tr>
          <th>6515</th>
          <td>WPB-WTH</td>
          <td>2018</td>
          <td>1915</td>
          <td>273.227127</td>
          <td>80430</td>
          <td>0.142677</td>
        </tr>
        <tr>
          <th>6516</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>2150</td>
          <td>307.672857</td>
          <td>90300</td>
          <td>0.143104</td>
        </tr>
      </tbody>
    </table>
    <p>6517 rows × 6 columns</p>
    </div>



.. code:: ipython3

    # Changes annual ridership column
    for_weights['annual_ridership_at_stnpair'] = for_weights['annual_ridership_at_stnpair_str'].astype(int)

.. code:: ipython3

    for_weights




.. raw:: html

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
          <th>stnpair</th>
          <th>year</th>
          <th>annual_ridership_at_stnpair_str</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
          <th>cluster_input</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>2013</td>
          <td>1320</td>
          <td>177.802453</td>
          <td>1320</td>
          <td>0.134699</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BAL</td>
          <td>2014</td>
          <td>1305</td>
          <td>174.763906</td>
          <td>1305</td>
          <td>0.133919</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-BAL</td>
          <td>2015</td>
          <td>1179</td>
          <td>172.376227</td>
          <td>1179</td>
          <td>0.146205</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-BAL</td>
          <td>2016</td>
          <td>1056</td>
          <td>151.674192</td>
          <td>1056</td>
          <td>0.143631</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-BAL</td>
          <td>2017</td>
          <td>1130</td>
          <td>158.849459</td>
          <td>1130</td>
          <td>0.140575</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>6512</th>
          <td>WPB-WTH</td>
          <td>2015</td>
          <td>2146</td>
          <td>287.479564</td>
          <td>2146</td>
          <td>0.133961</td>
        </tr>
        <tr>
          <th>6513</th>
          <td>WPB-WTH</td>
          <td>2016</td>
          <td>1984</td>
          <td>288.904095</td>
          <td>1984</td>
          <td>0.145617</td>
        </tr>
        <tr>
          <th>6514</th>
          <td>WPB-WTH</td>
          <td>2017</td>
          <td>2161</td>
          <td>312.784246</td>
          <td>2161</td>
          <td>0.144741</td>
        </tr>
        <tr>
          <th>6515</th>
          <td>WPB-WTH</td>
          <td>2018</td>
          <td>1915</td>
          <td>273.227127</td>
          <td>1915</td>
          <td>0.142677</td>
        </tr>
        <tr>
          <th>6516</th>
          <td>WPB-WTH</td>
          <td>2019</td>
          <td>2150</td>
          <td>307.672857</td>
          <td>2150</td>
          <td>0.143104</td>
        </tr>
      </tbody>
    </table>
    <p>6517 rows × 6 columns</p>
    </div>



.. code:: ipython3

    # Sums up total ridership (based on avg yearly ridership) over all years present in data
    for_weights = for_weights.groupby(['stnpair'])['annual_ridership_at_stnpair'].sum().reset_index()

.. code:: ipython3

    for_weights




.. raw:: html

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
          <th>stnpair</th>
          <th>annual_ridership_at_stnpair</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>8141</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BWI</td>
          <td>1552</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-MET</td>
          <td>735</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-NWK</td>
          <td>6796</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-NYP</td>
          <td>110058</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1010</th>
          <td>WAS-YEM</td>
          <td>3419</td>
        </tr>
        <tr>
          <th>1011</th>
          <td>WEM-WOB</td>
          <td>6901</td>
        </tr>
        <tr>
          <th>1012</th>
          <td>WIL-WLN</td>
          <td>1553</td>
        </tr>
        <tr>
          <th>1013</th>
          <td>WPB-WPK</td>
          <td>10137</td>
        </tr>
        <tr>
          <th>1014</th>
          <td>WPB-WTH</td>
          <td>14810</td>
        </tr>
      </tbody>
    </table>
    <p>1015 rows × 2 columns</p>
    </div>



.. code:: ipython3

    # Finds avg annual ridership per year
    for_weights['weights'] = for_weights['annual_ridership_at_stnpair']/number_years_df['year']

.. code:: ipython3

    # Creates weights df for top 5 city pairs
    for_weights_top5 = for_weights[for_weights["stnpair"].isin(['NYP-WAS','NYG-WAS', 'NYP-PHL','NYG-PHL', 'BOS-NYP','BOS-NYG','BBY-NYP','BBY-NYG', 'PHL-WAS', 'ALB-NYP','ALB-NYG'])]
    for_weights_top5




.. raw:: html

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
          <th>stnpair</th>
          <th>annual_ridership_at_stnpair</th>
          <th>weights</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>19</th>
          <td>ALB-NYP</td>
          <td>4475187</td>
          <td>6.393124e+05</td>
        </tr>
        <tr>
          <th>202</th>
          <td>BOS-NYP</td>
          <td>8219188</td>
          <td>1.174170e+06</td>
        </tr>
        <tr>
          <th>811</th>
          <td>NYP-PHL</td>
          <td>11628920</td>
          <td>1.661274e+06</td>
        </tr>
        <tr>
          <th>847</th>
          <td>NYP-WAS</td>
          <td>15532938</td>
          <td>2.218991e+06</td>
        </tr>
        <tr>
          <th>906</th>
          <td>PHL-WAS</td>
          <td>4910566</td>
          <td>7.015094e+05</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Creates weights table without top 5 city pairs
    for_weights = for_weights[~(for_weights["stnpair"].isin(['NYP-WAS','NYG-WAS', 'NYP-PHL','NYG-PHL', 'BOS-NYP','BOS-NYG','BBY-NYP','BBY-NYG', 'PHL-WAS', 'ALB-NYP','ALB-NYG']))]

.. code:: ipython3

    for_weights




.. raw:: html

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
          <th>stnpair</th>
          <th>annual_ridership_at_stnpair</th>
          <th>weights</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>8141</td>
          <td>1163.000000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BWI</td>
          <td>1552</td>
          <td>776.000000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-MET</td>
          <td>735</td>
          <td>735.000000</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-NWK</td>
          <td>6796</td>
          <td>970.857143</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-NYP</td>
          <td>110058</td>
          <td>15722.571429</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1010</th>
          <td>WAS-YEM</td>
          <td>3419</td>
          <td>854.750000</td>
        </tr>
        <tr>
          <th>1011</th>
          <td>WEM-WOB</td>
          <td>6901</td>
          <td>985.857143</td>
        </tr>
        <tr>
          <th>1012</th>
          <td>WIL-WLN</td>
          <td>1553</td>
          <td>776.500000</td>
        </tr>
        <tr>
          <th>1013</th>
          <td>WPB-WPK</td>
          <td>10137</td>
          <td>1448.142857</td>
        </tr>
        <tr>
          <th>1014</th>
          <td>WPB-WTH</td>
          <td>14810</td>
          <td>2115.714286</td>
        </tr>
      </tbody>
    </table>
    <p>1010 rows × 3 columns</p>
    </div>



.. code:: ipython3

    # Defines lists for weights and their corresponding station pairs
    weights = for_weights['weights']
    stnpairs = for_weights['stnpair']
    weights_top5 = for_weights_top5['weights']
    stnpairs_top5 = for_weights_top5['stnpair']

.. code:: ipython3

    # Saves matrix to csv
    mtx.to_csv('cluster_mtx.csv',index=False)

.. code:: ipython3

    # Saves weights to csv
    for_weights.to_csv('cluster_weights.csv',index=False)


Cluster Analysis
----------------

In this section of the notebook, we are conducting the K-Means Cluster
Analysis using scikit-learn.

.. code:: ipython3

    def cluster(min_clusters, max_clusters):
        """
            Perform multiple k-means cluster analyses.
         
            Conducts several weighted time-series k-means cluster analyses starting from a
            minimum amount of clusters to a maximum amount of clusters.
         
            Parameters
            ----------
            min_clusters : int
                The minimum number of clusters we want to perform our analysis on.
            max_clusters : int
                The maximum number of clusters we want to perform our analysis on.
            
            Returns
            --------
            inertias : list[float]
                A list of inertias for each cluster analysis.
            centroids : list[list[list[float]]]
                A list of centroids for each cluster analysis (each centroid is a list of lists).
            labels : list[list[int]]
                A list of each cluster number label for each cluster analysis.
            cluster_amt : list[int]
                A list of the number of clusters in each analysis.
            
        
        """
        
        inertias = []
        centroids =[]
        labels = []
        cluster_amt = []
    
        for n in range(min_clusters,max_clusters):
            kmeans = KMeans(n_clusters= n,random_state=280,n_init=1000).fit(mtx, y=0, sample_weight=weights)    
            centroid = kmeans.cluster_centers_
            label = kmeans.fit_predict(mtx,y=0, sample_weight=weights)
            inertias.append(kmeans.inertia_)
            centroids.append(centroid)
            labels.append(label) 
            cluster_amt.append(n)
            
        return inertias, centroids, labels, cluster_amt   

.. code:: ipython3

    inertias, centroids, labels, cluster_amt = cluster(4, 21)

.. code:: ipython3

    # Creates a df with the amount of clusters in the cluster analysis and the inertia generated
    # from the run
    inertia_df = pd.DataFrame()
    inertia_df['cluster num'] = cluster_amt
    inertia_df['inertias'] = inertias

.. code:: ipython3

    inertia_df




.. raw:: html

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
          <th>cluster num</th>
          <th>inertias</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>4</td>
          <td>206.852670</td>
        </tr>
        <tr>
          <th>1</th>
          <td>5</td>
          <td>185.869416</td>
        </tr>
        <tr>
          <th>2</th>
          <td>6</td>
          <td>175.386314</td>
        </tr>
        <tr>
          <th>3</th>
          <td>7</td>
          <td>167.080780</td>
        </tr>
        <tr>
          <th>4</th>
          <td>8</td>
          <td>158.783433</td>
        </tr>
        <tr>
          <th>5</th>
          <td>9</td>
          <td>152.723154</td>
        </tr>
        <tr>
          <th>6</th>
          <td>10</td>
          <td>148.728841</td>
        </tr>
        <tr>
          <th>7</th>
          <td>11</td>
          <td>144.312751</td>
        </tr>
        <tr>
          <th>8</th>
          <td>12</td>
          <td>141.398755</td>
        </tr>
        <tr>
          <th>9</th>
          <td>13</td>
          <td>136.865204</td>
        </tr>
        <tr>
          <th>10</th>
          <td>14</td>
          <td>133.383922</td>
        </tr>
        <tr>
          <th>11</th>
          <td>15</td>
          <td>130.760852</td>
        </tr>
        <tr>
          <th>12</th>
          <td>16</td>
          <td>127.343254</td>
        </tr>
        <tr>
          <th>13</th>
          <td>17</td>
          <td>126.418966</td>
        </tr>
        <tr>
          <th>14</th>
          <td>18</td>
          <td>123.643902</td>
        </tr>
        <tr>
          <th>15</th>
          <td>19</td>
          <td>122.289794</td>
        </tr>
        <tr>
          <th>16</th>
          <td>20</td>
          <td>119.826527</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Saves inertia df to csv
    inertia_df.to_csv("cluster_inertias.csv")

.. code:: ipython3

    len(centroids[14]) #checks the length of centroids for 18 clusters




.. parsed-literal::

    18



.. code:: ipython3

    centroids[14] # lists the centroids for 18 clusters

.. code:: ipython3

    def create_inertia_plot(inertia_df):
        """
            Creates an inertia plot.
         
            Creates a plot displaying the inertia for each cluster analysis in a given 
            inertia dataframe.
         
            Parameters
            ----------
            inertia_df : DataFrame
                A DataFrame that contains the cluster analysis number and their inertias.
            
            Generates a plot of inertias for every cluster analysis.
            
        """
        plt.subplots(figsize=(10,7))
        x_coordinates = inertia_df['cluster num']
    
        y1_coordinates = inertia_df['inertias']
        y2_coordinates = inertia_df['inertias']
    
        plt.title('Inertia Plot')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
    
    
        plt.plot(x_coordinates, y1_coordinates)
        plt.scatter(x_coordinates, y2_coordinates)
        plt.savefig('Inertias.png')
    
        plt.show()

.. code:: ipython3

    # Creates inertia plot
    # We want low inertia and a low number of clusters
    
    create_inertia_plot(inertia_df)



.. image:: official_analysis_files/official_analysis_47_0.png


Cleaning and Exporting Results
------------------------------

After running several cluster analyses with our cluster function and
reviewing the inertia plot, we decided on 18 clusters as the official
cluster analysis that we will be using for our results. In this section
of the notebook, I am parsing out and cleaning all of the data related
to the analysis with 18 clusters. After isolating the relevant data, I
am creating a dataframe to store and export the relevant and important
data.

.. code:: ipython3

    labels[14] # Stores the labels of all clusters in 18 cluster analysis




.. parsed-literal::

    array([ 9,  4, 12, ..., 13, 13, 12])



.. code:: ipython3

    # Adds cluster label to each station pair
    for_weights['cluster_num'] = labels[14]
    for_weights




.. raw:: html

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
          <th>stnpair</th>
          <th>annual_ridership_at_stnpair</th>
          <th>weights</th>
          <th>cluster_num</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ABE-BAL</td>
          <td>8141</td>
          <td>1163.000000</td>
          <td>9</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ABE-BWI</td>
          <td>1552</td>
          <td>776.000000</td>
          <td>4</td>
        </tr>
        <tr>
          <th>2</th>
          <td>ABE-MET</td>
          <td>735</td>
          <td>735.000000</td>
          <td>12</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ABE-NWK</td>
          <td>6796</td>
          <td>970.857143</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ABE-NYP</td>
          <td>110058</td>
          <td>15722.571429</td>
          <td>3</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1010</th>
          <td>WAS-YEM</td>
          <td>3419</td>
          <td>854.750000</td>
          <td>6</td>
        </tr>
        <tr>
          <th>1011</th>
          <td>WEM-WOB</td>
          <td>6901</td>
          <td>985.857143</td>
          <td>12</td>
        </tr>
        <tr>
          <th>1012</th>
          <td>WIL-WLN</td>
          <td>1553</td>
          <td>776.500000</td>
          <td>13</td>
        </tr>
        <tr>
          <th>1013</th>
          <td>WPB-WPK</td>
          <td>10137</td>
          <td>1448.142857</td>
          <td>13</td>
        </tr>
        <tr>
          <th>1014</th>
          <td>WPB-WTH</td>
          <td>14810</td>
          <td>2115.714286</td>
          <td>12</td>
        </tr>
      </tbody>
    </table>
    <p>1010 rows × 4 columns</p>
    </div>



.. code:: ipython3

    # Sorts values based on their cluster number
    clusters_by_stnpair = for_weights.sort_values(by=['cluster_num'])
    clusters_by_stnpair




.. raw:: html

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
          <th>stnpair</th>
          <th>annual_ridership_at_stnpair</th>
          <th>weights</th>
          <th>cluster_num</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>865</th>
          <td>ORB-POR</td>
          <td>10090</td>
          <td>1441.428571</td>
          <td>0</td>
        </tr>
        <tr>
          <th>182</th>
          <td>BON-ORB</td>
          <td>48894</td>
          <td>6984.857143</td>
          <td>0</td>
        </tr>
        <tr>
          <th>741</th>
          <td>NHV-STM</td>
          <td>23155</td>
          <td>3307.857143</td>
          <td>1</td>
        </tr>
        <tr>
          <th>480</th>
          <td>EXT-NYP</td>
          <td>258360</td>
          <td>36908.571429</td>
          <td>1</td>
        </tr>
        <tr>
          <th>730</th>
          <td>NHV-NLC</td>
          <td>66909</td>
          <td>9558.428571</td>
          <td>1</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>540</th>
          <td>HAR-LNC</td>
          <td>490659</td>
          <td>70094.142857</td>
          <td>17</td>
        </tr>
        <tr>
          <th>541</th>
          <td>HAR-MID</td>
          <td>33399</td>
          <td>4771.285714</td>
          <td>17</td>
        </tr>
        <tr>
          <th>542</th>
          <td>HAR-MJY</td>
          <td>229810</td>
          <td>32830.000000</td>
          <td>17</td>
        </tr>
        <tr>
          <th>546</th>
          <td>HAR-PAR</td>
          <td>28925</td>
          <td>4132.142857</td>
          <td>17</td>
        </tr>
        <tr>
          <th>786</th>
          <td>NWK-PAO</td>
          <td>37292</td>
          <td>5327.428571</td>
          <td>17</td>
        </tr>
      </tbody>
    </table>
    <p>1010 rows × 4 columns</p>
    </div>



.. code:: ipython3

    # Puts station pairs into respective clusters based on the 18 cluster analysis
    
    cluster_1 = mtx[labels[14] == 0]
    cluster_2 = mtx[labels[14] == 1]
    cluster_3 = mtx[labels[14] == 2]
    cluster_4 = mtx[labels[14] == 3]
    cluster_5 = mtx[labels[14] == 4]
    cluster_6 = mtx[labels[14] == 5]
    cluster_7 = mtx[labels[14] == 6]
    cluster_8 = mtx[labels[14] == 7]
    cluster_9 = mtx[labels[14] == 8]
    cluster_10 = mtx[labels[14] == 9]
    cluster_11 = mtx[labels[14] == 10]
    cluster_12 = mtx[labels[14] == 11]
    cluster_13 = mtx[labels[14] == 12]
    cluster_14 = mtx[labels[14] == 13]
    cluster_15 = mtx[labels[14] == 14]
    cluster_16 = mtx[labels[14] == 15]
    cluster_17 = mtx[labels[14] == 16]
    cluster_18 = mtx[labels[14] == 17]

.. code:: ipython3

    mtx




.. raw:: html

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
          <th>season_dow</th>
          <th>Thanksgiving Friday</th>
          <th>Thanksgiving Monday</th>
          <th>Thanksgiving Saturday</th>
          <th>Thanksgiving Sunday</th>
          <th>Thanksgiving Thursday</th>
          <th>Thanksgiving Tuesday</th>
          <th>Thanksgiving Wednesday</th>
          <th>december Friday</th>
          <th>december Monday</th>
          <th>december Saturday</th>
          <th>...</th>
          <th>summer Thursday</th>
          <th>summer Tuesday</th>
          <th>summer Wednesday</th>
          <th>winter Friday</th>
          <th>winter Monday</th>
          <th>winter Saturday</th>
          <th>winter Sunday</th>
          <th>winter Thursday</th>
          <th>winter Tuesday</th>
          <th>winter Wednesday</th>
        </tr>
        <tr>
          <th>stnpair</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>ABE-BAL</th>
          <td>0.003501</td>
          <td>0.001843</td>
          <td>0.007002</td>
          <td>0.004422</td>
          <td>0.004545</td>
          <td>0.002334</td>
          <td>0.003317</td>
          <td>0.001695</td>
          <td>0.002426</td>
          <td>0.006946</td>
          <td>...</td>
          <td>0.001932</td>
          <td>0.001710</td>
          <td>0.001769</td>
          <td>0.002348</td>
          <td>0.001935</td>
          <td>0.007051</td>
          <td>0.003918</td>
          <td>0.002248</td>
          <td>0.002275</td>
          <td>0.001927</td>
        </tr>
        <tr>
          <th>ABE-BWI</th>
          <td>0.004021</td>
          <td>0.004832</td>
          <td>0.002577</td>
          <td>0.006765</td>
          <td>0.001289</td>
          <td>0.002681</td>
          <td>0.008376</td>
          <td>0.004349</td>
          <td>0.002846</td>
          <td>0.004124</td>
          <td>...</td>
          <td>0.005000</td>
          <td>0.004236</td>
          <td>0.004631</td>
          <td>0.002761</td>
          <td>0.002577</td>
          <td>0.003093</td>
          <td>0.002556</td>
          <td>0.002416</td>
          <td>0.002094</td>
          <td>0.003195</td>
        </tr>
        <tr>
          <th>ABE-MET</th>
          <td>0.005442</td>
          <td>0.002721</td>
          <td>0.002721</td>
          <td>0.001361</td>
          <td>0.009524</td>
          <td>0.001361</td>
          <td>0.005442</td>
          <td>0.005442</td>
          <td>0.004422</td>
          <td>0.003401</td>
          <td>...</td>
          <td>0.004453</td>
          <td>0.002968</td>
          <td>0.002721</td>
          <td>0.004276</td>
          <td>0.005669</td>
          <td>0.004354</td>
          <td>0.002494</td>
          <td>0.003571</td>
          <td>0.003810</td>
          <td>0.002948</td>
        </tr>
        <tr>
          <th>ABE-NWK</th>
          <td>0.002869</td>
          <td>0.004341</td>
          <td>0.002487</td>
          <td>0.003090</td>
          <td>0.005003</td>
          <td>0.005304</td>
          <td>0.006164</td>
          <td>0.003181</td>
          <td>0.004341</td>
          <td>0.002060</td>
          <td>...</td>
          <td>0.004726</td>
          <td>0.003316</td>
          <td>0.004331</td>
          <td>0.003936</td>
          <td>0.003213</td>
          <td>0.002012</td>
          <td>0.002173</td>
          <td>0.004452</td>
          <td>0.004721</td>
          <td>0.004175</td>
        </tr>
        <tr>
          <th>ABE-NYP</th>
          <td>0.003743</td>
          <td>0.003503</td>
          <td>0.003734</td>
          <td>0.003957</td>
          <td>0.004225</td>
          <td>0.004189</td>
          <td>0.006996</td>
          <td>0.004120</td>
          <td>0.003686</td>
          <td>0.003988</td>
          <td>...</td>
          <td>0.003061</td>
          <td>0.002600</td>
          <td>0.002869</td>
          <td>0.002846</td>
          <td>0.002622</td>
          <td>0.001897</td>
          <td>0.002053</td>
          <td>0.002572</td>
          <td>0.002313</td>
          <td>0.002510</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>WAS-YEM</th>
          <td>0.003510</td>
          <td>0.005996</td>
          <td>0.005265</td>
          <td>0.005805</td>
          <td>0.001121</td>
          <td>0.008482</td>
          <td>0.015794</td>
          <td>0.003939</td>
          <td>0.002486</td>
          <td>0.002510</td>
          <td>...</td>
          <td>0.005120</td>
          <td>0.003483</td>
          <td>0.004937</td>
          <td>0.003826</td>
          <td>0.002306</td>
          <td>0.002525</td>
          <td>0.003471</td>
          <td>0.002859</td>
          <td>0.002428</td>
          <td>0.002554</td>
        </tr>
        <tr>
          <th>WEM-WOB</th>
          <td>0.006303</td>
          <td>0.002029</td>
          <td>0.002508</td>
          <td>0.002508</td>
          <td>0.005941</td>
          <td>0.002898</td>
          <td>0.008115</td>
          <td>0.004183</td>
          <td>0.002789</td>
          <td>0.003821</td>
          <td>...</td>
          <td>0.005497</td>
          <td>0.004266</td>
          <td>0.004699</td>
          <td>0.003091</td>
          <td>0.002451</td>
          <td>0.002581</td>
          <td>0.002897</td>
          <td>0.002144</td>
          <td>0.002012</td>
          <td>0.002267</td>
        </tr>
        <tr>
          <th>WIL-WLN</th>
          <td>0.004829</td>
          <td>0.004829</td>
          <td>0.006439</td>
          <td>0.009015</td>
          <td>0.001332</td>
          <td>0.007989</td>
          <td>0.007083</td>
          <td>0.003767</td>
          <td>0.004293</td>
          <td>0.002576</td>
          <td>...</td>
          <td>0.004090</td>
          <td>0.004341</td>
          <td>0.004037</td>
          <td>0.003955</td>
          <td>0.002743</td>
          <td>0.003134</td>
          <td>0.003917</td>
          <td>0.002783</td>
          <td>0.002189</td>
          <td>0.002468</td>
        </tr>
        <tr>
          <th>WPB-WPK</th>
          <td>0.004834</td>
          <td>0.003946</td>
          <td>0.006018</td>
          <td>0.007941</td>
          <td>0.007695</td>
          <td>0.012035</td>
          <td>0.007991</td>
          <td>0.003377</td>
          <td>0.003157</td>
          <td>0.003977</td>
          <td>...</td>
          <td>0.004363</td>
          <td>0.002669</td>
          <td>0.003141</td>
          <td>0.003167</td>
          <td>0.003042</td>
          <td>0.003491</td>
          <td>0.002751</td>
          <td>0.003109</td>
          <td>0.002250</td>
          <td>0.002634</td>
        </tr>
        <tr>
          <th>WPB-WTH</th>
          <td>0.004288</td>
          <td>0.003984</td>
          <td>0.005571</td>
          <td>0.004220</td>
          <td>0.007427</td>
          <td>0.005267</td>
          <td>0.003976</td>
          <td>0.003341</td>
          <td>0.002745</td>
          <td>0.004276</td>
          <td>...</td>
          <td>0.003916</td>
          <td>0.002933</td>
          <td>0.002686</td>
          <td>0.002838</td>
          <td>0.002487</td>
          <td>0.003377</td>
          <td>0.002985</td>
          <td>0.002528</td>
          <td>0.002227</td>
          <td>0.002165</td>
        </tr>
      </tbody>
    </table>
    <p>1010 rows × 42 columns</p>
    </div>



.. code:: ipython3

    # Creates dataframe with number of station pairs (markets) in each cluster
    num_of_markets = clusters_by_stnpair['cluster_num'].value_counts().reset_index()
    num_of_markets = num_of_markets.sort_values(by=['cluster_num'])
    num_of_markets.columns = ['cluster_num', 'number of markets']

.. code:: ipython3

    num_of_markets




.. raw:: html

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
          <th>cluster_num</th>
          <th>number of markets</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>16</th>
          <td>0</td>
          <td>2</td>
        </tr>
        <tr>
          <th>12</th>
          <td>1</td>
          <td>32</td>
        </tr>
        <tr>
          <th>7</th>
          <td>2</td>
          <td>54</td>
        </tr>
        <tr>
          <th>8</th>
          <td>3</td>
          <td>50</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4</td>
          <td>140</td>
        </tr>
        <tr>
          <th>11</th>
          <td>5</td>
          <td>40</td>
        </tr>
        <tr>
          <th>2</th>
          <td>6</td>
          <td>105</td>
        </tr>
        <tr>
          <th>4</th>
          <td>7</td>
          <td>76</td>
        </tr>
        <tr>
          <th>17</th>
          <td>8</td>
          <td>1</td>
        </tr>
        <tr>
          <th>14</th>
          <td>9</td>
          <td>24</td>
        </tr>
        <tr>
          <th>6</th>
          <td>10</td>
          <td>62</td>
        </tr>
        <tr>
          <th>5</th>
          <td>11</td>
          <td>63</td>
        </tr>
        <tr>
          <th>3</th>
          <td>12</td>
          <td>83</td>
        </tr>
        <tr>
          <th>0</th>
          <td>13</td>
          <td>147</td>
        </tr>
        <tr>
          <th>13</th>
          <td>14</td>
          <td>26</td>
        </tr>
        <tr>
          <th>15</th>
          <td>15</td>
          <td>14</td>
        </tr>
        <tr>
          <th>10</th>
          <td>16</td>
          <td>43</td>
        </tr>
        <tr>
          <th>9</th>
          <td>17</td>
          <td>48</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Joins clusters table with number of markets table
    clusters_by_stnpair = pd.merge(clusters_by_stnpair,num_of_markets, how = "left", on = "cluster_num" )

.. code:: ipython3

    clusters_by_stnpair




.. raw:: html

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
          <th>stnpair</th>
          <th>annual_ridership_at_stnpair</th>
          <th>weights</th>
          <th>cluster_num</th>
          <th>number of markets</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ORB-POR</td>
          <td>10090</td>
          <td>1441.428571</td>
          <td>0</td>
          <td>2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>BON-ORB</td>
          <td>48894</td>
          <td>6984.857143</td>
          <td>0</td>
          <td>2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NHV-STM</td>
          <td>23155</td>
          <td>3307.857143</td>
          <td>1</td>
          <td>32</td>
        </tr>
        <tr>
          <th>3</th>
          <td>EXT-NYP</td>
          <td>258360</td>
          <td>36908.571429</td>
          <td>1</td>
          <td>32</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NHV-NLC</td>
          <td>66909</td>
          <td>9558.428571</td>
          <td>1</td>
          <td>32</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1005</th>
          <td>HAR-LNC</td>
          <td>490659</td>
          <td>70094.142857</td>
          <td>17</td>
          <td>48</td>
        </tr>
        <tr>
          <th>1006</th>
          <td>HAR-MID</td>
          <td>33399</td>
          <td>4771.285714</td>
          <td>17</td>
          <td>48</td>
        </tr>
        <tr>
          <th>1007</th>
          <td>HAR-MJY</td>
          <td>229810</td>
          <td>32830.000000</td>
          <td>17</td>
          <td>48</td>
        </tr>
        <tr>
          <th>1008</th>
          <td>HAR-PAR</td>
          <td>28925</td>
          <td>4132.142857</td>
          <td>17</td>
          <td>48</td>
        </tr>
        <tr>
          <th>1009</th>
          <td>NWK-PAO</td>
          <td>37292</td>
          <td>5327.428571</td>
          <td>17</td>
          <td>48</td>
        </tr>
      </tbody>
    </table>
    <p>1010 rows × 5 columns</p>
    </div>



.. code:: ipython3

    # Creates a df with total ridership per cluster
    annual_totals = clusters_by_stnpair.groupby(['cluster_num','number of markets'])['annual_ridership_at_stnpair'].sum().reset_index()
    annual_totals.columns = ['cluster_num','number of markets','total_ridership_in_cluster']
    annual_totals.to_csv('final_clusters_data.csv', sep='\t', index=False)

.. code:: ipython3

    # Adds total ridership per cluster to larger df
    m_clusters_by_stnpair = pd.merge( clusters_by_stnpair, annual_totals, how = "inner", on = "number of markets", suffixes=('', '_drop'))
    clusters_by_stnpair = m_clusters_by_stnpair.drop_duplicates(subset=['stnpair'])
    clusters_by_stnpair.drop([col for col in clusters_by_stnpair.columns if 'drop' in col], axis=1, inplace=True)
    clusters_by_stnpair




.. raw:: html

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
          <th>stnpair</th>
          <th>annual_ridership_at_stnpair</th>
          <th>weights</th>
          <th>cluster_num</th>
          <th>number of markets</th>
          <th>total_ridership_in_cluster</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ORB-POR</td>
          <td>10090</td>
          <td>1441.428571</td>
          <td>0</td>
          <td>2</td>
          <td>58984</td>
        </tr>
        <tr>
          <th>1</th>
          <td>BON-ORB</td>
          <td>48894</td>
          <td>6984.857143</td>
          <td>0</td>
          <td>2</td>
          <td>58984</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NHV-STM</td>
          <td>23155</td>
          <td>3307.857143</td>
          <td>1</td>
          <td>32</td>
          <td>16411613</td>
        </tr>
        <tr>
          <th>3</th>
          <td>EXT-NYP</td>
          <td>258360</td>
          <td>36908.571429</td>
          <td>1</td>
          <td>32</td>
          <td>16411613</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NHV-NLC</td>
          <td>66909</td>
          <td>9558.428571</td>
          <td>1</td>
          <td>32</td>
          <td>16411613</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1005</th>
          <td>HAR-LNC</td>
          <td>490659</td>
          <td>70094.142857</td>
          <td>17</td>
          <td>48</td>
          <td>5726540</td>
        </tr>
        <tr>
          <th>1006</th>
          <td>HAR-MID</td>
          <td>33399</td>
          <td>4771.285714</td>
          <td>17</td>
          <td>48</td>
          <td>5726540</td>
        </tr>
        <tr>
          <th>1007</th>
          <td>HAR-MJY</td>
          <td>229810</td>
          <td>32830.000000</td>
          <td>17</td>
          <td>48</td>
          <td>5726540</td>
        </tr>
        <tr>
          <th>1008</th>
          <td>HAR-PAR</td>
          <td>28925</td>
          <td>4132.142857</td>
          <td>17</td>
          <td>48</td>
          <td>5726540</td>
        </tr>
        <tr>
          <th>1009</th>
          <td>NWK-PAO</td>
          <td>37292</td>
          <td>5327.428571</td>
          <td>17</td>
          <td>48</td>
          <td>5726540</td>
        </tr>
      </tbody>
    </table>
    <p>1010 rows × 6 columns</p>
    </div>



.. code:: ipython3

    # Checks all station pairs in cluster 15 for example
    clusters_by_stnpair[clusters_by_stnpair['cluster_num'] == 15]




.. raw:: html

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
          <th>stnpair</th>
          <th>annual_ridership_at_stnpair</th>
          <th>weights</th>
          <th>cluster_num</th>
          <th>number of markets</th>
          <th>total_ridership_in_cluster</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>905</th>
          <td>CHI-CIN</td>
          <td>31199</td>
          <td>4457.000000</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>906</th>
          <td>CIN-WAS</td>
          <td>7825</td>
          <td>1117.857143</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>907</th>
          <td>CIN-NYP</td>
          <td>2456</td>
          <td>818.666667</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>908</th>
          <td>CIN-CVS</td>
          <td>4360</td>
          <td>872.000000</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>909</th>
          <td>CHW-WAS</td>
          <td>8997</td>
          <td>1285.285714</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>910</th>
          <td>CHI-CHW</td>
          <td>10819</td>
          <td>1545.571429</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>911</th>
          <td>WAS-WSS</td>
          <td>1531</td>
          <td>765.500000</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>912</th>
          <td>STA-WAS</td>
          <td>7189</td>
          <td>1027.000000</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>913</th>
          <td>CHI-IND</td>
          <td>66797</td>
          <td>9542.428571</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>914</th>
          <td>CHI-PHL</td>
          <td>7611</td>
          <td>1268.500000</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>915</th>
          <td>CHW-NYP</td>
          <td>2293</td>
          <td>764.333333</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>916</th>
          <td>CHI-CRF</td>
          <td>10223</td>
          <td>1460.428571</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>917</th>
          <td>CHI-HUN</td>
          <td>6419</td>
          <td>917.000000</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
        <tr>
          <th>918</th>
          <td>CHI-CVS</td>
          <td>15293</td>
          <td>2184.714286</td>
          <td>15</td>
          <td>14</td>
          <td>183012</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Saves all cluster data in excel file
    clusters_by_stnpair.to_excel('final_cluster_data_2.xlsx', index=False)

.. code:: ipython3

    # Plots centroids of each cluster in 18 cluster analysis
    
    plt.subplots(figsize=(14,8))
    x_coordinates = list(range(0,42))
    for n in range(0,18): 
    
        y2_coordinates = centroids[14][n]
    
        plt.title('Cluster Centroids')
        plt.ylabel('centroids')
        plt.xlabel('time')
    
        plt.plot(x_coordinates, y2_coordinates)
    
    #plt.savefig('Centroids.png')
    y1_coordinates = top5_mtx.loc['ALB-NYP'].values
    y3_coordinates = top5_mtx.loc['BOS-NYP'].values
    y4_coordinates = top5_mtx.loc['NYP-PHL'].values
    y5_coordinates = top5_mtx.loc['NYP-WAS'].values
    y6_coordinates = top5_mtx.loc['PHL-WAS'].values
    
    plt.plot(x_coordinates, y1_coordinates, color='red')
    plt.plot(x_coordinates, y3_coordinates, color='red')
    plt.plot(x_coordinates, y4_coordinates, color='red')
    plt.plot(x_coordinates, y5_coordinates, color='red')
    plt.plot(x_coordinates, y6_coordinates, color='red')
    
    plt.savefig('Centroids.png')
    
    plt.show()



.. image:: official_analysis_files/official_analysis_62_0.png


.. code:: ipython3

    # Creates a df from every point in each centroid in our 18 clusters analysis
    centroid_list = centroids[14] 
    centroid_df = pd.DataFrame(centroid_list)

.. code:: ipython3

    centroid_df




.. raw:: html

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
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>...</th>
          <th>32</th>
          <th>33</th>
          <th>34</th>
          <th>35</th>
          <th>36</th>
          <th>37</th>
          <th>38</th>
          <th>39</th>
          <th>40</th>
          <th>41</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.000113</td>
          <td>0.000000e+00</td>
          <td>-8.673617e-19</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>-8.673617e-19</td>
          <td>-8.673617e-19</td>
          <td>-4.336809e-19</td>
          <td>1.168465e-04</td>
          <td>0.000000</td>
          <td>...</td>
          <td>0.008278</td>
          <td>0.007237</td>
          <td>0.008502</td>
          <td>-4.336809e-19</td>
          <td>1.154274e-04</td>
          <td>-4.336809e-19</td>
          <td>0.000106</td>
          <td>0.000115</td>
          <td>-4.336809e-19</td>
          <td>0.000115</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.003203</td>
          <td>3.258643e-03</td>
          <td>2.590068e-03</td>
          <td>0.003599</td>
          <td>0.002271</td>
          <td>4.318933e-03</td>
          <td>4.654331e-03</td>
          <td>3.096273e-03</td>
          <td>2.771825e-03</td>
          <td>0.002078</td>
          <td>...</td>
          <td>0.003207</td>
          <td>0.002821</td>
          <td>0.003022</td>
          <td>3.081754e-03</td>
          <td>2.486557e-03</td>
          <td>1.788334e-03</td>
          <td>0.002251</td>
          <td>0.002911</td>
          <td>2.564016e-03</td>
          <td>0.002648</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.004603</td>
          <td>4.144771e-03</td>
          <td>4.407791e-03</td>
          <td>0.008132</td>
          <td>0.003614</td>
          <td>8.825090e-03</td>
          <td>8.531157e-03</td>
          <td>3.611194e-03</td>
          <td>2.728320e-03</td>
          <td>0.003157</td>
          <td>...</td>
          <td>0.002673</td>
          <td>0.002114</td>
          <td>0.002241</td>
          <td>4.306331e-03</td>
          <td>2.136596e-03</td>
          <td>2.906892e-03</td>
          <td>0.004053</td>
          <td>0.002313</td>
          <td>1.737786e-03</td>
          <td>0.001632</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.003477</td>
          <td>3.102286e-03</td>
          <td>3.206678e-03</td>
          <td>0.003933</td>
          <td>0.003030</td>
          <td>3.904266e-03</td>
          <td>4.842751e-03</td>
          <td>3.438295e-03</td>
          <td>2.965589e-03</td>
          <td>0.002839</td>
          <td>...</td>
          <td>0.003163</td>
          <td>0.002668</td>
          <td>0.002825</td>
          <td>2.960678e-03</td>
          <td>2.242462e-03</td>
          <td>2.083751e-03</td>
          <td>0.002588</td>
          <td>0.002512</td>
          <td>2.122652e-03</td>
          <td>0.002170</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.003744</td>
          <td>4.859756e-03</td>
          <td>4.832710e-03</td>
          <td>0.004518</td>
          <td>0.002604</td>
          <td>5.400510e-03</td>
          <td>4.732572e-03</td>
          <td>3.757243e-03</td>
          <td>3.418668e-03</td>
          <td>0.003783</td>
          <td>...</td>
          <td>0.004048</td>
          <td>0.003463</td>
          <td>0.003537</td>
          <td>2.474136e-03</td>
          <td>2.209856e-03</td>
          <td>2.278595e-03</td>
          <td>0.002425</td>
          <td>0.002299</td>
          <td>1.870284e-03</td>
          <td>0.001970</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.002707</td>
          <td>3.389401e-03</td>
          <td>1.871834e-03</td>
          <td>0.002772</td>
          <td>0.001812</td>
          <td>3.888080e-03</td>
          <td>3.400638e-03</td>
          <td>2.626907e-03</td>
          <td>2.793637e-03</td>
          <td>0.001540</td>
          <td>...</td>
          <td>0.003513</td>
          <td>0.003261</td>
          <td>0.003430</td>
          <td>2.919488e-03</td>
          <td>2.896202e-03</td>
          <td>1.404894e-03</td>
          <td>0.001677</td>
          <td>0.003442</td>
          <td>3.079461e-03</td>
          <td>0.003180</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.004271</td>
          <td>4.435720e-03</td>
          <td>5.025022e-03</td>
          <td>0.006670</td>
          <td>0.003554</td>
          <td>7.845605e-03</td>
          <td>1.008005e-02</td>
          <td>3.960339e-03</td>
          <td>3.036945e-03</td>
          <td>0.003117</td>
          <td>...</td>
          <td>0.003475</td>
          <td>0.002608</td>
          <td>0.002775</td>
          <td>3.532225e-03</td>
          <td>2.129010e-03</td>
          <td>2.228852e-03</td>
          <td>0.003252</td>
          <td>0.002370</td>
          <td>1.756987e-03</td>
          <td>0.001791</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.003851</td>
          <td>3.699328e-03</td>
          <td>4.168286e-03</td>
          <td>0.004854</td>
          <td>0.002994</td>
          <td>5.448732e-03</td>
          <td>6.593988e-03</td>
          <td>3.980438e-03</td>
          <td>3.154703e-03</td>
          <td>0.003448</td>
          <td>...</td>
          <td>0.003022</td>
          <td>0.002350</td>
          <td>0.002481</td>
          <td>3.278374e-03</td>
          <td>2.024578e-03</td>
          <td>2.391616e-03</td>
          <td>0.003117</td>
          <td>0.002198</td>
          <td>1.653013e-03</td>
          <td>0.001694</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.002354</td>
          <td>1.177163e-03</td>
          <td>7.357269e-03</td>
          <td>0.001766</td>
          <td>0.006474</td>
          <td>8.673617e-19</td>
          <td>1.765745e-03</td>
          <td>4.002354e-03</td>
          <td>1.471454e-03</td>
          <td>0.002943</td>
          <td>...</td>
          <td>0.005008</td>
          <td>0.002943</td>
          <td>0.005271</td>
          <td>2.354326e-03</td>
          <td>1.373357e-03</td>
          <td>2.942908e-03</td>
          <td>0.002943</td>
          <td>0.001668</td>
          <td>1.961938e-03</td>
          <td>0.001491</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.003868</td>
          <td>2.221431e-03</td>
          <td>4.322423e-03</td>
          <td>0.004708</td>
          <td>0.001923</td>
          <td>3.445849e-03</td>
          <td>4.697158e-03</td>
          <td>3.458959e-03</td>
          <td>2.385378e-03</td>
          <td>0.004449</td>
          <td>...</td>
          <td>0.003075</td>
          <td>0.002622</td>
          <td>0.002798</td>
          <td>3.301657e-03</td>
          <td>1.856778e-03</td>
          <td>3.852410e-03</td>
          <td>0.003431</td>
          <td>0.002144</td>
          <td>1.935102e-03</td>
          <td>0.001967</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.003751</td>
          <td>4.027594e-03</td>
          <td>3.811111e-03</td>
          <td>0.005776</td>
          <td>0.002533</td>
          <td>6.700149e-03</td>
          <td>6.753052e-03</td>
          <td>2.954127e-03</td>
          <td>2.605358e-03</td>
          <td>0.002080</td>
          <td>...</td>
          <td>0.003253</td>
          <td>0.002655</td>
          <td>0.002811</td>
          <td>3.212510e-03</td>
          <td>2.416699e-03</td>
          <td>1.757587e-03</td>
          <td>0.002634</td>
          <td>0.002759</td>
          <td>2.187340e-03</td>
          <td>0.002238</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.005351</td>
          <td>6.755972e-03</td>
          <td>6.900149e-03</td>
          <td>0.011177</td>
          <td>0.003617</td>
          <td>1.465794e-02</td>
          <td>1.286083e-02</td>
          <td>4.043154e-03</td>
          <td>3.186455e-03</td>
          <td>0.003346</td>
          <td>...</td>
          <td>0.003261</td>
          <td>0.002595</td>
          <td>0.002620</td>
          <td>3.920848e-03</td>
          <td>2.317094e-03</td>
          <td>2.733897e-03</td>
          <td>0.003766</td>
          <td>0.002597</td>
          <td>1.861708e-03</td>
          <td>0.001816</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.004696</td>
          <td>3.777616e-03</td>
          <td>4.867877e-03</td>
          <td>0.004757</td>
          <td>0.006611</td>
          <td>4.956457e-03</td>
          <td>4.988593e-03</td>
          <td>3.886430e-03</td>
          <td>3.290364e-03</td>
          <td>0.004243</td>
          <td>...</td>
          <td>0.003571</td>
          <td>0.003053</td>
          <td>0.003173</td>
          <td>3.416586e-03</td>
          <td>2.655298e-03</td>
          <td>3.506984e-03</td>
          <td>0.003352</td>
          <td>0.002803</td>
          <td>2.331214e-03</td>
          <td>0.002436</td>
        </tr>
        <tr>
          <th>13</th>
          <td>0.005584</td>
          <td>6.132790e-03</td>
          <td>6.949845e-03</td>
          <td>0.006113</td>
          <td>0.003225</td>
          <td>8.108133e-03</td>
          <td>7.162819e-03</td>
          <td>4.677292e-03</td>
          <td>3.787787e-03</td>
          <td>0.004263</td>
          <td>...</td>
          <td>0.004308</td>
          <td>0.003347</td>
          <td>0.003593</td>
          <td>3.038699e-03</td>
          <td>2.455895e-03</td>
          <td>2.746325e-03</td>
          <td>0.002917</td>
          <td>0.002756</td>
          <td>2.160478e-03</td>
          <td>0.002327</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.002904</td>
          <td>2.439805e-03</td>
          <td>2.850012e-03</td>
          <td>0.002966</td>
          <td>0.002524</td>
          <td>3.414085e-03</td>
          <td>4.333235e-03</td>
          <td>2.706258e-03</td>
          <td>2.365576e-03</td>
          <td>0.002380</td>
          <td>...</td>
          <td>0.004286</td>
          <td>0.003308</td>
          <td>0.003715</td>
          <td>2.193038e-03</td>
          <td>1.669442e-03</td>
          <td>1.748874e-03</td>
          <td>0.002037</td>
          <td>0.001887</td>
          <td>1.611750e-03</td>
          <td>0.001687</td>
        </tr>
        <tr>
          <th>15</th>
          <td>0.004993</td>
          <td>-8.673617e-19</td>
          <td>4.650353e-03</td>
          <td>0.004916</td>
          <td>0.001841</td>
          <td>6.776644e-03</td>
          <td>4.335691e-03</td>
          <td>5.033730e-03</td>
          <td>-1.734723e-18</td>
          <td>0.004271</td>
          <td>...</td>
          <td>0.004386</td>
          <td>0.004356</td>
          <td>0.005007</td>
          <td>3.214592e-03</td>
          <td>-4.336809e-19</td>
          <td>2.662572e-03</td>
          <td>0.002455</td>
          <td>0.002386</td>
          <td>2.168944e-03</td>
          <td>0.002538</td>
        </tr>
        <tr>
          <th>16</th>
          <td>0.002624</td>
          <td>2.752082e-03</td>
          <td>2.637459e-03</td>
          <td>0.002749</td>
          <td>0.002693</td>
          <td>2.493644e-03</td>
          <td>2.095285e-03</td>
          <td>2.781816e-03</td>
          <td>2.588301e-03</td>
          <td>0.002834</td>
          <td>...</td>
          <td>0.003447</td>
          <td>0.003385</td>
          <td>0.003378</td>
          <td>2.322611e-03</td>
          <td>2.273413e-03</td>
          <td>2.406070e-03</td>
          <td>0.002173</td>
          <td>0.002386</td>
          <td>2.222128e-03</td>
          <td>0.002300</td>
        </tr>
        <tr>
          <th>17</th>
          <td>0.001887</td>
          <td>3.803183e-03</td>
          <td>1.027182e-03</td>
          <td>0.001076</td>
          <td>0.001054</td>
          <td>3.921148e-03</td>
          <td>2.601776e-03</td>
          <td>2.865013e-03</td>
          <td>3.241096e-03</td>
          <td>0.000972</td>
          <td>...</td>
          <td>0.003991</td>
          <td>0.003982</td>
          <td>0.003991</td>
          <td>3.338288e-03</td>
          <td>3.593798e-03</td>
          <td>9.045875e-04</td>
          <td>0.000802</td>
          <td>0.003899</td>
          <td>3.766723e-03</td>
          <td>0.003843</td>
        </tr>
      </tbody>
    </table>
    <p>18 rows × 42 columns</p>
    </div>



.. code:: ipython3

    # Creates label for each day in our centroids df
    centroid_df.columns = list(mtx.columns)
    centroid_df




.. raw:: html

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
          <th>Thanksgiving Friday</th>
          <th>Thanksgiving Monday</th>
          <th>Thanksgiving Saturday</th>
          <th>Thanksgiving Sunday</th>
          <th>Thanksgiving Thursday</th>
          <th>Thanksgiving Tuesday</th>
          <th>Thanksgiving Wednesday</th>
          <th>december Friday</th>
          <th>december Monday</th>
          <th>december Saturday</th>
          <th>...</th>
          <th>summer Thursday</th>
          <th>summer Tuesday</th>
          <th>summer Wednesday</th>
          <th>winter Friday</th>
          <th>winter Monday</th>
          <th>winter Saturday</th>
          <th>winter Sunday</th>
          <th>winter Thursday</th>
          <th>winter Tuesday</th>
          <th>winter Wednesday</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.000113</td>
          <td>0.000000e+00</td>
          <td>-8.673617e-19</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>-8.673617e-19</td>
          <td>-8.673617e-19</td>
          <td>-4.336809e-19</td>
          <td>1.168465e-04</td>
          <td>0.000000</td>
          <td>...</td>
          <td>0.008278</td>
          <td>0.007237</td>
          <td>0.008502</td>
          <td>-4.336809e-19</td>
          <td>1.154274e-04</td>
          <td>-4.336809e-19</td>
          <td>0.000106</td>
          <td>0.000115</td>
          <td>-4.336809e-19</td>
          <td>0.000115</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.003203</td>
          <td>3.258643e-03</td>
          <td>2.590068e-03</td>
          <td>0.003599</td>
          <td>0.002271</td>
          <td>4.318933e-03</td>
          <td>4.654331e-03</td>
          <td>3.096273e-03</td>
          <td>2.771825e-03</td>
          <td>0.002078</td>
          <td>...</td>
          <td>0.003207</td>
          <td>0.002821</td>
          <td>0.003022</td>
          <td>3.081754e-03</td>
          <td>2.486557e-03</td>
          <td>1.788334e-03</td>
          <td>0.002251</td>
          <td>0.002911</td>
          <td>2.564016e-03</td>
          <td>0.002648</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.004603</td>
          <td>4.144771e-03</td>
          <td>4.407791e-03</td>
          <td>0.008132</td>
          <td>0.003614</td>
          <td>8.825090e-03</td>
          <td>8.531157e-03</td>
          <td>3.611194e-03</td>
          <td>2.728320e-03</td>
          <td>0.003157</td>
          <td>...</td>
          <td>0.002673</td>
          <td>0.002114</td>
          <td>0.002241</td>
          <td>4.306331e-03</td>
          <td>2.136596e-03</td>
          <td>2.906892e-03</td>
          <td>0.004053</td>
          <td>0.002313</td>
          <td>1.737786e-03</td>
          <td>0.001632</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.003477</td>
          <td>3.102286e-03</td>
          <td>3.206678e-03</td>
          <td>0.003933</td>
          <td>0.003030</td>
          <td>3.904266e-03</td>
          <td>4.842751e-03</td>
          <td>3.438295e-03</td>
          <td>2.965589e-03</td>
          <td>0.002839</td>
          <td>...</td>
          <td>0.003163</td>
          <td>0.002668</td>
          <td>0.002825</td>
          <td>2.960678e-03</td>
          <td>2.242462e-03</td>
          <td>2.083751e-03</td>
          <td>0.002588</td>
          <td>0.002512</td>
          <td>2.122652e-03</td>
          <td>0.002170</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.003744</td>
          <td>4.859756e-03</td>
          <td>4.832710e-03</td>
          <td>0.004518</td>
          <td>0.002604</td>
          <td>5.400510e-03</td>
          <td>4.732572e-03</td>
          <td>3.757243e-03</td>
          <td>3.418668e-03</td>
          <td>0.003783</td>
          <td>...</td>
          <td>0.004048</td>
          <td>0.003463</td>
          <td>0.003537</td>
          <td>2.474136e-03</td>
          <td>2.209856e-03</td>
          <td>2.278595e-03</td>
          <td>0.002425</td>
          <td>0.002299</td>
          <td>1.870284e-03</td>
          <td>0.001970</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.002707</td>
          <td>3.389401e-03</td>
          <td>1.871834e-03</td>
          <td>0.002772</td>
          <td>0.001812</td>
          <td>3.888080e-03</td>
          <td>3.400638e-03</td>
          <td>2.626907e-03</td>
          <td>2.793637e-03</td>
          <td>0.001540</td>
          <td>...</td>
          <td>0.003513</td>
          <td>0.003261</td>
          <td>0.003430</td>
          <td>2.919488e-03</td>
          <td>2.896202e-03</td>
          <td>1.404894e-03</td>
          <td>0.001677</td>
          <td>0.003442</td>
          <td>3.079461e-03</td>
          <td>0.003180</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.004271</td>
          <td>4.435720e-03</td>
          <td>5.025022e-03</td>
          <td>0.006670</td>
          <td>0.003554</td>
          <td>7.845605e-03</td>
          <td>1.008005e-02</td>
          <td>3.960339e-03</td>
          <td>3.036945e-03</td>
          <td>0.003117</td>
          <td>...</td>
          <td>0.003475</td>
          <td>0.002608</td>
          <td>0.002775</td>
          <td>3.532225e-03</td>
          <td>2.129010e-03</td>
          <td>2.228852e-03</td>
          <td>0.003252</td>
          <td>0.002370</td>
          <td>1.756987e-03</td>
          <td>0.001791</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.003851</td>
          <td>3.699328e-03</td>
          <td>4.168286e-03</td>
          <td>0.004854</td>
          <td>0.002994</td>
          <td>5.448732e-03</td>
          <td>6.593988e-03</td>
          <td>3.980438e-03</td>
          <td>3.154703e-03</td>
          <td>0.003448</td>
          <td>...</td>
          <td>0.003022</td>
          <td>0.002350</td>
          <td>0.002481</td>
          <td>3.278374e-03</td>
          <td>2.024578e-03</td>
          <td>2.391616e-03</td>
          <td>0.003117</td>
          <td>0.002198</td>
          <td>1.653013e-03</td>
          <td>0.001694</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.002354</td>
          <td>1.177163e-03</td>
          <td>7.357269e-03</td>
          <td>0.001766</td>
          <td>0.006474</td>
          <td>8.673617e-19</td>
          <td>1.765745e-03</td>
          <td>4.002354e-03</td>
          <td>1.471454e-03</td>
          <td>0.002943</td>
          <td>...</td>
          <td>0.005008</td>
          <td>0.002943</td>
          <td>0.005271</td>
          <td>2.354326e-03</td>
          <td>1.373357e-03</td>
          <td>2.942908e-03</td>
          <td>0.002943</td>
          <td>0.001668</td>
          <td>1.961938e-03</td>
          <td>0.001491</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.003868</td>
          <td>2.221431e-03</td>
          <td>4.322423e-03</td>
          <td>0.004708</td>
          <td>0.001923</td>
          <td>3.445849e-03</td>
          <td>4.697158e-03</td>
          <td>3.458959e-03</td>
          <td>2.385378e-03</td>
          <td>0.004449</td>
          <td>...</td>
          <td>0.003075</td>
          <td>0.002622</td>
          <td>0.002798</td>
          <td>3.301657e-03</td>
          <td>1.856778e-03</td>
          <td>3.852410e-03</td>
          <td>0.003431</td>
          <td>0.002144</td>
          <td>1.935102e-03</td>
          <td>0.001967</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.003751</td>
          <td>4.027594e-03</td>
          <td>3.811111e-03</td>
          <td>0.005776</td>
          <td>0.002533</td>
          <td>6.700149e-03</td>
          <td>6.753052e-03</td>
          <td>2.954127e-03</td>
          <td>2.605358e-03</td>
          <td>0.002080</td>
          <td>...</td>
          <td>0.003253</td>
          <td>0.002655</td>
          <td>0.002811</td>
          <td>3.212510e-03</td>
          <td>2.416699e-03</td>
          <td>1.757587e-03</td>
          <td>0.002634</td>
          <td>0.002759</td>
          <td>2.187340e-03</td>
          <td>0.002238</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.005351</td>
          <td>6.755972e-03</td>
          <td>6.900149e-03</td>
          <td>0.011177</td>
          <td>0.003617</td>
          <td>1.465794e-02</td>
          <td>1.286083e-02</td>
          <td>4.043154e-03</td>
          <td>3.186455e-03</td>
          <td>0.003346</td>
          <td>...</td>
          <td>0.003261</td>
          <td>0.002595</td>
          <td>0.002620</td>
          <td>3.920848e-03</td>
          <td>2.317094e-03</td>
          <td>2.733897e-03</td>
          <td>0.003766</td>
          <td>0.002597</td>
          <td>1.861708e-03</td>
          <td>0.001816</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.004696</td>
          <td>3.777616e-03</td>
          <td>4.867877e-03</td>
          <td>0.004757</td>
          <td>0.006611</td>
          <td>4.956457e-03</td>
          <td>4.988593e-03</td>
          <td>3.886430e-03</td>
          <td>3.290364e-03</td>
          <td>0.004243</td>
          <td>...</td>
          <td>0.003571</td>
          <td>0.003053</td>
          <td>0.003173</td>
          <td>3.416586e-03</td>
          <td>2.655298e-03</td>
          <td>3.506984e-03</td>
          <td>0.003352</td>
          <td>0.002803</td>
          <td>2.331214e-03</td>
          <td>0.002436</td>
        </tr>
        <tr>
          <th>13</th>
          <td>0.005584</td>
          <td>6.132790e-03</td>
          <td>6.949845e-03</td>
          <td>0.006113</td>
          <td>0.003225</td>
          <td>8.108133e-03</td>
          <td>7.162819e-03</td>
          <td>4.677292e-03</td>
          <td>3.787787e-03</td>
          <td>0.004263</td>
          <td>...</td>
          <td>0.004308</td>
          <td>0.003347</td>
          <td>0.003593</td>
          <td>3.038699e-03</td>
          <td>2.455895e-03</td>
          <td>2.746325e-03</td>
          <td>0.002917</td>
          <td>0.002756</td>
          <td>2.160478e-03</td>
          <td>0.002327</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.002904</td>
          <td>2.439805e-03</td>
          <td>2.850012e-03</td>
          <td>0.002966</td>
          <td>0.002524</td>
          <td>3.414085e-03</td>
          <td>4.333235e-03</td>
          <td>2.706258e-03</td>
          <td>2.365576e-03</td>
          <td>0.002380</td>
          <td>...</td>
          <td>0.004286</td>
          <td>0.003308</td>
          <td>0.003715</td>
          <td>2.193038e-03</td>
          <td>1.669442e-03</td>
          <td>1.748874e-03</td>
          <td>0.002037</td>
          <td>0.001887</td>
          <td>1.611750e-03</td>
          <td>0.001687</td>
        </tr>
        <tr>
          <th>15</th>
          <td>0.004993</td>
          <td>-8.673617e-19</td>
          <td>4.650353e-03</td>
          <td>0.004916</td>
          <td>0.001841</td>
          <td>6.776644e-03</td>
          <td>4.335691e-03</td>
          <td>5.033730e-03</td>
          <td>-1.734723e-18</td>
          <td>0.004271</td>
          <td>...</td>
          <td>0.004386</td>
          <td>0.004356</td>
          <td>0.005007</td>
          <td>3.214592e-03</td>
          <td>-4.336809e-19</td>
          <td>2.662572e-03</td>
          <td>0.002455</td>
          <td>0.002386</td>
          <td>2.168944e-03</td>
          <td>0.002538</td>
        </tr>
        <tr>
          <th>16</th>
          <td>0.002624</td>
          <td>2.752082e-03</td>
          <td>2.637459e-03</td>
          <td>0.002749</td>
          <td>0.002693</td>
          <td>2.493644e-03</td>
          <td>2.095285e-03</td>
          <td>2.781816e-03</td>
          <td>2.588301e-03</td>
          <td>0.002834</td>
          <td>...</td>
          <td>0.003447</td>
          <td>0.003385</td>
          <td>0.003378</td>
          <td>2.322611e-03</td>
          <td>2.273413e-03</td>
          <td>2.406070e-03</td>
          <td>0.002173</td>
          <td>0.002386</td>
          <td>2.222128e-03</td>
          <td>0.002300</td>
        </tr>
        <tr>
          <th>17</th>
          <td>0.001887</td>
          <td>3.803183e-03</td>
          <td>1.027182e-03</td>
          <td>0.001076</td>
          <td>0.001054</td>
          <td>3.921148e-03</td>
          <td>2.601776e-03</td>
          <td>2.865013e-03</td>
          <td>3.241096e-03</td>
          <td>0.000972</td>
          <td>...</td>
          <td>0.003991</td>
          <td>0.003982</td>
          <td>0.003991</td>
          <td>3.338288e-03</td>
          <td>3.593798e-03</td>
          <td>9.045875e-04</td>
          <td>0.000802</td>
          <td>0.003899</td>
          <td>3.766723e-03</td>
          <td>0.003843</td>
        </tr>
      </tbody>
    </table>
    <p>18 rows × 42 columns</p>
    </div>



.. code:: ipython3

    # Saves centroids df in an excel file
    centroid_df.to_excel('centroids.xlsx')

Cluster Visualizations
----------------------

After creating and exporting uniform dataframes with the relevant data
for our cluster analysis, I created visualizations in the form of
time-series line graphs to show the patterns within each cluster and how
they are similar. I also show the centroid of every cluster in black for
reference.

.. code:: ipython3

    def create_cluster_plot(cluster, cluster_num, centroid):
        """
            Creates a plot for a single cluster.
         
            Creates a line plot displaying the ridership on each representative day 
            for every station pair in a single cluster.
         
            Parameters
            ----------
            cluster: DataFrame
                A DataFrame that contains cluster data in the form of a matrix
                for a single cluster.
            cluster_num: int
                The cluster number.
            centroids: DataFrame
                A DataFrame that contains the centroid for the cluster.
            
            Generates a plot of the every station pair and the centroid in a cluster.
            
        """
        plt.subplots(figsize=(14,8))
        for row in cluster.index: 
    
            x_coordinates = list(range(0,42))
    
            y1_coordinates = cluster.loc[row].values
    
            plt.title(f'Cluster {cluster_num}')
            plt.ylabel('stnpairs')
            plt.xlabel('time')
    
            plt.plot(x_coordinates, y1_coordinates)
    
        x2_coordinates = list(range(0,42))   
        y2_coordinates = centroid
        plt.plot(x2_coordinates, y2_coordinates, color='black')
        plt.show()

Cluster 1
~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_1, 1, centroids[14][0])



.. image:: official_analysis_files/official_analysis_70_0.png


Cluster 2
~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_2, 2, centroids[14][1])



.. image:: official_analysis_files/official_analysis_72_0.png


Cluster 3
~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_3, 3, centroids[14][2])



.. image:: official_analysis_files/official_analysis_74_0.png


Cluster 4
~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_4, 4, centroids[14][3])



.. image:: official_analysis_files/official_analysis_76_0.png


Cluster 5
~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_5, 5, centroids[14][4])



.. image:: official_analysis_files/official_analysis_78_0.png


Cluster 6
~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_6, 6, centroids[14][5])



.. image:: official_analysis_files/official_analysis_80_0.png


Cluster 7
~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_7, 7, centroids[14][6])



.. image:: official_analysis_files/official_analysis_82_0.png


Cluster 8
~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_8, 8, centroids[14][7])



.. image:: official_analysis_files/official_analysis_84_0.png


Cluster 9
~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_9, 9, centroids[14][8])



.. image:: official_analysis_files/official_analysis_86_0.png


Cluster 10
~~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_10, 10, centroids[14][9])



.. image:: official_analysis_files/official_analysis_88_0.png


Cluster 11
~~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_11, 11, centroids[14][10])



.. image:: official_analysis_files/official_analysis_90_0.png


Cluster 12
~~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_12, 12, centroids[14][11])



.. image:: official_analysis_files/official_analysis_92_0.png


Cluster 13
~~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_13, 13, centroids[14][12])



.. image:: official_analysis_files/official_analysis_94_0.png


Cluster 14
~~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_14, 14, centroids[14][13])



.. image:: official_analysis_files/official_analysis_96_0.png


Cluster 15
~~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_15, 15, centroids[14][14])



.. image:: official_analysis_files/official_analysis_98_0.png


Cluster 16
~~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_16, 16, centroids[14][15])



.. image:: official_analysis_files/official_analysis_100_0.png


Cluster 17
~~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_17, 17, centroids[14][16])



.. image:: official_analysis_files/official_analysis_102_0.png


Cluster 18
~~~~~~~~~~

.. code:: ipython3

    create_cluster_plot(cluster_18, 18, centroids[14][17])



.. image:: official_analysis_files/official_analysis_104_0.png


Top 5 Station Pairs
-------------------

I removed the top 5 station pairs from the analysis in the beginning as
I wanted each of these to represent their own individual clusters. In
this section of the notebook, I am parsing out all of the relevant data
from the top 5 station pairs and exporting it.

.. code:: ipython3

    top_5




.. raw:: html

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
          <th>stnpair</th>
          <th>season</th>
          <th>dow</th>
          <th>avg_daily_ridership</th>
          <th>annual_ridership_at_stnpair</th>
          <th>avg_cluster_input</th>
          <th>season_dow</th>
          <th>annual_ridership_at_stnpair_str</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>797</th>
          <td>ALB-NYP</td>
          <td>Thanksgiving</td>
          <td>Friday</td>
          <td>1967.357143</td>
          <td>639312.428571</td>
          <td>0.003077</td>
          <td>Thanksgiving Friday</td>
          <td>639312.4285714285</td>
        </tr>
        <tr>
          <th>798</th>
          <td>ALB-NYP</td>
          <td>Thanksgiving</td>
          <td>Monday</td>
          <td>1896.000000</td>
          <td>639312.428571</td>
          <td>0.002966</td>
          <td>Thanksgiving Monday</td>
          <td>639312.4285714285</td>
        </tr>
        <tr>
          <th>799</th>
          <td>ALB-NYP</td>
          <td>Thanksgiving</td>
          <td>Saturday</td>
          <td>1906.428571</td>
          <td>639312.428571</td>
          <td>0.002982</td>
          <td>Thanksgiving Saturday</td>
          <td>639312.4285714285</td>
        </tr>
        <tr>
          <th>800</th>
          <td>ALB-NYP</td>
          <td>Thanksgiving</td>
          <td>Sunday</td>
          <td>2238.357143</td>
          <td>639312.428571</td>
          <td>0.003501</td>
          <td>Thanksgiving Sunday</td>
          <td>639312.4285714285</td>
        </tr>
        <tr>
          <th>801</th>
          <td>ALB-NYP</td>
          <td>Thanksgiving</td>
          <td>Thursday</td>
          <td>1249.142857</td>
          <td>639312.428571</td>
          <td>0.001954</td>
          <td>Thanksgiving Thursday</td>
          <td>639312.4285714285</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>37865</th>
          <td>PHL-WAS</td>
          <td>winter</td>
          <td>Saturday</td>
          <td>1119.428571</td>
          <td>701509.428571</td>
          <td>0.001596</td>
          <td>winter Saturday</td>
          <td>701509.4285714285</td>
        </tr>
        <tr>
          <th>37866</th>
          <td>PHL-WAS</td>
          <td>winter</td>
          <td>Sunday</td>
          <td>1501.504000</td>
          <td>701509.428571</td>
          <td>0.002140</td>
          <td>winter Sunday</td>
          <td>701509.4285714285</td>
        </tr>
        <tr>
          <th>37867</th>
          <td>PHL-WAS</td>
          <td>winter</td>
          <td>Thursday</td>
          <td>2152.554429</td>
          <td>701509.428571</td>
          <td>0.003068</td>
          <td>winter Thursday</td>
          <td>701509.4285714285</td>
        </tr>
        <tr>
          <th>37868</th>
          <td>PHL-WAS</td>
          <td>winter</td>
          <td>Tuesday</td>
          <td>1799.303571</td>
          <td>701509.428571</td>
          <td>0.002565</td>
          <td>winter Tuesday</td>
          <td>701509.4285714285</td>
        </tr>
        <tr>
          <th>37869</th>
          <td>PHL-WAS</td>
          <td>winter</td>
          <td>Wednesday</td>
          <td>1900.311429</td>
          <td>701509.428571</td>
          <td>0.002709</td>
          <td>winter Wednesday</td>
          <td>701509.4285714285</td>
        </tr>
      </tbody>
    </table>
    <p>210 rows × 8 columns</p>
    </div>



.. code:: ipython3

    # Creates a mtx for our top 5 city pairs
    top5_mtx = pd.pivot_table(top_5,values = "avg_cluster_input", columns = "season_dow", index = "stnpair", aggfunc = np.sum )
    top5_mtx = top5_mtx.fillna(0)
    top5_mtx




.. raw:: html

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
          <th>season_dow</th>
          <th>Thanksgiving Friday</th>
          <th>Thanksgiving Monday</th>
          <th>Thanksgiving Saturday</th>
          <th>Thanksgiving Sunday</th>
          <th>Thanksgiving Thursday</th>
          <th>Thanksgiving Tuesday</th>
          <th>Thanksgiving Wednesday</th>
          <th>december Friday</th>
          <th>december Monday</th>
          <th>december Saturday</th>
          <th>...</th>
          <th>summer Thursday</th>
          <th>summer Tuesday</th>
          <th>summer Wednesday</th>
          <th>winter Friday</th>
          <th>winter Monday</th>
          <th>winter Saturday</th>
          <th>winter Sunday</th>
          <th>winter Thursday</th>
          <th>winter Tuesday</th>
          <th>winter Wednesday</th>
        </tr>
        <tr>
          <th>stnpair</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>ALB-NYP</th>
          <td>0.003077</td>
          <td>0.002966</td>
          <td>0.002982</td>
          <td>0.003501</td>
          <td>0.001954</td>
          <td>0.004155</td>
          <td>0.004490</td>
          <td>0.003319</td>
          <td>0.002917</td>
          <td>0.003032</td>
          <td>...</td>
          <td>0.003129</td>
          <td>0.002638</td>
          <td>0.002924</td>
          <td>0.003069</td>
          <td>0.002416</td>
          <td>0.002079</td>
          <td>0.002549</td>
          <td>0.002820</td>
          <td>0.002556</td>
          <td>0.002609</td>
        </tr>
        <tr>
          <th>BOS-NYP</th>
          <td>0.002992</td>
          <td>0.002886</td>
          <td>0.002657</td>
          <td>0.003782</td>
          <td>0.002153</td>
          <td>0.003735</td>
          <td>0.004882</td>
          <td>0.003117</td>
          <td>0.002535</td>
          <td>0.002093</td>
          <td>...</td>
          <td>0.003180</td>
          <td>0.002686</td>
          <td>0.002940</td>
          <td>0.003123</td>
          <td>0.002213</td>
          <td>0.001688</td>
          <td>0.002580</td>
          <td>0.002621</td>
          <td>0.002119</td>
          <td>0.002233</td>
        </tr>
        <tr>
          <th>NYP-PHL</th>
          <td>0.002915</td>
          <td>0.002974</td>
          <td>0.002035</td>
          <td>0.002607</td>
          <td>0.001963</td>
          <td>0.003294</td>
          <td>0.003154</td>
          <td>0.002823</td>
          <td>0.002540</td>
          <td>0.001912</td>
          <td>...</td>
          <td>0.003308</td>
          <td>0.002975</td>
          <td>0.003298</td>
          <td>0.003069</td>
          <td>0.002582</td>
          <td>0.001803</td>
          <td>0.001892</td>
          <td>0.003264</td>
          <td>0.002812</td>
          <td>0.002992</td>
        </tr>
        <tr>
          <th>NYP-WAS</th>
          <td>0.002767</td>
          <td>0.002632</td>
          <td>0.002263</td>
          <td>0.003417</td>
          <td>0.002033</td>
          <td>0.002986</td>
          <td>0.003945</td>
          <td>0.003189</td>
          <td>0.002709</td>
          <td>0.002068</td>
          <td>...</td>
          <td>0.003125</td>
          <td>0.002708</td>
          <td>0.002922</td>
          <td>0.003055</td>
          <td>0.002273</td>
          <td>0.001641</td>
          <td>0.002423</td>
          <td>0.002688</td>
          <td>0.002277</td>
          <td>0.002441</td>
        </tr>
        <tr>
          <th>PHL-WAS</th>
          <td>0.002773</td>
          <td>0.002906</td>
          <td>0.002047</td>
          <td>0.003304</td>
          <td>0.001748</td>
          <td>0.003660</td>
          <td>0.003536</td>
          <td>0.002554</td>
          <td>0.002373</td>
          <td>0.001613</td>
          <td>...</td>
          <td>0.003349</td>
          <td>0.003006</td>
          <td>0.003210</td>
          <td>0.002976</td>
          <td>0.002459</td>
          <td>0.001596</td>
          <td>0.002140</td>
          <td>0.003068</td>
          <td>0.002565</td>
          <td>0.002709</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 42 columns</p>
    </div>



.. code:: ipython3

    # Creates df for every centroid including our top 5 city pairs
    total_stepdown_factors = pd.concat([centroid_df,top5_mtx])
    total_stepdown_factors




.. raw:: html

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
          <th>Thanksgiving Friday</th>
          <th>Thanksgiving Monday</th>
          <th>Thanksgiving Saturday</th>
          <th>Thanksgiving Sunday</th>
          <th>Thanksgiving Thursday</th>
          <th>Thanksgiving Tuesday</th>
          <th>Thanksgiving Wednesday</th>
          <th>december Friday</th>
          <th>december Monday</th>
          <th>december Saturday</th>
          <th>...</th>
          <th>summer Thursday</th>
          <th>summer Tuesday</th>
          <th>summer Wednesday</th>
          <th>winter Friday</th>
          <th>winter Monday</th>
          <th>winter Saturday</th>
          <th>winter Sunday</th>
          <th>winter Thursday</th>
          <th>winter Tuesday</th>
          <th>winter Wednesday</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.000113</td>
          <td>0.000000e+00</td>
          <td>-8.673617e-19</td>
          <td>0.000000</td>
          <td>0.000000</td>
          <td>-8.673617e-19</td>
          <td>-8.673617e-19</td>
          <td>-4.336809e-19</td>
          <td>1.168465e-04</td>
          <td>0.000000</td>
          <td>...</td>
          <td>0.008278</td>
          <td>0.007237</td>
          <td>0.008502</td>
          <td>-4.336809e-19</td>
          <td>1.154274e-04</td>
          <td>-4.336809e-19</td>
          <td>0.000106</td>
          <td>0.000115</td>
          <td>-4.336809e-19</td>
          <td>0.000115</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.003203</td>
          <td>3.258643e-03</td>
          <td>2.590068e-03</td>
          <td>0.003599</td>
          <td>0.002271</td>
          <td>4.318933e-03</td>
          <td>4.654331e-03</td>
          <td>3.096273e-03</td>
          <td>2.771825e-03</td>
          <td>0.002078</td>
          <td>...</td>
          <td>0.003207</td>
          <td>0.002821</td>
          <td>0.003022</td>
          <td>3.081754e-03</td>
          <td>2.486557e-03</td>
          <td>1.788334e-03</td>
          <td>0.002251</td>
          <td>0.002911</td>
          <td>2.564016e-03</td>
          <td>0.002648</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.004603</td>
          <td>4.144771e-03</td>
          <td>4.407791e-03</td>
          <td>0.008132</td>
          <td>0.003614</td>
          <td>8.825090e-03</td>
          <td>8.531157e-03</td>
          <td>3.611194e-03</td>
          <td>2.728320e-03</td>
          <td>0.003157</td>
          <td>...</td>
          <td>0.002673</td>
          <td>0.002114</td>
          <td>0.002241</td>
          <td>4.306331e-03</td>
          <td>2.136596e-03</td>
          <td>2.906892e-03</td>
          <td>0.004053</td>
          <td>0.002313</td>
          <td>1.737786e-03</td>
          <td>0.001632</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.003477</td>
          <td>3.102286e-03</td>
          <td>3.206678e-03</td>
          <td>0.003933</td>
          <td>0.003030</td>
          <td>3.904266e-03</td>
          <td>4.842751e-03</td>
          <td>3.438295e-03</td>
          <td>2.965589e-03</td>
          <td>0.002839</td>
          <td>...</td>
          <td>0.003163</td>
          <td>0.002668</td>
          <td>0.002825</td>
          <td>2.960678e-03</td>
          <td>2.242462e-03</td>
          <td>2.083751e-03</td>
          <td>0.002588</td>
          <td>0.002512</td>
          <td>2.122652e-03</td>
          <td>0.002170</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.003744</td>
          <td>4.859756e-03</td>
          <td>4.832710e-03</td>
          <td>0.004518</td>
          <td>0.002604</td>
          <td>5.400510e-03</td>
          <td>4.732572e-03</td>
          <td>3.757243e-03</td>
          <td>3.418668e-03</td>
          <td>0.003783</td>
          <td>...</td>
          <td>0.004048</td>
          <td>0.003463</td>
          <td>0.003537</td>
          <td>2.474136e-03</td>
          <td>2.209856e-03</td>
          <td>2.278595e-03</td>
          <td>0.002425</td>
          <td>0.002299</td>
          <td>1.870284e-03</td>
          <td>0.001970</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.002707</td>
          <td>3.389401e-03</td>
          <td>1.871834e-03</td>
          <td>0.002772</td>
          <td>0.001812</td>
          <td>3.888080e-03</td>
          <td>3.400638e-03</td>
          <td>2.626907e-03</td>
          <td>2.793637e-03</td>
          <td>0.001540</td>
          <td>...</td>
          <td>0.003513</td>
          <td>0.003261</td>
          <td>0.003430</td>
          <td>2.919488e-03</td>
          <td>2.896202e-03</td>
          <td>1.404894e-03</td>
          <td>0.001677</td>
          <td>0.003442</td>
          <td>3.079461e-03</td>
          <td>0.003180</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.004271</td>
          <td>4.435720e-03</td>
          <td>5.025022e-03</td>
          <td>0.006670</td>
          <td>0.003554</td>
          <td>7.845605e-03</td>
          <td>1.008005e-02</td>
          <td>3.960339e-03</td>
          <td>3.036945e-03</td>
          <td>0.003117</td>
          <td>...</td>
          <td>0.003475</td>
          <td>0.002608</td>
          <td>0.002775</td>
          <td>3.532225e-03</td>
          <td>2.129010e-03</td>
          <td>2.228852e-03</td>
          <td>0.003252</td>
          <td>0.002370</td>
          <td>1.756987e-03</td>
          <td>0.001791</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.003851</td>
          <td>3.699328e-03</td>
          <td>4.168286e-03</td>
          <td>0.004854</td>
          <td>0.002994</td>
          <td>5.448732e-03</td>
          <td>6.593988e-03</td>
          <td>3.980438e-03</td>
          <td>3.154703e-03</td>
          <td>0.003448</td>
          <td>...</td>
          <td>0.003022</td>
          <td>0.002350</td>
          <td>0.002481</td>
          <td>3.278374e-03</td>
          <td>2.024578e-03</td>
          <td>2.391616e-03</td>
          <td>0.003117</td>
          <td>0.002198</td>
          <td>1.653013e-03</td>
          <td>0.001694</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.002354</td>
          <td>1.177163e-03</td>
          <td>7.357269e-03</td>
          <td>0.001766</td>
          <td>0.006474</td>
          <td>8.673617e-19</td>
          <td>1.765745e-03</td>
          <td>4.002354e-03</td>
          <td>1.471454e-03</td>
          <td>0.002943</td>
          <td>...</td>
          <td>0.005008</td>
          <td>0.002943</td>
          <td>0.005271</td>
          <td>2.354326e-03</td>
          <td>1.373357e-03</td>
          <td>2.942908e-03</td>
          <td>0.002943</td>
          <td>0.001668</td>
          <td>1.961938e-03</td>
          <td>0.001491</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.003868</td>
          <td>2.221431e-03</td>
          <td>4.322423e-03</td>
          <td>0.004708</td>
          <td>0.001923</td>
          <td>3.445849e-03</td>
          <td>4.697158e-03</td>
          <td>3.458959e-03</td>
          <td>2.385378e-03</td>
          <td>0.004449</td>
          <td>...</td>
          <td>0.003075</td>
          <td>0.002622</td>
          <td>0.002798</td>
          <td>3.301657e-03</td>
          <td>1.856778e-03</td>
          <td>3.852410e-03</td>
          <td>0.003431</td>
          <td>0.002144</td>
          <td>1.935102e-03</td>
          <td>0.001967</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.003751</td>
          <td>4.027594e-03</td>
          <td>3.811111e-03</td>
          <td>0.005776</td>
          <td>0.002533</td>
          <td>6.700149e-03</td>
          <td>6.753052e-03</td>
          <td>2.954127e-03</td>
          <td>2.605358e-03</td>
          <td>0.002080</td>
          <td>...</td>
          <td>0.003253</td>
          <td>0.002655</td>
          <td>0.002811</td>
          <td>3.212510e-03</td>
          <td>2.416699e-03</td>
          <td>1.757587e-03</td>
          <td>0.002634</td>
          <td>0.002759</td>
          <td>2.187340e-03</td>
          <td>0.002238</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.005351</td>
          <td>6.755972e-03</td>
          <td>6.900149e-03</td>
          <td>0.011177</td>
          <td>0.003617</td>
          <td>1.465794e-02</td>
          <td>1.286083e-02</td>
          <td>4.043154e-03</td>
          <td>3.186455e-03</td>
          <td>0.003346</td>
          <td>...</td>
          <td>0.003261</td>
          <td>0.002595</td>
          <td>0.002620</td>
          <td>3.920848e-03</td>
          <td>2.317094e-03</td>
          <td>2.733897e-03</td>
          <td>0.003766</td>
          <td>0.002597</td>
          <td>1.861708e-03</td>
          <td>0.001816</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.004696</td>
          <td>3.777616e-03</td>
          <td>4.867877e-03</td>
          <td>0.004757</td>
          <td>0.006611</td>
          <td>4.956457e-03</td>
          <td>4.988593e-03</td>
          <td>3.886430e-03</td>
          <td>3.290364e-03</td>
          <td>0.004243</td>
          <td>...</td>
          <td>0.003571</td>
          <td>0.003053</td>
          <td>0.003173</td>
          <td>3.416586e-03</td>
          <td>2.655298e-03</td>
          <td>3.506984e-03</td>
          <td>0.003352</td>
          <td>0.002803</td>
          <td>2.331214e-03</td>
          <td>0.002436</td>
        </tr>
        <tr>
          <th>13</th>
          <td>0.005584</td>
          <td>6.132790e-03</td>
          <td>6.949845e-03</td>
          <td>0.006113</td>
          <td>0.003225</td>
          <td>8.108133e-03</td>
          <td>7.162819e-03</td>
          <td>4.677292e-03</td>
          <td>3.787787e-03</td>
          <td>0.004263</td>
          <td>...</td>
          <td>0.004308</td>
          <td>0.003347</td>
          <td>0.003593</td>
          <td>3.038699e-03</td>
          <td>2.455895e-03</td>
          <td>2.746325e-03</td>
          <td>0.002917</td>
          <td>0.002756</td>
          <td>2.160478e-03</td>
          <td>0.002327</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.002904</td>
          <td>2.439805e-03</td>
          <td>2.850012e-03</td>
          <td>0.002966</td>
          <td>0.002524</td>
          <td>3.414085e-03</td>
          <td>4.333235e-03</td>
          <td>2.706258e-03</td>
          <td>2.365576e-03</td>
          <td>0.002380</td>
          <td>...</td>
          <td>0.004286</td>
          <td>0.003308</td>
          <td>0.003715</td>
          <td>2.193038e-03</td>
          <td>1.669442e-03</td>
          <td>1.748874e-03</td>
          <td>0.002037</td>
          <td>0.001887</td>
          <td>1.611750e-03</td>
          <td>0.001687</td>
        </tr>
        <tr>
          <th>15</th>
          <td>0.004993</td>
          <td>-8.673617e-19</td>
          <td>4.650353e-03</td>
          <td>0.004916</td>
          <td>0.001841</td>
          <td>6.776644e-03</td>
          <td>4.335691e-03</td>
          <td>5.033730e-03</td>
          <td>-1.734723e-18</td>
          <td>0.004271</td>
          <td>...</td>
          <td>0.004386</td>
          <td>0.004356</td>
          <td>0.005007</td>
          <td>3.214592e-03</td>
          <td>-4.336809e-19</td>
          <td>2.662572e-03</td>
          <td>0.002455</td>
          <td>0.002386</td>
          <td>2.168944e-03</td>
          <td>0.002538</td>
        </tr>
        <tr>
          <th>16</th>
          <td>0.002624</td>
          <td>2.752082e-03</td>
          <td>2.637459e-03</td>
          <td>0.002749</td>
          <td>0.002693</td>
          <td>2.493644e-03</td>
          <td>2.095285e-03</td>
          <td>2.781816e-03</td>
          <td>2.588301e-03</td>
          <td>0.002834</td>
          <td>...</td>
          <td>0.003447</td>
          <td>0.003385</td>
          <td>0.003378</td>
          <td>2.322611e-03</td>
          <td>2.273413e-03</td>
          <td>2.406070e-03</td>
          <td>0.002173</td>
          <td>0.002386</td>
          <td>2.222128e-03</td>
          <td>0.002300</td>
        </tr>
        <tr>
          <th>17</th>
          <td>0.001887</td>
          <td>3.803183e-03</td>
          <td>1.027182e-03</td>
          <td>0.001076</td>
          <td>0.001054</td>
          <td>3.921148e-03</td>
          <td>2.601776e-03</td>
          <td>2.865013e-03</td>
          <td>3.241096e-03</td>
          <td>0.000972</td>
          <td>...</td>
          <td>0.003991</td>
          <td>0.003982</td>
          <td>0.003991</td>
          <td>3.338288e-03</td>
          <td>3.593798e-03</td>
          <td>9.045875e-04</td>
          <td>0.000802</td>
          <td>0.003899</td>
          <td>3.766723e-03</td>
          <td>0.003843</td>
        </tr>
        <tr>
          <th>ALB-NYP</th>
          <td>0.003077</td>
          <td>2.965686e-03</td>
          <td>2.981998e-03</td>
          <td>0.003501</td>
          <td>0.001954</td>
          <td>4.155134e-03</td>
          <td>4.489645e-03</td>
          <td>3.318707e-03</td>
          <td>2.916787e-03</td>
          <td>0.003032</td>
          <td>...</td>
          <td>0.003129</td>
          <td>0.002638</td>
          <td>0.002924</td>
          <td>3.069298e-03</td>
          <td>2.416175e-03</td>
          <td>2.079249e-03</td>
          <td>0.002549</td>
          <td>0.002820</td>
          <td>2.555576e-03</td>
          <td>0.002609</td>
        </tr>
        <tr>
          <th>BOS-NYP</th>
          <td>0.002992</td>
          <td>2.886052e-03</td>
          <td>2.657136e-03</td>
          <td>0.003782</td>
          <td>0.002153</td>
          <td>3.734797e-03</td>
          <td>4.882234e-03</td>
          <td>3.117461e-03</td>
          <td>2.534942e-03</td>
          <td>0.002093</td>
          <td>...</td>
          <td>0.003180</td>
          <td>0.002686</td>
          <td>0.002940</td>
          <td>3.122747e-03</td>
          <td>2.213366e-03</td>
          <td>1.687653e-03</td>
          <td>0.002580</td>
          <td>0.002621</td>
          <td>2.119163e-03</td>
          <td>0.002233</td>
        </tr>
        <tr>
          <th>NYP-PHL</th>
          <td>0.002915</td>
          <td>2.974395e-03</td>
          <td>2.034712e-03</td>
          <td>0.002607</td>
          <td>0.001963</td>
          <td>3.293685e-03</td>
          <td>3.154119e-03</td>
          <td>2.822804e-03</td>
          <td>2.539638e-03</td>
          <td>0.001912</td>
          <td>...</td>
          <td>0.003308</td>
          <td>0.002975</td>
          <td>0.003298</td>
          <td>3.069107e-03</td>
          <td>2.581947e-03</td>
          <td>1.802644e-03</td>
          <td>0.001892</td>
          <td>0.003264</td>
          <td>2.812071e-03</td>
          <td>0.002992</td>
        </tr>
        <tr>
          <th>NYP-WAS</th>
          <td>0.002767</td>
          <td>2.631762e-03</td>
          <td>2.262933e-03</td>
          <td>0.003417</td>
          <td>0.002033</td>
          <td>2.985784e-03</td>
          <td>3.945229e-03</td>
          <td>3.188519e-03</td>
          <td>2.708899e-03</td>
          <td>0.002068</td>
          <td>...</td>
          <td>0.003125</td>
          <td>0.002708</td>
          <td>0.002922</td>
          <td>3.055433e-03</td>
          <td>2.272841e-03</td>
          <td>1.640844e-03</td>
          <td>0.002423</td>
          <td>0.002688</td>
          <td>2.277093e-03</td>
          <td>0.002441</td>
        </tr>
        <tr>
          <th>PHL-WAS</th>
          <td>0.002773</td>
          <td>2.906080e-03</td>
          <td>2.047320e-03</td>
          <td>0.003304</td>
          <td>0.001748</td>
          <td>3.659660e-03</td>
          <td>3.535641e-03</td>
          <td>2.553609e-03</td>
          <td>2.373301e-03</td>
          <td>0.001613</td>
          <td>...</td>
          <td>0.003349</td>
          <td>0.003006</td>
          <td>0.003210</td>
          <td>2.975808e-03</td>
          <td>2.459260e-03</td>
          <td>1.595743e-03</td>
          <td>0.002140</td>
          <td>0.003068</td>
          <td>2.564903e-03</td>
          <td>0.002709</td>
        </tr>
      </tbody>
    </table>
    <p>23 rows × 42 columns</p>
    </div>



.. code:: ipython3

    # Saves all centroids to an excel file
    total_stepdown_factors.to_excel("total_stepdown_factors.xlsx")

.. code:: ipython3

    # Saves top 5 city pair centroids to an excel file
    top5_mtx.to_excel('top5_clusters.xlsx')

Top 5 Station Pair Visualizations
---------------------------------

Here I am graphing the ridership patterns for the top 5 station pairs as
line graphs over time.

.. code:: ipython3

    def create_top_5_plot(top5_mtx, station_pair):
        """
            Creates a plot for a single top station pair.
         
            Creates a line plot displaying the ridership on each representative day 
            for a single station pair.
         
            Parameters
            ----------
            top5_mtx: DataFrame
                Cluster data in the form of a matrix for a single station pair.
            station_pair: str
                Name of the station pair.
            
            Generates a plot of the station pair.
            
        """
        plt.subplots(figsize=(14,8))
    
        x_coordinates = list(range(0,42))
    
        y1_coordinates = top5_mtx.loc[station_pair].values
    
        plt.title(station_pair)
        plt.ylabel('stnpairs')
        plt.xlabel('time')
        rgb = np.random.rand(3,)
    
        plt.plot(x_coordinates, y1_coordinates, color=rgb)
    
        plt.show()

ALB-NYC
~~~~~~~

.. code:: ipython3

    create_top_5_plot(top5_mtx, 'ALB-NYP')



.. image:: official_analysis_files/official_analysis_114_0.png


BOS-NYP
~~~~~~~

.. code:: ipython3

    create_top_5_plot(top5_mtx, 'BOS-NYP')



.. image:: official_analysis_files/official_analysis_116_0.png


NYP-PHL
~~~~~~~

.. code:: ipython3

    create_top_5_plot(top5_mtx, 'NYP-PHL')



.. image:: official_analysis_files/official_analysis_118_0.png


NYP-WAS
~~~~~~~

.. code:: ipython3

    create_top_5_plot(top5_mtx, 'NYP-WAS')



.. image:: official_analysis_files/official_analysis_120_0.png


PHL-WAS
~~~~~~~

.. code:: ipython3

    create_top_5_plot(top5_mtx, 'PHL-WAS')



.. image:: official_analysis_files/official_analysis_122_0.png


