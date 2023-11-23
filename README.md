# time-series-cluster-analysis
This is a time series cluster analysis for ridership over seven years. My goal was to compare over 1000 station pairs, check their patterns over time (42 representative days), cluster them according to their patterns, and visualize each cluster. I was able to accomplish this using Python's Pandas, NumPy, and Scikit-learn.

## What my code does?
First the code sorts the data according to the needs of the analysis library, in this case, Skikit-learn's K-Means clustering library. I originally had a dataset with cluster inputs for every representative day over seven years. I needed to create an average cluster input for all seven years grouped by station pair, season, and day of week, so that there were just 42 cluster inputs for each station pair. 
