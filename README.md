# Time Series Cluster Analysis

This is a time series cluster analysis for train ridership over seven years. My goal was to compare over 1000 station pairs, check their patterns over time (42 representative days), cluster them according to their patterns, and visualize each cluster. I was able to accomplish this using Python's Pandas, NumPy, and Scikit-learn.

### Prepping the Data
First the code sorts the data according to the needs of the analysis library, in this case, Skikit-learn's K-Means clustering library. I originally had a dataset with cluster inputs for every representative day over seven years. I needed to create an average cluster input for all seven years grouped by station pair, season, and day of week, so that there were just 42 cluster inputs for each station pair. I did this by finding the average daily ridership for each station pair on each representative day and dividing that by the average yearly ridership for that station pair. Once I found the cluster inputs, I created a pivot-table which contained the cluster inputs by day and station pair. I also created a dataframe with weights to show which station pairs were more important to our analysis. Lastly I separated the top 5 station pairs from the rest of the station pairs as each of these would represent their own cluster and could not be included in our analysis.

### Performing the Analysis
Once the pivot table and weights dataframe were created, I had all of the inputs needed for the K-means cluster analysis. I created a loop which would perform the analysis from 4 clusters up to 20 clusters and document the inertias, centroids, and cluster data for each. I then graphed the inertia results using MatPlotLib. We were looking for the lowest inertia with the least amount of clusters. My team and I decided on 18 clusters based on the inertia plot. Once I knew that we were choosing the analysis with 18 clusters, I created a dataframe which detailed each cluster's station pairs, total ridership, and the amount of station pairs in each cluster and saved that to an Excel file.

### Creating Visualizations
After documenting the cluster data, I created a visualization of each cluster to show their centroids and how the stations in each cluster relate to one another. I did this by creating a line plot in MatPlotLib and showing the cluster inputs over time for each station pair, highlighting the centroid in black. I also created visualizations for each of the top 5 station pairs.
