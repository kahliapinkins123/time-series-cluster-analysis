<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Time Series Cluster Analysis</h3>

  <p align="center">
    This is a time series cluster analysis for train ridership over seven years. My goal was to compare over 1000 station pairs, check their patterns over time (42 representative days), cluster them according to their patterns, and visualize each cluster. I was able to accomplish this using Python's Pandas, NumPy, and Scikit-learn.
    <br />
    <br />
    <a href="https://kahliapinkins123.wixsite.com/kahliapinkins/about-3"><strong>Cluster Analysis Portfolio Page »</strong></a>
    <br />
    
    
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#prep">Prepping the Data</a></li>
    <li><a href="#analysis">Performing the Analysis</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contact">Contact Me</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
<div id="about-the-project"></div>

## About The Project

I needed to create a model which predicted ridership in over 1000 train station-pairs for a particular train company. To simplify this process, I decided to create a time-series cluster analysis comparing station-pairs and their ridership patterns over the course of 7 years. The cluster analysis would group all station-pairs with similar ridership patterns together, allowing us to predict future ridership patterns based on each cluster rather than by individual station-pair. In order to do this my team and I determined 6 representative weeks (the 4 seasons, Thanksgiving, and Christmas) for each year, giving us a total of 42 reperesentative days (7 weekdays in all 6 weeks) to observe ridership patterns over each year. Using Python libraries, I was able to create a cluster analysis which clearly showed which station-pairs had similar ridership patterns over time.


<div id="prep"></div>

## Prepping the Data

I started by collecting a dataset which contained the ridership data for every single ride taken to and from these station-pairs from 2013 to 2019. After cleaning and parsing through the data for outliers and missing values, I created a cluster input for each station-pair on each representative day by aggregating them by year and getting their weighted averages by season and day of week. This gave me cluster inputs for every representative day over all 7 years, so a total of 294 inputs per station-pair. I needed to create an average cluster input over all 7 years, grouped by station pair, season, and day of week, so that there were just 42 cluster inputs for each station-pair. I did this by finding the average daily ridership for each station pair on each representative day and dividing that by the average yearly ridership for that station-pair. Once I found the 42 cluster inputs, I created a pivot-table to use in the K-Means clustering algorithm. I also created a dataframe with weights to show which station pairs were more important to our analysis. Lastly I separated the top 5 station pairs from the rest of the station pairs as each of these would represent their own cluster and could not be included in our analysis.

<div id="analysis"></div>

## Performing the Analysis
Once the pivot table and weights dataframe were created, I had all of the inputs needed for the K-means cluster analysis. I created a loop which would perform the analysis from 4 clusters up to 20 clusters and document the inertias, centroids, and cluster data for each. I then graphed the inertia results using MatPlotLib. We were looking for the lowest inertia with the least amount of clusters. My team and I decided on 18 clusters based on the inertia plot. Once I knew that we were choosing the analysis with 18 clusters, I created a dataframe which detailed each cluster's station pairs, total ridership, and the amount of station pairs in each cluster and saved that to an Excel file.

<div id="results"></div>

## Results
After documenting the cluster data, I created a visualization of each cluster to show their centroids and how the stations in each cluster relate to one another. I did this by creating a line plot in MatPlotLib and showing the cluster inputs over time for each station pair, highlighting the centroid in black. I also created visualizations for each of the top 5 station pairs.

<div id="contact"></div>
