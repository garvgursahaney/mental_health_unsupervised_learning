# Mental Health Cluster Model 

This project makes direct use of **unsupervised machine learning** to analyze and group a particular workplace mental health survey data.
Instead of just basing a prediction, it focuses on discovering **natural patterns and groupings** among respondents.

The actual pipeline of the model cleans and encodes the data, applies **PCA** for dimensionality reduction, and uses
**K-Means clustering** to group similar responses.
After this, the cluster quality is evaluated using silhouette scores and visualized in a more 2-Dimensional PCA space.

This particular system is built for the dataset named **Mental Health in Tech Survey (2016)**.
The dataset can be found here for the full file download:
https://www.kaggle.com/osmi/mental-health-in-tech-2016 
If the dataset is missing or not downloaded, realistic synthetic data is then generated automatically to produce an expected graph to show the user what it is capable of.

This project can be run with:
```bash
pip install -r requirements.txt
python mental_health_cluster_model.py


Alternatively, the code can also be cloned from this repository with all necessary files within the zip compress folder.
