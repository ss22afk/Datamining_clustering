#import packages from different Modules 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import  NearestNeighbors #Importing a library for finding a nearest neighbors for DBSCAN.
from sklearn.cluster import DBSCAN  # module from sklearn. cluster to perfome a density based clustering using DBSCAN alogorithm
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import seaborn as sns
from sklearn.metrics import silhouette_score
import os

def getDirecotry_path():
    '''
        This function is fetch a directory file path using os module and return a csvfile path.
    '''
    #get Current directory path
    current_direcotry_path = os.getcwd()
    print('Current direcotry path: ', current_direcotry_path) # Current direcotry path:s:\Visual_Studio\ASD_Assignment_Clustering

    #Change directory from current to previous
    os.chdir('..')
    chg_file_path = os.getcwd()
    print('csvFiles  direcotry path: ', chg_file_path) # csvFiles  direcotry path:  s:\Visual_Studio

    #change direcotory from current location to 'csvFiles'
    os.chdir('CSVFiles')
    csv_file_path = os.getcwd()
    print('CSV filepath: ', csv_file_path) # CSV filepath:  s:\Visual_Studio\csvFiles
    
    return csv_file_path
    
def readFileFromLocation(filename):
    '''
        This function is use for read file from "csvFiles" directory. 
        After a reading file processa load file and retrun a dataframe
        filename: Name of file as parameter.
    '''
    dataframe = pd.read_csv(getDirecotry_path()+'/'+filename)
    return dataframe

# Load a data from a CSVFiles Directory using readFilefrom readFileFromLocation()
breast_cancer_df = readFileFromLocation('BreastCancerDataSet.csv')
print('Breast Cancer DB: ', breast_cancer_df.head(10))

# Check For missing value
print('\n Total Count of Missing Value: ', breast_cancer_df.isna().sum())

# Information about Breast Cancer db
print('\n Information about DB: ', breast_cancer_df.info())

# Drop a unneccessary column from a breast cancer db 
breast_cancer_df = breast_cancer_df.drop(['id', 'Unnamed: 32'], axis=1)
print('After column drop: ', breast_cancer_df.head(10))

# Map dignosis value form Malignant  to 1 and Benign to 0
breast_cancer_df['diagnosis'] = breast_cancer_df['diagnosis'].map({'M':1, 'B':0})
print('\n diagnosis value mapped: ', breast_cancer_df.head(10))

# Extract columns from breast_cancer_df to process a clustering
st = StandardScaler()
breast_cancer_df_X = pd.DataFrame(st.fit_transform(breast_cancer_df), columns=breast_cancer_df.columns)
#breast_cancer_df_X = breast_cancer_df.drop('diagnosis', axis=1).values
#breast_cancer_df_X = StandardScaler().fit_transform(breast_cancer_df_X)

# Summary statistics of breast cacncer db
print('\n Discribed Breast cancer db: ', breast_cancer_df.describe())

# Checking a shape of data
print('\n Shape of data: ', breast_cancer_df.shape)


# shape of extracted dataframe
print('Shape of extracted DF: ', breast_cancer_df_X.shape)

'''
   To apply a DBSCAN algorithm we need neighborhood redius (eps) and minimum number of points (min_samples) as parameter.
   we can Compute this parameter using nearestneighbors.
'''
# Creating an object of the NearestNeighbors class and fitting the data to the object
'''
n_neighbors : To seek nearestneighobor
'''
nearestneighbor = NearestNeighbors(n_neighbors=11).fit(breast_cancer_df_X)

# Finding a nearest neighbors 
distances, indices = nearestneighbor.kneighbors(breast_cancer_df_X)
print('Distanced: and indices: ', distances, indices)

# Sort Distances with 10th nearestneighbor and plot the variation
distances = np.sort(distances[:10], axis=0)
distances = distances[:,1]
plt.figure(figsize=(5,5))
plt.plot(distances, label= 'data')
plt.title('Distance variation at the 10th neighbour')
plt.legend()
plt.xlabel('Points')
plt.ylabel('Distance')

# KneeLocator to Detect Elbow point
i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
eps_cal = float("{:.2f}".format(distances[knee.knee]))
print('eps_cal: ', eps_cal) # eps_cal:  2.7

# plot destances with elbow point
knee.plot_knee()
plt.xlabel('Points')
plt.ylabel('Distance')


'''
    Appling DBSCAN algorith on Breast cancer Dataset, ploting to scatter plot, calculate number of clusters.
    eps_cal = 0.35
    min_sample_pre = 10
'''
# Predicated value of min_sample from nearestneighbors
min_sameple_pre = 10

dbscan = DBSCAN(eps = eps_cal, min_samples = min_sameple_pre).fit(breast_cancer_df_X)
breast_cancer_df_X_pred = dbscan.fit_predict(breast_cancer_df_X)
labels = dbscan.labels_


# Number of clusters in a labels, and noise
n_clusters = len(set(labels))
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters :', n_clusters)
print('Estimated number of noise points:' , n_noise_)
print(f"silhouette_score: {silhouette_score(breast_cancer_df_X,labels)}")


# Scatter plot
plt.figure(figsize=(5,5))
sns.scatterplot(x=breast_cancer_df_X['diagnosis'], y=breast_cancer_df_X['perimeter_mean'], hue=labels, palette='viridis').set(title = 'cluster of diagnosis and perimeter_mean.')
plt.show()
