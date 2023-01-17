#import packages from different Modules 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import  NearestNeighbors #Importing a library for finding a nearest neighbors for DBSCAN.
from sklearn.cluster import DBSCAN  # module from sklearn. cluster to perfome a density based clustering using DBSCAN alogorithm
import seaborn as sns
from kneed import KneeLocator
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

# Summary statistics of breast cacncer db
print('\n Discribed Breast cancer db: ', breast_cancer_df.describe())

# Checking a shape of data
print('\n Shape of data: ', breast_cancer_df.shape)

# Extract columns from breast_cancer_df to process a clustering
breast_cancer_df_X = breast_cancer_df.loc[:,['diagnosis','perimeter_mean']].values

# shape of extracted dataframe
print('Shape of extracted DF: ', breast_cancer_df_X.shape)

'''
   To apply a DBSCAN algorithm we need neighborhood redius (eps) and minimum number of points (min_samples) as parameter.
   we can Compute this parameter using nearestneighbors
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
plt.plot(distances)
plt.title('Distance variation at the 10th neighbour')
plt.xlabel('Points')
plt.ylabel('Distance')

# KneeLocator to Detect Elbow point
i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
eps = float("{:.2f}".format(distances[knee.knee]))
print('eps: ', eps) # eps:  0.35

# plot destances with elbow point

knee.plot_knee()
plt.xlabel('Points')
plt.ylabel('Distance')
plt.show()
