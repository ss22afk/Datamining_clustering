#import packages from different Modules 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import  NearestNeighbors as nn #Importing a library for finding a nearest neighbors for DBSCAN.
from sklearn.cluster import DBSCAN  # module from sklearn. cluster to perfome a density based clustering using DBSCAN alogorithm
import seaborn as sns
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
breast_Cancer_db = readFileFromLocation('BreastCancerDataSet.csv')
print('Breast Cancer DB: ', breast_Cancer_db.head(10))

# Check For missing value
print('\n Total Count of Missing Value: ', breast_Cancer_db.isna().sum())

# Information about Breast Cancer db
print('\n Information about DB: ', breast_Cancer_db.info())

# Drop a unneccessary column from a breast cancer db 
breast_Cancer_db = breast_Cancer_db.drop(['id', 'Unnamed: 32'], axis=1)
print('After column drop: ', breast_Cancer_db.head(10))

# Summary statistics of breast cacncer db
print('\n Discribed Breast cancer db: ', breast_Cancer_db.describe())

# Checking a shape of data
print('\n Shape of data: ', breast_Cancer_db.shape)


