import numpy as np # importing numpy 
import pandas as pd# importing pandas for data struxturing 
import matplotlib.pyplot as plt # matplotlib for visualization
de = pd.read_csv(r"C:\Users\sheik\Downloads\Data.csv") # rwading the given data set
x = de.iloc[:,:-1].values # slicing the data first 3 columns as independent variable
y = de.iloc[:,3].values# slicing last column as independent variable
from sklearn.impute import SimpleImputer # importing the imputers which are used to clean the data in ml


imputer = SimpleImputer(strategy='mean ' )#can use median , mode ,most recent etc # saves the Simple imputer in a variable
imputer = imputer.fit(x[:, 1:3]) #fitting the entire rows of column one and two which are numerical in model

x[:,1:3] = imputer.transform(x[:,1:3]) # using transform method to replace the missing vlaues with mean values by deefault
