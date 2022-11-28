# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:52:23 2022

@author: luke Haws, Jared Cordova, Dario Fumarola, Ryan Messick
"""


# import statements
import pandas as pd
import numpy as np
from mlxtend.plotting import heatmap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, Normalizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

global df

def returnData():
   
# read in df
   pd.set_option('display.max_columns', 30)
   pd.options.mode.chained_assignment = None
   global df
   df = pd.read_excel('C:/Users/lthaw/OneDrive/CSCI 230/Final Project/LH Project.xlsx')
   global lengthBeforeDrop
   lengthBeforeDrop = len(df)
   
   global nanCounts
   nanCounts = df.isnull().sum()

   '''
   NOTES
   
   Drop age, use years out from graduation
   SMOTE for the output bc class imbalance
   drop PG
   bin all alumni engagement and try not binning
   bin industry similar to majors
   
   
   
   '''
   df['Years Out'] = 2022 - df['Class']

   df = df.replace({'Spouse a Grad': {np.nan:0, 'Yes':1},
                    'Academic Activities': {np.nan:0, 'Yes':1},
                    'Academic Honors': {np.nan:0, 'Yes':1},
                    'Activities': {np.nan:0, 'Yes':1},
                    'Varsity Athletics': {np.nan:0, 'Yes':1},
                    'All American': {np.nan:0, 'Yes':1},
                    'Honor Societies': {np.nan:0, 'Yes':1},
                    'UG Academic Honors': {np.nan:0, 'Yes':1},
                    'Alumni Admissions Program': {np.nan:0, 'Yes':1},
                    'Alumni Board': {np.nan:0, 'Yes':1},
                    'Chapter Volunteers': {np.nan:0, 'Yes':1},
                    'Reunion Class Committee': {np.nan:0, 'Yes':1},
                    'Given': {'No':0, 'Yes':1}})
   
   
      
   X = df[['Years Out', 
           'Spouse a Grad',
           'Academic Activities',
           'Activities',
           'Varsity Athletics',
           'All American',
           'Honor Societies',
           'UG Academic Honors',
           'Alumni Admissions Program',
           'Alumni Board',
           'Chapter Volunteers',
           'Reunion Class Committee']]
  
   y = df[['Given?']]
   
# prepare data
   def fitData(df):

      # scale the data
      sc = StandardScaler()
   
      sc.fit(X)
      X = sc.transform(X)
   
      return X, y
   #X,y = fitData(df)
def plots(df):
   
   cols = df.columns.values
   spm = pd.plotting.scatter_matrix(df, figsize = (14,14))
   
   cm = np.corrcoef(df[cols].values.T)
   hm = heatmap(cm, row_names = cols, column_names = cols, figsize = (14,14))  
   
def information(df):
   
   print("\nLength before dropping NAs:", lengthBeforeDrop)
   print("\nLength of dataset after dropping NAs:", len(df))
   print("\nDescriptive Statistics:\n", df.describe())
   print("\nData types:\n", df.dtypes) 
   print("\nMissing data before manipulation:\n", nanCounts)
   print("\nMissing data after manipulation:\n", df.isnull().sum())
   def returnCounts(df, lst):
      for item in lst:
         print("\n", item, ":\n", df[item].value_counts())
   #returnCounts(df, df.columns)
   
if __name__ == "__main__":
   returnData()
   #plots(df)
   information(df)
