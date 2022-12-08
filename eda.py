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
   pd.set_option('display.max_columns', None)
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
                    'Given?': {'No':0, 'Yes':1},
                    'PG?': {'No':0, 'Yes':1}})
   
   df = pd.get_dummies(df, columns = ['Gender'])
   
   
   # BINNING MAJORS
   
   businessList = ['Business Administration', 'Economics', 'Business Adm & Acct', 'Accounting & Bus Adm', 'Public Accounting', 'Commerce']
   artsList = ['Art', 'Theater', 'Music', 'Studio Art', 'Drama']
   humanitiesList = ['History', 'English', 'Philosophy', 'Spanish', 'Art History', 'French', 'Religion', 'Classics', 'East Asian Studies', 'Romance Languages', 'German Language', 'German', 'Mediev & Renaiss Stds Minor', 'German Literature', 'Russian Studies', 'Russian Area Studies', 'Latin American & Caribbean Studies', 'Afro Amer Studies', 'Creative Writing']
   socialSciencesList = ['Politics', 'Journalism & Mass Communication', 'Sociology & Anthro', 'Archaelogy & Anthro', 'Poverty & Human Capability Studies']
   stemList = ['Biology', 'Psychology', 'Chemistry', 'Mathematics', 'Geology', 'Computer Science', 'Physics-Engineering', 'Neuroscience', 'Biochemistry', 'Physics', 'Geology-Environmental Studies', 'Chemistry-Engineering', 'Environmental Studies', 'Psych-Biol-Sociology', 'Cognitive & Behavioral Science', 'Nat Science & Math']
   otherList = ['Public Policy', 'Independent Work', 'COMBINAT LAW 3-3 W&L']
   colList = ['Business', 'Arts', 'Humanities', 'Social Sciences', 'STEM', 'OtherMajor']
   listList = [businessList, artsList, humanitiesList, socialSciencesList, stemList, otherList]
   
   # major 1
   df = df.assign(Business1 = lambda dataframe: dataframe['Major1'].map(lambda major: 1 if major in businessList else 0))
   df = df.assign(Arts1     = lambda dataframe: dataframe['Major1'].map(lambda major: 1 if major in artsList     else 0))
   df = df.assign(Humanities1 = lambda dataframe: dataframe['Major1'].map(lambda major: 1 if major in humanitiesList else 0))
   df = df.assign(SocialSciences1 = lambda dataframe: dataframe['Major1'].map(lambda major: 1 if major in socialSciencesList else 0))
   df = df.assign(STEM1 = lambda dataframe: dataframe['Major1'].map(lambda major: 1 if major in stemList else 0))
   df = df.assign(OtherMajor1 = lambda dataframe: dataframe['Major1'].map(lambda major: 1 if major in otherList else 0))
   
   # major 2
   df = df.assign(Business2 = lambda dataframe: dataframe['Major2'].map(lambda major: 1 if major in businessList else 0))
   df = df.assign(Arts2     = lambda dataframe: dataframe['Major2'].map(lambda major: 1 if major in artsList     else 0))
   df = df.assign(Humanities2 = lambda dataframe: dataframe['Major2'].map(lambda major: 1 if major in humanitiesList else 0))
   df = df.assign(SocialSciences2 = lambda dataframe: dataframe['Major2'].map(lambda major: 1 if major in socialSciencesList else 0))
   df = df.assign(STEM2 = lambda dataframe: dataframe['Major2'].map(lambda major: 1 if major in stemList else 0))
   df = df.assign(OtherMajor2 = lambda dataframe: dataframe['Major2'].map(lambda major: 1 if major in otherList else 0))
   
   # major 3
   df = df.assign(Business3 = lambda dataframe: dataframe['Major3'].map(lambda major: 1 if major in businessList else 0))
   df = df.assign(Arts3     = lambda dataframe: dataframe['Major3'].map(lambda major: 1 if major in artsList     else 0))
   df = df.assign(Humanities3 = lambda dataframe: dataframe['Major3'].map(lambda major: 1 if major in humanitiesList else 0))
   df = df.assign(SocialSciences3 = lambda dataframe: dataframe['Major3'].map(lambda major: 1 if major in socialSciencesList else 0))
   df = df.assign(STEM3 = lambda dataframe: dataframe['Major3'].map(lambda major: 1 if major in stemList else 0))
   df = df.assign(OtherMajor3 = lambda dataframe: dataframe['Major3'].map(lambda major: 1 if major in otherList else 0))
   
   # minor 1
   df = df.assign(Business4 = lambda dataframe: dataframe['Minor1'].map(lambda minor: 1 if minor in businessList else 0))
   df = df.assign(Arts4     = lambda dataframe: dataframe['Minor1'].map(lambda minor: 1 if minor in artsList     else 0))
   df = df.assign(Humanities4 = lambda dataframe: dataframe['Minor1'].map(lambda minor: 1 if minor in humanitiesList else 0))
   df = df.assign(SocialSciences4 = lambda dataframe: dataframe['Minor1'].map(lambda minor: 1 if minor in socialSciencesList else 0))
   df = df.assign(STEM4 = lambda dataframe: dataframe['Minor1'].map(lambda minor: 1 if minor in stemList else 0))
   df = df.assign(OtherMajor4 = lambda dataframe: dataframe['Minor1'].map(lambda minor: 1 if minor in otherList else 0))
   
   # minor 2
   df = df.assign(Business5 = lambda dataframe: dataframe['Minor2'].map(lambda minor: 1 if minor in businessList else 0))
   df = df.assign(Arts5     = lambda dataframe: dataframe['Minor2'].map(lambda minor: 1 if minor in artsList     else 0))
   df = df.assign(Humanities5 = lambda dataframe: dataframe['Minor2'].map(lambda minor: 1 if minor in humanitiesList else 0))
   df = df.assign(SocialSciences5 = lambda dataframe: dataframe['Minor2'].map(lambda minor: 1 if minor in socialSciencesList else 0))
   df = df.assign(STEM5 = lambda dataframe: dataframe['Minor2'].map(lambda minor: 1 if minor in stemList else 0))
   df = df.assign(OtherMajor5 = lambda dataframe: dataframe['Minor2'].map(lambda minor: 1 if minor in otherList else 0))
                                                                          
   # total
   df['Business'] = df['Business1'] + df['Business2'] + df['Business3'] + df['Business4'] + df['Business5']
   df['Arts'] = df['Arts1'] + df['Arts2'] + df['Arts3'] + df['Arts4'] + df['Arts5']
   df['Humanities'] = df['Humanities1'] + df['Humanities2'] + df['Humanities3'] + df['Humanities4'] + df['Humanities5']
   df['SocialSciences'] = df['SocialSciences1'] + df['SocialSciences2'] + df['SocialSciences3'] + df['SocialSciences4'] + df['SocialSciences5']
   df['STEM'] = df['STEM1'] + df['STEM2'] + df['STEM3'] + df['STEM4'] + df['STEM5']
   df['OtherMajor'] = df['OtherMajor1'] + df['OtherMajor2'] + df['OtherMajor3'] + df['OtherMajor4'] + df['OtherMajor5']
   
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
           'Reunion Class Committee',
           'Business',
           'Arts',
           'Humanities',
           'SocialSciences',
           'STEM', 
           'OtherMajor']]
  
   y = df[['Given?']]
   
# prepare data
   def fitData(df,X):

      # scale the data
      sc = StandardScaler()
   
      sc.fit(X)
      X = sc.transform(X)
   
      return X, y
      
def plots(df):
   
   cols = df.columns.values
   spm = pd.plotting.scatter_matrix(df, figsize = (14,14))
   
   cm = np.corrcoef(df[cols].values.T)
   hm = heatmap(cm, row_names = cols, column_names = cols, figsize = (14,14))  
   
def information(df):
   
   #print("\nLength before dropping NAs:", lengthBeforeDrop)
   #print("\nLength of dataset after dropping NAs:", len(df))
   #print("\nDescriptive Statistics:\n", df.describe())
   #print("\nData types:\n", df.dtypes) 
   #print("\nMissing data before manipulation:\n", nanCounts)
   #print("\nMissing data after manipulation:\n", df.isnull().sum())
   def returnCounts(df, lst):
      for item in lst:
         newSeries = df[item].value_counts()
         print("\n", item, ":\n", newSeries.to_string())
   returnCounts(df, df.columns)
   
if __name__ == "__main__":
   returnData()
   #plots(df)
   information(df)
