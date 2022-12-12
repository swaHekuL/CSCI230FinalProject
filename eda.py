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
   
   # BINNING INDUSTRIES
   Legal_Services = ['Legal Services']
   Investments = ['Investments/Finance', 'Banking/Commerce', 'Accounting ', 'Consulting', 'Insurance/Diversified Financials', 'Financial Services', 'Finance - Investment Management', 'Finance - Investment Banking', 'Finance - Private Equity', 'Insurance', 'Bus. Other', 'Finance - Hedge Fund', 'Finance - Sales & Trading', 'Financial Services - Diversified', 'Finance - Venture Capital', 'Finance - Commercial Banking', 'Finance - Consumer Banking', 'Finance - LBO']
   Real_Estate = ['Real Estate - Residential', 'Real Estate - Commercial', 'Real Estate - Finance']
   Education = ['Education - High Ed', 'Education - Pre K - 12', 'Education - Other']
   Healthcare = ['Healthcare - Delivery/Provider', 'Healthcare - Services/Other', 'Healthcare - Medicine/Research', 'Healthcare - Pharmaceuticals', 'Healthcare - Medical Devices', 'Healthcare - Biotechnology']
   Government = ['Government - Federal', 'Government - State/Local', 'Government - Armed Forces', 'Government - Other', 'Government - International']
   Retail = ['Retail/Wholesale', 'Food/Beverage', 'Hospitality Services', 'Consumer Products', 'Food Products', 'Textiles & Apparel']
   Public_Relations = ['Advertising/Public Relations', 'Public Charity', 'Non-Profit Organization', 'Political Campaign', 'Human Resources', 'International Development', 'Development-Fundraising', 'Human/Social Services', 'Economic/Community Development', 'Private Charitable Organization']
   Utilities = ['Energy', 'Manufacturing', 'Construction', 'Transportation', 'Telecommunications', 'Aerospace & Defense', 'Agriculture', 'Engineering', 'Environment ', 'Utilities', 'Engineering & Science', 'Paper and Forest Products', 'Transportation Equipment']
   Tech = ['Tech - Software', 'Tech - Internet Services', 'Tech - Services', 'Tech - Consumer Electronics', 'Tech - Hardware', 'Computer', 'Electronics']
   Entertainment = ['Entertainment/Leisure/Sports', 'Print/Publishing', 'Arts/Culture', 'Broadcast/Film/Multimedia']
   Other = ['Foundation - Independent', 'Organizations/Associations/Society', 'Professional Services', 'Religion', 'Foundation - Community', 'Architecture', 'Foundation - Other', 'Foundation - Corp-Sponsored', 'Scientific Research', 'Chemical', 'Service', 'Foundation - Family', 'Foundations/Philanthropy', 'Foundation', 'Foundation - Private', 'Foundation - Operating']
   
   # industry
   df = df.assign(Legal_Services0   = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Legal_Services   else 0))
   df = df.assign(Investments0      = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Investments      else 0))
   df = df.assign(Real_Estate0      = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Real_Estate      else 0))
   df = df.assign(Education0        = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Education        else 0))
   df = df.assign(Healthcare0       = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Healthcare       else 0))
   df = df.assign(Government0       = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Government       else 0))
   df = df.assign(Retail0           = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Retail           else 0))
   df = df.assign(Public_Relations0 = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Public_Relations else 0))
   df = df.assign(Utilities0        = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Utilities        else 0))
   df = df.assign(Tech0             = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Tech             else 0))
   df = df.assign(Entertainment0    = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Entertainment    else 0))
   df = df.assign(Other0            = lambda dataframe: dataframe['Industry'].map(lambda ind: 1 if ind in Other            else 0))
   
   # industry.1
   df = df.assign(Legal_Services1   = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Legal_Services   else 0))
   df = df.assign(Investments1      = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Investments      else 0))
   df = df.assign(Real_Estate1      = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Real_Estate      else 0))
   df = df.assign(Education1        = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Education        else 0))
   df = df.assign(Healthcare1       = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Healthcare       else 0))
   df = df.assign(Government1       = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Government       else 0))
   df = df.assign(Retail1           = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Retail           else 0))
   df = df.assign(Public_Relations1 = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Public_Relations else 0))
   df = df.assign(Utilities1        = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Utilities        else 0))
   df = df.assign(Tech1             = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Tech             else 0))
   df = df.assign(Entertainment1    = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Entertainment    else 0))
   df = df.assign(Other1            = lambda dataframe: dataframe['Industry.1'].map(lambda ind: 1 if ind in Other            else 0))
      
   # industry.2
   df = df.assign(Legal_Services2   = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Legal_Services   else 0))
   df = df.assign(Investments2      = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Investments      else 0))
   df = df.assign(Real_Estate2      = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Real_Estate      else 0))
   df = df.assign(Education2        = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Education        else 0))
   df = df.assign(Healthcare2       = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Healthcare       else 0))
   df = df.assign(Government2       = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Government       else 0))
   df = df.assign(Retail2           = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Retail           else 0))
   df = df.assign(Public_Relations2 = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Public_Relations else 0))
   df = df.assign(Utilities2        = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Utilities        else 0))
   df = df.assign(Tech2             = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Tech             else 0))
   df = df.assign(Entertainment2    = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Entertainment    else 0))
   df = df.assign(Other2            = lambda dataframe: dataframe['Industry.2'].map(lambda ind: 1 if ind in Other            else 0))
      
   # industry.3
   df = df.assign(Legal_Services3   = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Legal_Services   else 0))
   df = df.assign(Investments3      = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Investments      else 0))
   df = df.assign(Real_Estate3      = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Real_Estate      else 0))
   df = df.assign(Education3        = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Education        else 0))
   df = df.assign(Healthcare3       = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Healthcare       else 0))
   df = df.assign(Government3       = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Government       else 0))
   df = df.assign(Retail3           = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Retail           else 0))
   df = df.assign(Public_Relations3 = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Public_Relations else 0))
   df = df.assign(Utilities3        = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Utilities        else 0))
   df = df.assign(Tech3             = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Tech             else 0))
   df = df.assign(Entertainment3    = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Entertainment    else 0))
   df = df.assign(Other3            = lambda dataframe: dataframe['Industry.3'].map(lambda ind: 1 if ind in Other            else 0))
   
   # industry.4
   df = df.assign(Legal_Services4   = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Legal_Services   else 0))
   df = df.assign(Investments4      = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Investments      else 0))
   df = df.assign(Real_Estate4      = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Real_Estate      else 0))
   df = df.assign(Education4        = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Education        else 0))
   df = df.assign(Healthcare4       = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Healthcare       else 0))
   df = df.assign(Government4       = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Government       else 0))
   df = df.assign(Retail4           = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Retail           else 0))
   df = df.assign(Public_Relations4 = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Public_Relations else 0))
   df = df.assign(Utilities4        = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Utilities        else 0))
   df = df.assign(Tech4             = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Tech             else 0))
   df = df.assign(Entertainment4    = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Entertainment    else 0))
   df = df.assign(Other4            = lambda dataframe: dataframe['Industry.4'].map(lambda ind: 1 if ind in Other            else 0))
   
   # total
   df['Legal_Services'] = df['Legal_Services0'] + df['Legal_Services1'] + df['Legal_Services2'] + df['Legal_Services3'] + df['Legal_Services4']
   df['Investments'] = df['Investments0'] + df['Investments1'] + df['Investments2'] + df['Investments3'] + df['Investments4']
   df['Real_Estate'] = df['Real_Estate0'] + df['Real_Estate1'] + df['Real_Estate2'] + df['Real_Estate3'] + df['Real_Estate4']
   df['Education'] = df['Education0'] + df['Education1'] + df['Education2'] + df['Education3'] + df['Education4']
   df['Healthcare'] = df['Healthcare0'] + df['Healthcare1'] + df['Healthcare2'] + df['Healthcare3'] + df['Healthcare4']
   df['Government'] = df['Government0'] + df['Government1'] + df['Government2'] + df['Government3'] + df['Government4']
   df['Retail'] = df['Retail0'] + df['Retail1'] + df['Retail2'] + df['Retail3'] + df['Retail4']
   df['Public_Relations'] = df['Public_Relations0'] + df['Public_Relations1'] + df['Public_Relations2'] + df['Public_Relations3'] + df['Public_Relations4']
   df['Utilities'] = df['Utilities0'] + df['Utilities1'] + df['Utilities2'] + df['Utilities3'] + df['Utilities4']
   df['Tech'] = df['Tech0'] + df['Tech1'] + df['Tech2'] + df['Tech3'] + df['Tech4']
   df['Entertainment'] = df['Entertainment0'] + df['Entertainment1'] + df['Entertainment2'] + df['Entertainment3'] + df['Entertainment4']
   df['Other'] = df['Other0'] + df['Other1'] + df['Other2'] + df['Other3'] + df['Other4']
   '''
   # indDict = {}
   # for item in df['Industry']:
   #    if item not in indDict:
   #       indDict[item] = 1
   #    elif item in indDict:
   #       indDict[item] = indDict[item] + 1
   # for item in df['Industry.1']:
   #    if item not in indDict:
   #       indDict[item] = 1
   #    elif item in indDict:
   #       indDict[item] = indDict[item] + 1
   # for item in df['Industry.2']:
   #    if item not in indDict:
   #       indDict[item] = 1
   #    elif item in indDict:
   #       indDict[item] = indDict[item] + 1
   # for item in df['Industry.3']:
   #    if item not in indDict:
   #       indDict[item] = 1
   #    elif item in indDict:
   #       indDict[item] = indDict[item] + 1 
   # indDict = dict(sorted(indDict.items(), key = lambda item: item[1], reverse = True))
   # for item in indDict:
   #    print(item,'-', indDict[item])
   '''
   
   
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
           'OtherMajor',
           'Legal_Services',
           'Investments',
           'Real_Estate',
           'Education',
           'Healthcare',
           'Government',
           'Retail',
           'Public_Relations',
           'Utilities',
           'Tech',
           'Entertainment',
           'Other']]
  
   #returnCounts(X, X.columns)
   
   y = df[['Given?']]
   
   df = df[['Years Out', 
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
           'OtherMajor',
           'Legal_Services',
           'Investments',
           'Real_Estate',
           'Education',
           'Healthcare',
           'Government',
           'Retail',
           'Public_Relations',
           'Utilities',
           'Tech',
           'Entertainment',
           'Other',
           'Given?']]
   
   # scale the data
   sc = StandardScaler()
   # sc = MinMaxScaler()    
   # sc = MaxAbsScaler()
   # sc = RobustScaler()
   # sc = QuantileTransformer()
   # sc = PowerTransformer()
   # sc = Normalizer()

   sc.fit(X)
   X = sc.transform(X)

   return X, y
   
def plots(df):
   
   cols = df.columns.values
   spm = pd.plotting.scatter_matrix(df, figsize = (30,30))
   
   cm = np.corrcoef(df[cols].values.T)
   hm = heatmap(cm, row_names = cols, column_names = cols, figsize = (20,20))  
   
def information(df):
   
   #print("\nLength before dropping NAs:", lengthBeforeDrop)
   #print("\nLength of dataset after dropping NAs:", len(df))
   #print("\nDescriptive Statistics:\n", df.describe())
   #print("\nData types:\n", df.dtypes) 
   #print("\nMissing data before manipulation:\n", nanCounts)
   #print("\nMissing data after manipulation:\n", df.isnull().sum())
   
   returnCounts(df, df.columns)
 
def returnCounts(df, lst):
   for item in lst:
      newSeries = df[item].value_counts()
      print("\n", item, ":\n", newSeries.to_string())  
 
if __name__ == "__main__":
   returnData()
   #plots(df)
   #information(df)
