import uproot
import pandas as pd
import numpy
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from array import array



file_list = ["BDTPreds.csv", "MLPPreds.csv", "kNNPreds.csv"]

def Stacker(file_list, include_signal=True, include_bg=True, outname="StackedPreds.csv"):
     """ Builds combined dataframe from output of several models and outputs it to file.

     Parameters
     ----------
     file_list : list of str
        List of names of files containing input data.
     include_signal : bool
        Controls whether signal events are included in output dataframe. Default is true/yes.
     include_bg : bool
        Controls whether bg events are included in output dataframe. Default is true/yes.
     outname : str
        Name of output file. Default is StackedPreds.csv



     """
     comb_df = pd.DataFrame()
     for f in file_list:
        temp_df = pd.read_csv(f)
        comb_df = pd.concat([comb_df, temp_df], axis=1)

     comb_df = comb_df.loc[:,~comb_df.columns.duplicated()]
     if (not include_signal): 
         comb_df = comb_df[comb_df.label != 1].reset_index()
     if (not include_bg): 
         comb_df = comb_df[comb_df.label != 0].reset_index()

     print(comb_df)
     
     comb_df.to_csv(outname,index=False)
   



#Stacker(file_list, include_signal = False)
#Stacker(file_list)
