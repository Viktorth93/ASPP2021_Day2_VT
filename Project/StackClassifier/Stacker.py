import uproot
import pandas as pd
#import ROOT
import numpy
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from array import array

#from rootInterfaces import predOutputRoot
#from rootInterfaces import dataLoader


BDTBGFile = "BDTBGPreds.csv"
BDTSigFile = "BDTSigPreds.csv"
MLPBGFile = "MLPBGPreds.csv"
MLPSigFile = "MLPSigPreds.csv"
kNNBGFile = "kNNBGPreds.csv"
kNNSigFile = "kNNSigPreds.csv"

fileList = ["BDTPreds.csv", "MLPPreds.csv", "kNNPreds.csv"]

#def Stacker(BDTBGFile, BDTSigFile, MLPBGFile, MLPSigFile, kNNBGFile, kNNSigFile):
def Stacker(fileList, includeSignal=True, includeBG=True, outName="StackedPreds.csv"):
     """ Builds combined dataframe from output of several models.

     Parameters
     ----------
     fileList : list of str
        List of names of files containing input data.
     includeSignal : bool
        Controls whether signal events are included in output dataframe. Default is true/yes.
     includeBG : bool
        Controls whether bg events are included in output dataframe. Default is true/yes.
     outName : str
        Name of output file. Default is StackedPreds.csv



     """
     comb_DF = pd.DataFrame()
     for f in fileList:
        temp_DF = pd.read_csv(f)
        comb_DF = pd.concat([comb_DF, temp_DF], axis=1)

     comb_DF = comb_DF.loc[:,~comb_DF.columns.duplicated()]
     if (not includeSignal): 
         comb_DF = comb_DF[comb_DF.label != 1].reset_index()
     if (not includeBG): 
         comb_DF = comb_DF[comb_DF.label != 0].reset_index()

     print(comb_DF)
     
     comb_DF.to_csv("StackedPreds.csv",index=False)
   



Stacker(fileList, includeSignal = False)
