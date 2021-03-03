import uproot
import pandas as pd
import ROOT
import numpy
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from array import array

from rootInterfaces import predOutputRoot
from rootInterfaces import dataLoader


BDTBGFile = "BDTBGPreds.csv"
BDTSigFile = "BDTSigPreds.csv"
MLPBGFile = "MLPBGPreds.csv"
MLPSigFile = "MLPSigPreds.csv"
kNNBGFile = "kNNBGPreds.csv"
kNNSigFile = "kNNSigPreds.csv"

def Stacker(BDTBGFile, BDTSigFile, MLPBGFile, MLPSigFile, kNNBGFile, kNNSigFile):
     """ Builds combined dataframe from output of several models.

     Parameters
     ----------
     BDTBGFile : str
        Name of file containing BDT predictions for background data.
     BDTSig : str
        Name of file containing BDT predictions for signal data.
     MLPBGFile : str
        Name of file containing MLP predictions for background data.
     MLPSig : str
        Name of file containing MLP predictions for signal data.
     kNNBGFile : str
        Name of file containing kNN predictions for background data.
     kNNSig : str
        Name of file containing kNN predictions for signal data.



     """
     BDT_BG_DF = pd.read_csv(BDTBGFile)
     BDT_Sig_DF = pd.read_csv(BDTSigFile)
     MLP_BG_DF = pd.read_csv(MLPBGFile)
     MLP_Sig_DF = pd.read_csv(MLPSigFile)
     kNN_BG_DF = pd.read_csv(kNNBGFile)
     kNN_Sig_DF = pd.read_csv(kNNSigFile)
     
     BDT_DF = pd.concat([BDT_BG_DF, BDT_Sig_DF]).reset_index(drop=True)
     MLP_DF = pd.concat([MLP_BG_DF, MLP_Sig_DF]).reset_index(drop=True)
     kNN_DF = pd.concat([kNN_BG_DF, kNN_Sig_DF]).reset_index(drop=True)
     
     comb_DF = pd.concat([BDT_DF,MLP_DF['MLPPred'], kNN_DF['kNNPred']], axis=1)
     
     print(comb_DF)
     
     comb_DF.to_csv("StackedPreds.csv",index=False)
   

def SigStacker(BDTSigFile, MLPSigFile):
     """ Builds combined dataframe of only signal predictions
     
     Parameters
     ----------
     BDTSig : str
        Name of file containing BDT predictions for signal data.
     MLPSig : str
        Name of file containing MLP predictions for signal data.

    
     """
     BDT_Sig_DF = pd.read_csv(BDTSigFile)
     MLP_Sig_DF = pd.read_csv(MLPSigFile)
     
     #comb_DF = pd.concat([BDT_Sig_DF,MLP_Sig_DF['MLPPred']], axis=1)
     comb_DF = pd.concat([BDT_DF,MLP_DF['MLPPred'], kNN_DF['kNNPred']], axis=1)
     
     comb_DF.to_csv("StackedSigPreds.csv",index=False)

def BGStacker(BDTBGFile, MLPBGFile):
     """ Builds combined dataframe of only background predictions.
     
     Parameters
     ----------
     BDTSig : str
        Name of file containing BDT predictions for signal data.
     MLPSig : str
        Name of file containing MLP predictions for signal data.

     """

     BDT_BG_DF = pd.read_csv(BDTBGFile)
     MLP_BG_DF = pd.read_csv(MLPBGFile)
     
     comb_DF = pd.concat([BDT_BG_DF,MLP_BG_DF['MLPPred']], axis=1)
     
     comb_DF.to_csv("StackedBGPreds.csv",index=False)


#Stacker(BDTBGFile, BDTSigFile,MLPBGFile,MLPSigFile, kNNBGFile, kNNSigFile)
SigStacker(BDTSigFile, MLPSigFile)
#BGStacker(BDTBGFile, MLPBGFile)
