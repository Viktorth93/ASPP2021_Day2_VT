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

   BDT_Sig_DF = pd.read_csv(BDTSigFile)
   MLP_Sig_DF = pd.read_csv(MLPSigFile)
 
   #comb_DF = pd.concat([BDT_Sig_DF,MLP_Sig_DF['MLPPred']], axis=1)
   comb_DF = pd.concat([BDT_DF,MLP_DF['MLPPred'], kNN_DF['kNNPred']], axis=1)

   comb_DF.to_csv("StackedSigPreds.csv",index=False)

def BGStacker(BDTBGFile, MLPBGFile):

   BDT_BG_DF = pd.read_csv(BDTBGFile)
   MLP_BG_DF = pd.read_csv(MLPBGFile)
 
   comb_DF = pd.concat([BDT_BG_DF,MLP_BG_DF['MLPPred']], axis=1)

   comb_DF.to_csv("StackedBGPreds.csv",index=False)


#Stacker(BDTBGFile, BDTSigFile,MLPBGFile,MLPSigFile, kNNBGFile, kNNSigFile)
SigStacker(BDTSigFile, MLPSigFile)
#BGStacker(BDTBGFile, MLPBGFile)
