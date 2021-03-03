import uproot
import pandas as pd 
import ROOT
from array import array
import tensorflow as tf
import numpy as np

def dataLoader(sigName, bgName, dataName, treeName, features, sig_size, bg_size):
   """ Load data from root trees into pandas dataframes.

   Reads root trees (a common data format in high energy and nuclear physics) and turns them into pandas dataframes for ease of use in machinelearning.

   Parameters
   ----------
   sigName : str
      Name of file containing signal data
   bgName : str
      Name of file containing background data
   dataName : str
      Name of file containing data to be evaluated by the model.
   treeName : str
      Name of tree within 

   Returns
   -------
   trainDF : pandas.DataFrame
      Dataframe containing training data set.
   valDF : pandas.DataFrame
      Dataframe containing validation data set.
   dataDF : pandas.DataFrame
      Dataframe containing data set to be classified.


   """
   sigFile = uproot.open(sigName)
   bgFile = uproot.open(bgName)
   dataFile = uproot.open(dataName)

   sigTree = sigFile[treeName]
   bgTree = bgFile[treeName]
   dataTree = dataFile[treeName]

   sigDF = sigTree.pd.df(features)
   bgDF = bgTree.pd.df(features)
   dataDF = dataTree.pd.df(features)

   # Label MC
   sigDF['label'] = 1
   bgDF['label'] = 0
   dataDF['label'] = -1. # necessary?

   # Create Training set
   sigDF_train = sigDF[0:int(sig_size)]
   bgDF_train = bgDF[0:int(bg_size)]

   trainDF = pd.concat([sigDF_train, bgDF_train])
   trainDF = trainDF.sample(frac=1).reset_index(drop=True)

   # Create Validation set 
   sigDF_validation = sigDF[int(sig_size):int(2*sig_size)]
   bgDF_validation = bgDF[int(bg_size):int(2*bg_size)]

   valDF = pd.concat([sigDF_validation, bgDF_validation])
   valDF = trainDF.sample(frac=1).reset_index(drop=True)

   return trainDF, valDF, dataDF



def predOutputRoot(preds, inName, treeName, outName):

   dataFile_local = ROOT.TFile.Open(inName)
   datTree = dataFile_local.Get(treeName)

   outFile = ROOT.TFile.Open(outName, "RECREATE")
   new_Tree = datTree.CloneTree()

   predictions = array( 'f', [0])

   outTree = outFile.Get(treeName)

   pred_branch = outTree.Branch("Prediction", predictions, 'Predictions/F')

   for i in range(len(preds)):
      predictions[0] = preds[i]
      pred_branch.Fill()

   outFile.Write()
   outFile.Close()

   

def featureOutputRoot(ep_px, ep_py, ep_pz, ep_e, em_px, em_py,em_pz, em_e, m2e, inFile, outFile, treeName):
    dataFile_in = ROOT.TFile.Open(inFile)
    dataTree_in = dataFile_in.Get(treeName)

    dataFile_out = ROOT.TFile.Open(outFile,"RECREATE")
    dataTree_out = dataTree_in.CloneTree()

    eppx = array('f', [0])
    eppy = array('f', [0])
    eppz = array('f', [0])
    epe = array('f', [0])
    empx = array('f', [0])
    empy = array('f', [0])
    empz = array('f', [0])
    eme = array('f', [0])
    mee = array('f', [0])

    eppx_branch = dataTree_out.Branch("ep_px_decoded", eppx, 'ep_px_decoded/F')
    eppy_branch = dataTree_out.Branch("ep_py_decoded", eppy, 'ep_py_decoded/F')
    eppz_branch = dataTree_out.Branch("ep_pz_decoded", eppz, 'ep_pz_decoded/F')
    epe_branch = dataTree_out.Branch("ep_e_decoded", epe, 'ep_e_decoded/F')
    empx_branch = dataTree_out.Branch("em_px_decoded", empx, 'em_px_decoded/F')
    empy_branch = dataTree_out.Branch("em_py_decoded", empy, 'em_py_decoded/F')
    empz_branch = dataTree_out.Branch("em_pz_deodcoded", empz, 'em_pz_decoded/F')
    eme_branch = dataTree_out.Branch("em_e_decoded", eme, 'em_e_decoded/F')
    mee_branch = dataTree_out.Branch("m2e_decoded", mee, 'm2e_decoded/F')

    for i in range(len(m2e)):
        eppx = ep_px[0]
        eppy = ep_py[0]
        eppz = ep_pz[0]
        epe = ep_e[0]
        empx = em_px[0]
        empy = em_py[0]
        empz = em_pz[0]
        eme = em_e[0]
        mee = m2e[0]
        eppx_branch.Fill()
        eppy_branch.Fill()
        eppz_branch.Fill()
        epe_branch.Fill()
        empx_branch.Fill()
        empy_branch.Fill()
        empz_branch.Fill()
        eme_branch.Fill()
        mee_branch.Fill()

    dataFile_out.Write()
    dataFile_out.Close()

    
def tfDataSetMaker(sigFileName, bgFileName, treeName, features,size, batch_size, training_portion):
   # Load data from root trees into suitable format for sklearn analysis
   
   sigFile = uproot.open(sigFileName)
   bgFile = uproot.open(sigFileName)

   sigTree = sigFile[treeName]
   bgTree = bgFile[treeName]
   
   # Create admixture of signal and BG here

   sigDF = sigTree.pd.df(features)
   bgDF = bgTree.pd.df(features)
   
   sigDF['label'] = 1
   bgDF['label'] = 0
   
   inDF = pd.concat([sigDF, bgDF])
   inDF = inDF.sample(frac=1).reset_index(drop=True)
   
   trainDF = inDF[0:int(training_portion*size)]
   valDF = inDF[int(training_portion*size):size]

   # Create Training set
   inDF = inDF[0:int(size)]
   pip = np.array([inDF['pip_e'].values,inDF['pip_px'].values, inDF['pip_py'].values, inDF['pip_pz'].values])
   pim = np.array([inDF['pim_e'].values,inDF['pim_px'].values, inDF['pim_py'].values, inDF['pim_pz'].values])
   ep = np.array([inDF['ep_e'].values,inDF['ep_px'].values, inDF['ep_py'].values, inDF['ep_pz'].values])
   em = np.array([inDF['em_e'].values,inDF['em_px'].values, inDF['em_py'].values, inDF['em_pz'].values])
   g = np.array([inDF['g_e'].values,inDF['g_px'].values, inDF['g_py'].values, inDF['g_pz'].values])

   particles = np.array([ pip, pim, ep, em, g])
   #data_placeholder = tf.placeholder(particles.dtype, particles.shape)


   particles_reshaped =  np.empty([size, 5, 4])

   for i in range(size):
       for j in range (5):
           for k in range (4):
               particles_reshaped[i,j,k] = particles [j,k,i]
   
   train_particles = particles_reshaped[0:int(training_portion*size), :,:]
   val_particles = particles_reshaped[int(training_portion*size):size,:,:]


   train_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(train_particles,tf.float32),tf.cast(trainDF['label'], tf.int32))))
   train_Batches = (train_dataset.take(size).repeat().batch(batch_size)) 
   train_Iterator = train_Batches.make_one_shot_iterator()
   
   val_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(val_particles,tf.float32),tf.cast(valDF['label'], tf.float32))))
   #val_Batches = (val_dataset.take(size).repeat().batch(len(valDF['label'].values))) 
   val_Batches = (val_dataset.take(size).repeat().batch(batch_size)) 
   val_Iterator = val_Batches.make_one_shot_iterator()

   handle = tf.placeholder(tf.string, shape=[])
   feedable_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_Batches.output_shapes)
   inputs, labels = feedable_iterator.get_next()

   return train_Iterator, val_Iterator, inputs, labels, handle


def tfTrainDataSetMaker(sigFileName, bgFileName, treeName, features,size, batch_size, training_portion):
   # Load data from root trees into suitable format for sklearn analysis
   
   sigFile = uproot.open(sigFileName)
   bgFile = uproot.open(bgFileName)

   sigTree = sigFile[treeName]
   bgTree = bgFile[treeName]
   
   # Create admixture of signal and BG here

   sigDF = sigTree.pd.df(features)
   bgDF = bgTree.pd.df(features)
   
   sigDF['label'] = 1
   bgDF['label'] = 0
   
   inDF = pd.concat([sigDF, bgDF])
   inDF = inDF.sample(frac=1).reset_index(drop=True)
   
   trainDF = inDF[0:int(training_portion*size)]
   valDF = inDF[int(training_portion*size):size]

   
   train_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(trainDF[features].values,tf.float32),tf.cast(trainDF['label'].values, tf.int32))))
   train_Batches = (train_dataset.take(size).repeat().batch(batch_size))
   #train_Batches = (train_dataset.shuffle(10000, reshuffle_each_iteration=True).repeat().batch(batch_size))  
   train_Iterator = train_Batches.make_one_shot_iterator()
   
   val_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(valDF[features].values,tf.float32),tf.cast(valDF['label'], tf.int32))))
   val_Batches = (val_dataset.take(size).repeat().batch(len(valDF['label'].values))) 
   #val_Batches = (val_dataset.take(size).repeat().batch(batch_size)) 
   val_Iterator = val_Batches.make_one_shot_iterator()

   handle = tf.placeholder(tf.string, shape=[])
   feedable_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_Batches.output_shapes)
   inputs, labels = feedable_iterator.get_next()

   return train_Iterator, val_Iterator, inputs, labels, handle
   
########################################################################
   
def tfEvalDataSetMaker(fileName, treeName, features, datalabel=-1.):
   # Load data from root trees into suitable format for sklearn analysis
   
   evalFile = uproot.open(fileName)


   evalTree = evalFile[treeName]

   
   # Create admixture of signal and BG here

   evalDF = evalTree.pd.df(features)
   evalDF['label'] = datalabel
   size = len(evalDF['label'].values)
   
   eval_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(evalDF[features].values,tf.float32),tf.cast(evalDF['label'].values, tf.int32))))
   eval_Batches = (eval_dataset.take(size).repeat().batch(size))
   
   eval_Iterator = eval_Batches.make_one_shot_iterator()
   

   handle = tf.placeholder(tf.string, shape=[])
   feedable_iterator = tf.data.Iterator.from_string_handle(handle, eval_dataset.output_types, eval_Batches.output_shapes)
   inputs, labels = feedable_iterator.get_next()

   return eval_Iterator, inputs, labels, handle




def tfSetFromCSV(fileName, features, trainportion, batch_size):
   # Load data from root trees into suitable format for sklearn analysis

   dataset = pd.read_csv(fileName)   
   dataset = dataset.sample(frac=1).reset_index(drop=True)

   # Create Training set
   
   trainDF = dataset[0:int(trainportion*len(dataset))]
   valDF = dataset[int(trainportion*len(dataset)):len(dataset)]


   train_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(trainDF[features].values,tf.float32),tf.cast(trainDF['label'].values, tf.int32))))
   train_Batches = (train_dataset.take(len(trainDF)).repeat().batch(batch_size))
   train_Iterator = train_Batches.make_one_shot_iterator()
   
   val_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(valDF[features].values,tf.float32),tf.cast(valDF['label'], tf.int32))))
   val_Batches = (val_dataset.take(len(valDF)).repeat().batch(len(valDF['label'].values))) 
   val_Iterator = val_Batches.make_one_shot_iterator()

   handle = tf.placeholder(tf.string, shape=[])
   feedable_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_Batches.output_shapes)
   inputs, labels = feedable_iterator.get_next()

   return train_Iterator, val_Iterator, inputs, labels, handle

def tfEvalSetFromCSV(fileName, features):
   # Load data from root trees into suitable format for sklearn analysis

   dataset = pd.read_csv(fileName)   

   # Create Eval set
   eval_dataset = (tf.data.Dataset.from_tensor_slices((tf.cast(dataset[features].values,tf.float32),tf.cast(dataset['label'], tf.int32))))
   eval_Batches = (eval_dataset.take(len(dataset)).repeat().batch(len(dataset))) 
   eval_Iterator = eval_Batches.make_one_shot_iterator()

   handle = tf.placeholder(tf.string, shape=[])
   feedable_iterator = tf.data.Iterator.from_string_handle(handle, eval_dataset.output_types, eval_Batches.output_shapes)
   inputs, labels = feedable_iterator.get_next()

   return eval_Iterator, inputs, labels, handle


