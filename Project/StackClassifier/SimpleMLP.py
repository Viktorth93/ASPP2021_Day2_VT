import numpy as np
import pandas as pd
import uproot
import tensorflow as tf
import tensorflow_probability as tfp
from SimpleModel import SimpleModel


from rootInterfaces import predOutputRoot
from rootInterfaces import tfSetFromCSV
from rootInterfaces import tfEvalSetFromCSV



sigName = "Signal.root"
fileName = "StackedPreds.csv"
bgName = "BG.root"
dataName = "Data.root"
treeName = "PiPiee"
evalName = "Signal.root"
outputName = "Predicted.root"
saveFLAG = True
csvFileName = "StackedSigPreds.csv"


#features = ['pip_px', 'pip_py', 'pip_pz', 'pip_e', 'pim_px', 'pim_py', 
#        'pim_pz','pim_e', 'ep_px', 'ep_py', 'ep_pz', 'ep_e', 'em_px', 
#        'em_py', 'em_pz', 'em_e', 'g_px', 'g_py', 'g_pz', 'g_e'];

features = ['BDTPred', 'MLPPred', 'kNNPred']

def SimpleMLP(fileName, features, saveFlag):
    """ Train a neural network on stacked outputs of several other ML algorithms.
    Parameters
    ----------
    fileName : str
       Name of file containing stacked model outputs.
    features : list of str
       List of input features to use in the model.
    saveFlag : bool
       Flag that controls whether the model parameters should be saved or not.

    See Also
    --------
    ApplyModel

    """
    seed = 1121
    batchsize = 100
    trainportion = 0.75
    size = 150000

    (train_Iterator, val_Iterator, inputs, labels, handle)  = tfSetFromCSV(fileName, features, trainportion, batchsize)
  
    
    # Construct classification network
    learn_rate = 0.001
  
    num_features = 3
    
    SM = SimpleModel(num_features, seed, learn_rate)
    
    output_layer = SM.HandwrittenModel(inputs)
    
    #labels_distribution = tfd.Categorical(logits=output_layer)
    
    #neg_likelihood = -tf.reduce_mean(labels_distribution.prob(labels))
    #kl = sum(model.losses) / (trainportion*size)
    #elbo_loss = neg_likelihood + kl

    #tf.summary.scalar('Loss', elbo_loss)
    
    predictions = tf.argmax(output_layer, axis=1)


    #loss = lossFunc(predictions, labels)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_layer)
    

    #train = optimizer.minimize(elbo_loss)
    train = SM.optimizer.minimize(loss)

    print("Model built OK! Beginning Training")
        
    accuracy, accuracy_op = tf.metrics.accuracy(labels=labels, predictions = predictions)
    
    tf.summary.scalar('accuracy',accuracy)
    
    saver = tf.train.Saver()

    #val_pred = np.array([int((1-trainportion)*size)])

    # Train

    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer() )
    with tf.Session() as sess:
        sess.run(init)
        train_handle = sess.run(train_Iterator.string_handle())
        val_handle = sess.run(val_Iterator.string_handle())
        for i in range(10000):
            
            _ = sess.run([train, accuracy_op], feed_dict={handle: train_handle})

            
            
            if i % 100 == 0:
                loss_value, acc_value = sess.run([loss, accuracy], feed_dict={handle: train_handle})
                pred, lab = sess.run([predictions, labels], feed_dict={handle: train_handle})
                print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(i, loss_value, acc_value))
                #print(pred)
                #print(lab)
            
                
        pred_vals = sess.run(predictions, feed_dict={handle: val_handle})
        label_vals = sess.run(labels, feed_dict={handle: val_handle})
        val_acc = np.sum(np.equal(pred_vals,label_vals))/float(len(pred_vals))
        print("Validation Accuracy: {:.3f}".format(val_acc))
        
        #predOutputRoot(pred_vals, evalName, treeName, outputName)


                
        if saveFlag :        
            save_path = saver.save(sess, './model', global_step= 10000)
            print("Model saved in : %s" % save_path)
        
########################################################################
        
def ApplyModel(csvFileName, rootFileName, treeName, features):
    """ Apply trained model to data.

    Parameters
    ----------
    csvFileName : str
       Name of file containing stacked outputs of different ML models.
    rootFileName : str
       Name of file containing full data.
    treeName : str
       Name of data Tree where predictions should be written.
    features : list of str
       List of data features used in model.

    See Also
    --------
    SimpleMLP


    """
    (eval_Iterator, inputs, labels, handle) = tfEvalSetFromCSV(csvFileName, features)
    seed = 1121
        # Construct classification network
    learn_rate = 0.001
            
    num_features = 2
    
    SM = SimpleModel(num_features, seed, learn_rate)
    
    outName = "Applied.root"
    
    #output_layer = SM.HandwrittenModel(inputs)
    
    #predictions = tf.argmax(output_layer, axis=1)
       
    
    
    #accuracy, accuracy_op = tf.metrics.accuracy(labels=labels, predictions = predictions)
        
        
    #saver = tf.train.Saver()
        
    
    
    predictions = SM.LoadHandwrittenModel(handle, eval_Iterator, inputs)
    
    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer() )
    with tf.Session() as sess:
        sess.run(init)
        eval_handle = sess.run(eval_Iterator.string_handle())
        label_vals = sess.run(labels, feed_dict={handle: eval_handle})
       
    #pred_DF.to_csv("MLPPreds.csv",index=False)
     
    predOutputRoot(predictions, rootFileName, treeName, outName)
         
#SimpleMLP(fileName, features, saveFLAG)
ApplyModel(csvFileName,sigName, treeName, features)
