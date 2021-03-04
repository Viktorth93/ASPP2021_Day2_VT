import numpy as np
import pandas as pd
import uproot
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
from SimpleModel import SimpleModel


from RootInterfaces import pred_output_root
from RootInterfaces import tf_set_from_csv
from RootInterfaces import tf_evalset_from_csv



signame = "Signal.root"
filename = "StackedPreds.csv"
bgName = "BG.root"
dataName = "Data.root"
treename = "PiPiee"
evalName = "Signal.root"
outputName = "Predicted.root"
saveflag = True
csvfilename = "StackedSigPreds.csv"



features = ['BDTPred', 'MLPPred', 'kNNPred']

def train_model(filename, features, saveflag):
    """ Train a neural network on stacked outputs of several other ML algorithms.

    Parameters
    ----------
    filename : str
       Name of file containing stacked model outputs.
    features : list of str
       List of input features to use in the model.
    saveflag : bool
       Flag that controls whether the model parameters should be saved or not.

    See Also
    --------
    apply_model


    """
    seed = 1121
    batchsize = 100
    trainportion = 0.75
    size = 150000

    (train_Iterator, val_Iterator, inputs, labels, handle)  = tf_set_from_csv(filename, features, trainportion, batchsize)
  
    
    # Construct classification network
    learn_rate = 0.001
  
    num_features = 3
    
    SM = SimpleModel(num_features, seed, learn_rate)
    
    output_layer = SM.handwritten_model(inputs)
    
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
        
        #pred_output_root(pred_vals, evalName, treename, outputName)


                
        if saveflag :        
            save_path = saver.save(sess, './model', global_step= 10000)
            print("Model saved in : %s" % save_path)
        
########################################################################
        
def apply_model(csv_in_filename, features, save_to_csv=True,save_to_root=False, csv_out_filename="out.csv", rootfilename="data.root", treename="Tree"):
    """ Apply trained model to data.

    Parameters
    ----------
    csv_in_filename : str
       Name of file containing input.
    features : list of str
       List of data features used in model.
    save_to_csv : bool
       True if output is to be written to csv file. Default is True.
    csv_out_filename : str
       Name of file where csv output is to be written. Default is out.csv. Only relevant if save_to_cvs is True.
    save_to_root : bool
       True if output is to be written to root file. Default is False.
    rootfilename : str
       Name of file containing full data where predictions are to be appended. Default is data.root. Only relevant if save_to_root is True.
    treename : str
       Name of data Tree where predictions should be written. Default is Tree. Only relevant if save_to_root is True.

    See Also
    --------
    train_model


    """
    (eval_Iterator, inputs, labels, handle) = tf_evalset_from_csv(csvfilename, features)
    seed = 1121
        # Construct classification network
    learn_rate = 0.001
            
    num_features = 2
    
    SM = SimpleModel(num_features, seed, learn_rate)
    
    outname = "Applied.root"
    
    #output_layer = SM.handwritten_model(inputs)
    
    #predictions = tf.argmax(output_layer, axis=1)
       
    
    
    #accuracy, accuracy_op = tf.metrics.accuracy(labels=labels, predictions = predictions)
        
        
    #saver = tf.train.Saver()
        
    
    
    predictions = SM.load_handwritten_model(handle, eval_Iterator, inputs)
    
    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer() )
    with tf.Session() as sess:
        sess.run(init)
        eval_handle = sess.run(eval_Iterator.string_handle())
        label_vals = sess.run(labels, feed_dict={handle: eval_handle})
       
    if save_to_csv :
        pred_DF.to_csv(csvfilename,index=False)
    if save_to_root : 
        pred_output_root(predictions, rootfilename, treename, outname)
         
#train_model(filename, features, saveflag)
#apply_model(csvfilename, features, csvfilename="StackedModelPreds.csv")
