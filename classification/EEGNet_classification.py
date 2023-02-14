'''
 MPDB: A multimodal physiological dataset for driving behaviour analysis

 This code is used for driving behavior classification with EEGNet. 
 
 This program requires tenorflow 2.X (2.10 have been verified as working)

 Date: 2023.2.14
'''

# Import necessary python libraries
import numpy as np
import keras.models
import tensorflow
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from mne.io import read_epochs_eeglab
from sklearn.model_selection import train_test_split

def EEGNet(nb_classes, Chans = 64, Samples = 128,
    dropoutRate = 0.5, kernLength = 64, F1 = 8,
    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    EEGNet model, published by Vernon et al in 2018.
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.

    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False,
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)

    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)

    flatten      = Flatten(name = 'flatten')(block2)

    dense        = Dense(nb_classes, name = 'dense',
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)

    return Model(inputs=input1, outputs=softmax)

def get_label(epochs):
    '''
    This function uses the inverse dictionary to count the labels of data 
    samples after using mne to read the dataset files.
    '''
    true_label = []
    ivd = {v: k for k, v in epochs.event_id.items()}
    for each in epochs.events:
        if ivd[each[2]] in ['139','141','145']:#brake
            true_label.append(0)
        elif ivd[each[2]] in ['125','127']:#turn
            true_label.append(1)
        elif ivd[each[2]] in ['129','131']:#change
            true_label.append(2)
        elif ivd[each[2]] in ['137','143']:#throttle
            true_label.append(3)
        elif ivd[each[2]] in ['133']:#stable
            true_label.append(4)
    return true_label

def find_class_index(label,c):
    '''
    This function finds all subscripts with category c from the given label array.
    '''
    y = []
    for i in range(len(label)):
        if label[i] == c:
            y.append(i)
    return y

def plot_result(model,history,X_test,y_test):
    from sklearn import metrics
    from sklearn.metrics import ConfusionMatrixDisplay
    scores = model.evaluate(X_test, np.array(y_test),verbose = 1)
    print(model.metrics_names)
    print(scores)
    print('[INFO] The classification result is shown below!')
    y_test_pred = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred,axis = 1)
    ov_acc = metrics.accuracy_score(y_test,y_test_pred)
    ov_acc1 = metrics.balanced_accuracy_score(y_test,y_test_pred)
    print("overall accuracy: %f"%(ov_acc))
    print("balanced accuracy: %f"%(ov_acc1))
    print("===========================================")
    acc_for_each_class = metrics.precision_score(y_test,y_test_pred,average=None)
    print("acc_for_each_class:\n",acc_for_each_class)
    print("===========================================")
    avg_acc = np.mean(acc_for_each_class)
    print("average accuracy:%f"%(avg_acc))

    names=('brake','turning','change','throttle','stable')
    ConfusionMatrixDisplay.from_predictions(y_test,y_test_pred,display_labels=names,normalize = 'true')
    plt.savefig(time.strftime("%Y%m%d%H%M%S",time.localtime())+'_cm.eps')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.figure()
    plt.plot(epochs,acc, 'b', label='Training accuracy',)
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.savefig('Classificaion'+time.strftime("%Y%m%d%H%M%S",time.localtime())+'accuracy.eps')

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('Classificaion'+time.strftime("%Y%m%d%H%M%S",time.localtime())+'loss.eps')

def main():
    import tensorflow as tf
    from keras import optimizers
    from sklearn.model_selection import train_test_split

    DATA = np.zeros(shape = (5234,64,2000))
    y = []
    label = ['brake','turn','change','throttle','stable']
    sfreq = 1000
    pointer = 0
    for i in range(1,31):
        EEG_temp = read_epochs_eeglab('EEG_'+ str(i) +'.set',
                           eog=(),
                           uint16_codec=None,
                           verbose=None)
        EMG_temp = read_epochs_eeglab('EMG_'+ str(i) +'.set',
                           eog=(),
                           uint16_codec=None,
                           verbose=None)
        GSR_temp = read_epochs_eeglab('GSR_'+ str(i) +'.set',
                           eog=(),
                           uint16_codec=None,
                           verbose=None)
        label_EEG = get_label(EEG_temp)
        label_EMG = get_label(EMG_temp)
        label_GSR = get_label(GSR_temp)

        EEG = EEG_temp.get_data()
        EMG = EMG_temp.get_data()
        GSR = GSR_temp.get_data()

        EEG_index = []
        EMG_index = []
        GSR_index = []

        for i in range(5):
            EEG_index.append(find_class_index(label_EEG,i))
            EMG_index.append(find_class_index(label_EMG,i))
            GSR_index.append(find_class_index(label_GSR,i))

        
        # Build multimodal array according to the number of EEG
        for i in range(5):
            for j in range(min(len(EEG_index[i]),len(EMG_index[i]),len(GSR_index[i]))):
                sample = np.concatenate([EEG[EEG_index[i][j],:,:],EMG[EMG_index[i][j],:,:],GSR[GSR_index[i][j],:,:]],axis = 0)
                DATA[pointer,:,:] = sample
                pointer += 1
                y.append(i)

    print(DATA.shape)
    X_train, X_test, y_train, y_test = train_test_split(DATA, y, test_size=.2,random_state = None)
    y_train = np.array(y_train)

    print("[INFO] An overview of the data sample is as follows:")

    count_train = [0,0,0,0,0]
    for i in y_train:
        if i == 0:
            count_train[0] +=1
        elif i == 1:
            count_train[1] +=1
        elif i == 2:
            count_train[2] +=1
        elif i == 3:
            count_train[3] +=1
        elif i == 4:
            count_train[4] +=1
    print("The distribution of training data is ",count_train)
    epochs = []

    # Part II, feature extraction and classification
    samples = 2*sfreq# Number of data frame samples
    
    model = EEGNet(5, Chans = 59+5, Samples = samples, 
            dropoutRate = 0, kernLength = 512, F1 = 32,  
            D = 2, F2 = 64, norm_rate = 0.2, dropoutType = 'Dropout')
    
    lr_schedule = optimizers.schedules.learning_rate_schedule.ExponentialDecay(initial_learning_rate=0.001,
                                                                decay_steps=10000,
                                                                decay_rate=0.9)
    opt = optimizers.adam_v2.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

    print('[INFO] The network model is shown below.')
    model.summary()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',min_delta = 0,patience = 2500,verbose = 0,mode = 'auto',baseline = None,restore_best_weights = False)
    print('[INFO] Model training begins!')
    history = model.fit(X_train,y_train,epochs = 15000,shuffle = True,
                batch_size = 128,
                verbose = 2, 
                validation_data = tuple([np.array(X_test),np.array(y_test)]), 
                class_weight = {0:1.30,1:1.89,2:1,3:1.40,4:2.0},
                callbacks = [early_stopping])
    model.save(time.strftime("%Y%m%d%H%M%S",time.localtime())+'.h5')
    plot_result(model,history,X_test,y_test)
    print('[INFO] Finished!')

# program entrance
if __name__ == "__main__":
    main()




