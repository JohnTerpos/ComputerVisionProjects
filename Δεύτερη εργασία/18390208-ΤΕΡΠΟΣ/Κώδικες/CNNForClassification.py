#import the required libraries
import keras
from keras import backend as KerasBackend
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

# get the data
from DataLoadClassif import X_train, Y_train, X_test, Y_test, X_val, Y_val, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS

#define functions to calculate precision, recall and f1 performance scores
def PrecisionMetric(y_true, y_pred):
    true_positives = KerasBackend.sum(KerasBackend.round(KerasBackend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = KerasBackend.sum(KerasBackend.round(KerasBackend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + KerasBackend.epsilon())
    return precision

def RecallMetric(y_true, y_pred):
    true_positives = KerasBackend.sum(KerasBackend.round(KerasBackend.clip(y_true * y_pred, 0, 1)))
    possible_positives = KerasBackend.sum(KerasBackend.round(KerasBackend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + KerasBackend.epsilon())
    return recall

def F1Metric(y_true, y_pred):
    precision = PrecisionMetric(y_true, y_pred)
    recall = RecallMetric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + KerasBackend.epsilon()))

if __name__ == '__main__':

    #define some CNN parameters
    batch_size = 100
    num_classes = np.unique(Y_train).__len__()
    epochs = 15
    baseNumOfFilters = 16

    #the data, split between train and test sets
    #(X_train, Y_train), (x_test, y_test) = mnist.load_data()

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices

    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    Y_val = keras.utils.to_categorical(Y_val, num_classes)

    # here we define and load the model

    inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = keras.layers.Lambda(lambda x: x / 255)(inputs) #normalize the input
    conv1 = keras.layers.Conv2D(filters=baseNumOfFilters, kernel_size=(13, 13))(s)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(filters=baseNumOfFilters * 2, kernel_size=(7, 7))(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(filters=baseNumOfFilters * 4, kernel_size=(3, 3))(pool2)
    drop3 = keras.layers.Dropout(0.25)(conv3)
    flat1 = keras.layers.Flatten()(drop3)
    dense1 = keras.layers.Dense(128, activation='relu')(flat1)
    outputs = keras.layers.Dense(Y_train.shape[1], activation='softmax')(dense1)

    CNNmodel = keras.Model(inputs=[inputs], outputs=[outputs])
    CNNmodel.compile(optimizer='sgd', loss=keras.losses.categorical_crossentropy, metrics=['accuracy',PrecisionMetric, RecallMetric,F1Metric])
    # print model summary
    CNNmodel.summary()

    # fit model parameters, given a set of training data
    callbacksOptions = [
        keras.callbacks.EarlyStopping(patience=15, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
        keras.callbacks.ModelCheckpoint('tmpCNN.h5', verbose=1, save_best_only=True, save_weights_only=True)]

    CNNmodel.fit(X_train, Y_train, batch_size=batch_size, shuffle=True, epochs=epochs, verbose=1,
          callbacks=callbacksOptions, validation_data=(X_val, Y_val))

    # calculate some common performance scores
    lossTrain,accuracyTrain,precisionTrain,recallTrain,f1Train=CNNmodel.evaluate(X_train, Y_train, verbose=0)
    lossTest,accuracyTest,precisionTest,recallTest,f1Test = CNNmodel.evaluate(X_test, Y_test, verbose=0)

    #print results
    print('Train loss: ' +str(lossTrain))
    print('Test loss: ' +str(lossTest))
    print('Train accuracy: ' +str(accuracyTrain))
    print('Test accuracy: ' +str(accuracyTest))
    print('Train precision: ' +str(precisionTrain))
    print('Test precision: ' +str(precisionTest))
    print('Train recall: ' +str(recallTrain))
    print('Test recall: ' +str(recallTest))
    print('Train f1 score: ' +str(f1Train))
    print('Test f1 score: ' +str(f1Test))

    # saving the trained model
    model_name = 'finalCNN.h5'
    CNNmodel.save(model_name)