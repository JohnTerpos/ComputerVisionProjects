#import the required libraries
import cv2
import os
import numpy as np
import keras

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
INPUT_PATH = 'C:/Users/user/PycharmProjects/ExercisesComputerVision/Exercise2/CNNClassificationDataset/'

#define the format types you shall have
imgFormatType2WorkWithInput = ('PNG', 'png', 'jpeg', 'jpg')

#initialize the variables
X_train = []
ImageNamesListTrain = []
Y_train = []

X_val = []
ImageNamesListval = []
Y_val = []

X_test = []
ImageNamesListTest = []
Y_test = []

_, subCategoryDirectoriesInputSet, _ = next(os.walk(INPUT_PATH))

for TrainValidationOrTestIdx in range(0, subCategoryDirectoriesInputSet.__len__()):
    tmpTrainValOrTestPath = INPUT_PATH + subCategoryDirectoriesInputSet[TrainValidationOrTestIdx]
    _, SubcategorySet, _ = next(os.walk(tmpTrainValOrTestPath))
    for tmpCategoryIdx in range(0, SubcategorySet.__len__()):
        _, _, SubcategoryFiles = next(os.walk(tmpTrainValOrTestPath+'/'+ SubcategorySet[tmpCategoryIdx]))
        print(' . we are in directory:', subCategoryDirectoriesInputSet[TrainValidationOrTestIdx], '/', SubcategorySet[tmpCategoryIdx] )
        print(' .. there are', str(len(SubcategoryFiles)), 'available images')
        for ImageIdx in range(0, len(SubcategoryFiles)):
            # first check if we have the requested image format type
            if SubcategoryFiles[ImageIdx].endswith(imgFormatType2WorkWithInput):
                print(' . Working on input image', SubcategoryFiles[ImageIdx], '(',
                      str(ImageIdx + 1), '/', str(len(SubcategoryFiles)), ')')
                tmpFullImgName = INPUT_PATH + subCategoryDirectoriesInputSet[TrainValidationOrTestIdx] +\
                                 '/' + SubcategorySet[tmpCategoryIdx] +\
                                 '/' + SubcategoryFiles[ImageIdx]
                TmpImg = cv2.imread(tmpFullImgName)  # remember its height, width, chanels cv2.imread returns

                #just check that image is red correctly
                if TmpImg is not None:
                    #kill all small images
                    if (TmpImg.shape[0] < 50) | (TmpImg.shape[0] < 50):
                        print(' . Warning: too small image size for image:', SubcategoryFiles[ImageIdx], 'Ignoring it!')
                    else:
                        # check the image size and type remember it's according to CV2 format
                        WidthSizeCheck = TmpImg.shape[1] - IMG_WIDTH
                        HeightSizeCheck = TmpImg.shape[0] - IMG_HEIGHT
                        NumOfChannelsCheck = TmpImg.shape[2] - IMG_CHANNELS
                        if (WidthSizeCheck == 0) & (HeightSizeCheck == 0) & (NumOfChannelsCheck == 0):
                            print(' ... image was in correct shape')
                        else:
                            print(' ... reshaping image')
                            TmpImg = cv2.resize(TmpImg, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST) #remember it's CV2 here

                        #special check that resize has not caused any unwanted problem
                        if subCategoryDirectoriesInputSet[TrainValidationOrTestIdx] == 'train':
                            X_train.append(TmpImg)
                            Y_train.append(tmpCategoryIdx)
                            ImageNamesListTrain.append(SubcategoryFiles[ImageIdx])
                        elif subCategoryDirectoriesInputSet[TrainValidationOrTestIdx] == 'test':
                            X_test.append(TmpImg)
                            Y_test.append(tmpCategoryIdx)
                            ImageNamesListTest.append(SubcategoryFiles[ImageIdx])
                        else:
                            X_val.append(TmpImg)
                            Y_val.append(tmpCategoryIdx)
                            ImageNamesListval.append(SubcategoryFiles[ImageIdx])
                else:
                    print(' .. CV Warning: could not read image:', tmpFullImgName)

#For CNN, your input must be a 4-D tensor [batch_size, dimension(e.g. width), dimension (e.g. height), channels]
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_val = np.array(X_val)
Y_val = np.array(Y_val)

print('Done!')