#import the required libraries
import cv2
import os
import numpy as np
import keras


# Set some parameters, these also define the FCN input layer size latter
IMG_WIDTH = 240
IMG_HEIGHT = 240
IMG_CHANNELS = 3
INPUT_PATH = 'C:/Users/user/PycharmProjects/ExercisesComputerVision/Exercise2/Semantic_segmentation_dataset/input/'
OUTPUT_PATH = 'C:/Users/user/PycharmProjects/ExercisesComputerVision/Exercise2/Semantic_segmentation_dataset/output/'


#IMPORTANT: this script runs on the following file routine
# .. input folder (or output folder)
#     + .. train
#         + .. image 1
#         + .. image 2
#         + ..
#     + .. test
#         + .. image 1
#         + .. image 2
#         + ..
#     + .. validation
#         + .. image 1
#         + .. image 2
#         + ..

#IMPORTANT: images in input and output folders must have the same name. if NOT go to line 95, 96 and change the code


#define the format types you shall have
imgFormatType2WorkWithInput = ('png', 'PNG')
imgFormatType2WorkWithOutput = ('png', 'PNG') #define the possible types you work with

#Eliminate a predefined number of pixels on the edge, no need for that here!
edgePixelsToEliminate = 0

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

NotUsedImagesCounter = 0

for TrainValidationOrTestIdx in range(0, subCategoryDirectoriesInputSet.__len__()):
    tmpTrainValidationOrTestPath = INPUT_PATH + subCategoryDirectoriesInputSet[TrainValidationOrTestIdx]
    _, _, SubcategoryFiles = next(os.walk(tmpTrainValidationOrTestPath))
    print(' . we are in directory:', subCategoryDirectoriesInputSet[TrainValidationOrTestIdx])
    print(' .. there are', str(len(SubcategoryFiles)), 'available images')
    for ImageIdx in range(0, len(SubcategoryFiles)):
        # first check if we have the requested image format type
        if SubcategoryFiles[ImageIdx].endswith(imgFormatType2WorkWithInput):
            print(' . Working on input image', SubcategoryFiles[ImageIdx], '(',
                  str(ImageIdx + 1), '/', str(len(SubcategoryFiles)), ')')
            tmpFullImgName = INPUT_PATH + subCategoryDirectoriesInputSet[TrainValidationOrTestIdx] +\
                             '/' + SubcategoryFiles[ImageIdx]
            TmpImg = cv2.imread(tmpFullImgName)  # remember its height, width, chanels cv2.imread returns

            WidthSizeCheck = TmpImg.shape[1] - IMG_WIDTH
            HeightSizeCheck = TmpImg.shape[0] - IMG_HEIGHT
            NumOfChannelsCheck = TmpImg.shape[2] - IMG_CHANNELS
            if (WidthSizeCheck == 0) & (HeightSizeCheck == 0) & (NumOfChannelsCheck == 0):
                print(' ... image was in correct shape')
            else:
                print(' ... reshaping image')
                TmpImg = cv2.resize(TmpImg, (IMG_WIDTH, IMG_HEIGHT)) #remember it's CV2 here

            print(' . Check if we have the corresponding mask')
            tmpMaskName = SubcategoryFiles[ImageIdx][:-4]
            #find the specific image, including the file extension
            for FileExtensionCheckIdx in range(0, len(imgFormatType2WorkWithOutput)):
                tmpFullMaskName = OUTPUT_PATH + subCategoryDirectoriesInputSet[TrainValidationOrTestIdx] + '/' + \
                                  tmpMaskName + '.' + imgFormatType2WorkWithOutput[FileExtensionCheckIdx]
                if tmpFullMaskName is not None:
                    break

            TmpMask = cv2.imread(tmpFullMaskName, 0)  # remember its height, width, channels cv2.imread returns
            if TmpMask is None:
                print(' .. unable to load the corresponding mask')
            else:
                print(' .. Corresponding mask successfully loaded. Checking the size and the instances.')

                WidthSizeCheck = TmpMask.shape[1] - IMG_WIDTH
                HeightSizeCheck = TmpMask.shape[0] - IMG_HEIGHT
                if (WidthSizeCheck == 0) & (HeightSizeCheck == 0): # & (NumOfChannelsCheck == 0):
                    print(' ... mask has the correct size')
                else:
                    print(' ... resizing mask')
                    TmpMask = cv2.resize(TmpMask, (IMG_WIDTH, IMG_HEIGHT)) #remember it's CV2 here


                tmpUniqueVals, tmpAppearanceFrequence = np.unique(TmpMask, return_counts=True)
                print(' ... we have the following values:', tmpUniqueVals, 'with appearance frequency:', tmpAppearanceFrequence)
                TmpMask[TmpMask < 1] = 0
                TmpMask[TmpMask >= 1] = 1

                #convert mask labels to binarize vectors. Here we know that we have two classes
                TmpMask = keras.utils.to_categorical (TmpMask, 2)

            #finaly update train or test sets
            if sum(sum((TmpMask >= 1).astype(int)))[1] > 0:
                if subCategoryDirectoriesInputSet[TrainValidationOrTestIdx] == 'train':
                    X_train.append(TmpImg)
                    Y_train.append(TmpMask)
                    ImageNamesListTrain.append(tmpMaskName)
                elif subCategoryDirectoriesInputSet[TrainValidationOrTestIdx] == 'test':
                    X_test.append(TmpImg)
                    Y_test.append(TmpMask)
                    ImageNamesListTest.append(tmpMaskName)
                else:
                    X_val.append(TmpImg)
                    Y_val.append(TmpMask)
                    ImageNamesListval.append(tmpMaskName)
            else:
                print(' .. not using specific image!')
                NotUsedImagesCounter = NotUsedImagesCounter + 1


print(' .. warning: Number of images without satelite cases that had to be excluded were:', NotUsedImagesCounter)
#For CNN, your input must be a 4-D tensor [batch_size, dimension(e.g. width), dimension (e.g. height), channels]
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)


X_val = np.array(X_val)
Y_val = np.array(Y_val)

print('All done! Datasets creation completed.')