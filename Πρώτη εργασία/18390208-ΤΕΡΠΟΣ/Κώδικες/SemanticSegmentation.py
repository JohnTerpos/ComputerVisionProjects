#import necessary libraries
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.filters.rank import entropy, equalize
from skimage.morphology import disk
import matplotlib.pyplot as plt
#import KNeighbors, DecisionTree and GaussianNB classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Declare path of RGB pictures and their annotated masks
pathToWork='C:/Users/user/PycharmProjects/ExercisesComputerVision/Exercise1/SemanticSegmentation'
# Read RGB images
imageRGB1Name='horse'
imageRGB2Name='sailboat'
inputImg1= cv2.imread(pathToWork +'/' +imageRGB1Name + '.jpg')
inputImg2= cv2.imread(pathToWork +'/' +imageRGB2Name + '.jpg')
#convert  images from one color space to another
inputImg1= cv2.cvtColor(inputImg1, cv2.COLOR_BGR2RGB)
inputImg2= cv2.cvtColor(inputImg2, cv2.COLOR_BGR2RGB)

# Load annonated masks of RGB images
annotatedImg1 = cv2.imread(pathToWork +'/' +imageRGB1Name + '_annotated.jpg',0)
annotatedImg2 = cv2.imread(pathToWork +'/' +imageRGB2Name + '_annotated.jpg',0)

#print shape of RGB images and their annonated masks
print('input image original shape with name ' +imageRGB1Name +' is:', inputImg1.shape)
print('annotated image original shape with name ' +imageRGB1Name +' is:', annotatedImg1.shape)
print('input image original shape with name ' +imageRGB2Name +' is:', inputImg2.shape)
print('annotated image original shape with name ' +imageRGB2Name +' is:', annotatedImg2.shape)

#Set dimensions size to resize images
resizedDimensions1 = (int(inputImg1.shape[0]/4), int(inputImg1.shape[1]/4))
resizedDimensions2 = (int(inputImg2.shape[0]/4), int(inputImg2.shape[1]/4))
resizedDimensions3= (int(inputImg1.shape[0]/8), int(inputImg1.shape[1]/4))
resizedDimensions4= (int(inputImg2.shape[0]/8), int(inputImg2.shape[1]/4))
resizedDimensions5=(int(inputImg1.shape[0]/4),28)
resizedDimensions6=(40,int(inputImg2.shape[1]/4))
resizedDimensions7=(33,49)
resizedDimensions8=(70,int(inputImg2.shape[1]/4))

# We need these for reshaping the fullImgPredict images in lines 376-388 predicted by classifiers
testannotatedImg1=cv2.resize(annotatedImg1, resizedDimensions3)
testannotatedImg2=cv2.resize(annotatedImg2, resizedDimensions4)
testannotatedImg3=cv2.resize(annotatedImg1, resizedDimensions5)
testannotatedImg4=cv2.resize(annotatedImg2, resizedDimensions6)
testannotatedImg5=cv2.resize(annotatedImg1, resizedDimensions7)
testannotatedImg6=cv2.resize(annotatedImg2, resizedDimensions8)

#reduce dimension size by 4
inputImg1 = cv2.resize(inputImg1, resizedDimensions1)
annotatedImg1 = cv2.resize(annotatedImg1, resizedDimensions1)
print('First input image new shape is:', inputImg1.shape)
print('First annotated image new shape is:', annotatedImg1.shape)
inputImg2 = cv2.resize(inputImg2, resizedDimensions2)
annotatedImg2 = cv2.resize(annotatedImg2, resizedDimensions2)
print('Second input image new shape is:', inputImg2.shape)
print('Second annotated image new shape is:', annotatedImg2.shape)

#change annonated image values to binary {0,1}
annotatedImg1 = np.where(annotatedImg1 > 15, 1, 0)
np.unique(annotatedImg1)
annotatedImg2 = np.where(annotatedImg2 > 15, 1, 0)
np.unique(annotatedImg2)

# illustrate the images
fig1, ax1 = plt.subplots(ncols=2, figsize=(10, 5))
ax1[0].imshow(inputImg1)
ax1[0].axis('off')
ax1[0].set_title('Horse Original Image')

ax1[1].imshow(annotatedImg1, cmap='gray')
ax1[1].axis('off')
ax1[1].set_title('Horse Annotated Mask')

plt.tight_layout()

fig2, ax2 = plt.subplots(ncols=2, figsize=(10, 5))
ax2[0].imshow(inputImg2)
ax2[0].axis('off')
ax2[0].set_title('Sailboat Original Image')

ax2[1].imshow(annotatedImg2, cmap='gray')
ax2[1].axis('off')
ax2[1].set_title('Sailboat Annotated Mask')

plt.tight_layout()

#create feature values; they help us to compare pixels among them
colorFeatures1 = inputImg1/255. #normalize color values of inputImg1 in [0,1]
edgeBasedFeatures1 = cv2.Canny(inputImg1, 10, 10, edges=None, apertureSize=5)/255 #edge features of inputImg1
colorFeatures2 = inputImg2/255. #normalize color values of InputImg2 in [0,1]
edgeBasedFeatures2 = cv2.Canny(inputImg2, 10, 10, edges=None, apertureSize=5)/255 #edge features of inputImg2
print('Color features of image 1: ', colorFeatures1)
print('Edge features of image 1: ', edgeBasedFeatures1)
print('Color features of image 2: ', colorFeatures2)
print('Edge features of image 2: ', edgeBasedFeatures2)

#features based on grayscale image, using rank from skimage library
#convert RGB images into grayscale images
grayImg1 = cv2.cvtColor(inputImg1, cv2.COLOR_BGR2GRAY)
grayImg2 = cv2.cvtColor(inputImg2, cv2.COLOR_BGR2GRAY)
entropyBasedFeatures1 = entropy(grayImg1, disk(5))/6. #entropy features of grayImg1
equalizationBasedFeatures1 = equalize(grayImg1, disk(5))/255  #equalization features of grayImg1
entropyBasedFeatures2 = entropy(grayImg2, disk(5))/6. #entropy features of grayImg2
equalizationBasedFeatures2 = equalize(grayImg2, disk(5))/255 #equalization features of grayImg2
cornerFeatures1=cv2.cornerHarris(grayImg1, 8, 3 , 0.04)/255 #corner features of grayImg1
cornerFeatures1= cv2.dilate(cornerFeatures1,None)# Applying dilation to increase the object area and to emphasize features
cornerFeatures2=cv2.cornerHarris(grayImg2, 8, 3 , 0.04)/255 #corner features of grayImg2
cornerFeatures2= cv2.dilate(cornerFeatures2,None)# Applying dilation to increase the object area and to emphasize features

#design plots for features based on grayscale images
fig1, ax1 = plt.subplots(ncols=3, figsize=(10, 5))
ax1[0].imshow(grayImg1, cmap='gray')
ax1[0].axis('off')
ax1[0].set_title('Gray Img 1')

ax1[1].imshow(entropyBasedFeatures1, cmap='gray')
ax1[1].axis('off')
ax1[1].set_title('Entropy')

ax1[2].imshow(equalizationBasedFeatures1, cmap='gray')
ax1[2].axis('off')
ax1[2].set_title('Global equalization')

plt.tight_layout()

fig2, ax2 = plt.subplots(ncols=3, figsize=(10, 5))
ax2[0].imshow(grayImg2, cmap='gray')
ax2[0].axis('off')
ax2[0].set_title('Gray Img 2')

ax2[1].imshow(entropyBasedFeatures2, cmap='gray')
ax2[1].axis('off')
ax2[1].set_title('Entropy')

ax2[2].imshow(equalizationBasedFeatures2, cmap='gray')
ax2[2].axis('off')
ax2[2].set_title('Global equalization')

plt.tight_layout()

# Threshold for an optimal value. It may vary depending on the images.
inputImg1[cornerFeatures1>0.02*cornerFeatures1.max()]=[255, 0, 255]
inputImg2[cornerFeatures2>0.02*cornerFeatures2.max()]=[255, 0, 255]

# show RGB images
cv2.imshow('Corner features in horse image',inputImg1)
cv2.imshow('Corner features in sailboat image',inputImg2)

for inputFeatureVals in range(3):  # Loop for every input feature values.

      # choices of feature values
      if inputFeatureVals==0: #Only RGB values
            featureVals1 = np.dstack(colorFeatures1)
            featureVals2 = np.dstack(colorFeatures2)
            print("Experiment 1 with only RGB values")
            # reshape the data to feed them to the classifier
            # data to use
            inputData1 = np.reshape(featureVals1, [-1, 6])
            outputData1 = np.reshape(annotatedImg1, [-1, 1])
            inputData2 = np.reshape(featureVals2, [-1, 6])
            outputData2 = np.reshape(annotatedImg2, [-1, 1])

      elif inputFeatureVals==1: #Edge, corner, entropy and equalization values
            featureVals1 = np.dstack((edgeBasedFeatures1,entropyBasedFeatures1,equalizationBasedFeatures1,cornerFeatures1))
            featureVals2 = np.dstack((edgeBasedFeatures2,entropyBasedFeatures2,equalizationBasedFeatures2,cornerFeatures2))
            print("Experiment 2 with all feature values except from RGB values")
            # reshape the data to feed them to the classifier
            # data to use
            inputData1 = np.reshape(featureVals1, [-1, 6])
            outputData1 = np.reshape(annotatedImg1, [-1, 1])
            inputData2 = np.reshape(featureVals2, [-1, 10])
            outputData2 = np.reshape(annotatedImg2, [-1, 1])

      else: #RGB, edge, corner, entropy and equalization values
            featureVals1 = np.dstack(
                  (colorFeatures1,edgeBasedFeatures1, entropyBasedFeatures1, equalizationBasedFeatures1, cornerFeatures1))
            featureVals2 = np.dstack(
                  (colorFeatures2,edgeBasedFeatures2, entropyBasedFeatures2, equalizationBasedFeatures2, cornerFeatures2))
            print("Experiment 3 with all feature values")
            # reshape the data to feed them to the classifier
            # data to use
            inputData1 = np.reshape(featureVals1, [-1, 8])
            outputData1 = np.reshape(annotatedImg1, [-1, 1])
            inputData2 = np.reshape(featureVals2, [-1, 10])
            outputData2 = np.reshape(annotatedImg2, [-1, 1])

      #  print max and shapes of feature values, input and output datas.
      print(featureVals1.shape)
      print(inputData1.shape)
      print(inputData1.max())
      print(outputData1.shape)
      print(outputData1.max())
      print(featureVals2.shape)
      print(inputData2.shape)
      print(inputData2.max())
      print(outputData2.shape)
      print(outputData2.max())

      # create the train/test data sets indexes for the RGB images
      trainDataPercentage = 0.8
      indices1 = np.random.permutation(inputData1.shape[0])
      train_idx1, test_idx1 = indices1[:int(inputData1.shape[0] * trainDataPercentage)], \
                              indices1[int(inputData1.shape[0] * trainDataPercentage) + 1:]
      indices2 = np.random.permutation(inputData2.shape[0])
      train_idx2, test_idx2 = indices2[:int(inputData2.shape[0] * trainDataPercentage)], \
                              indices2[int(inputData2.shape[0] * trainDataPercentage) + 1:]

      # train and evaluate classifiers
      # knn
      knn1 = KNeighborsClassifier()
      knn1.fit(inputData1[train_idx1, :], outputData1[train_idx1].ravel())
      knn2 = KNeighborsClassifier()
      knn2.fit(inputData2[train_idx2, :], outputData2[train_idx2].ravel())
      # classification tree
      clf1 = DecisionTreeClassifier().fit(inputData1[train_idx1, :], outputData1[train_idx1].ravel())
      clf2 = DecisionTreeClassifier().fit(inputData2[train_idx2, :], outputData2[train_idx2].ravel())
      # Naive Bayes
      gnb1 = GaussianNB()
      gnb2 = GaussianNB()
      gnb1.fit(inputData1[train_idx1, :], outputData1[train_idx1].ravel())
      gnb2.fit(inputData2[train_idx2, :], outputData2[train_idx2].ravel())

      # knn predictions
      # now check for both train and test data, how well the model learned the patterns
      y_pred_train1 = knn1.predict(inputData1[train_idx1, :])
      y_pred_test1 = knn1.predict(inputData1[test_idx1, :])
      y_pred_train2 = knn2.predict(inputData2[train_idx2, :])
      y_pred_test2 = knn2.predict(inputData2[test_idx2, :])

      # calculate the accuracy, precision, recall and f1 scores
      acc_train1 = accuracy_score(outputData1[train_idx1].ravel(), y_pred_train1)
      acc_train2 = accuracy_score(outputData2[train_idx2].ravel(), y_pred_train2)
      acc_test1 = accuracy_score(outputData1[test_idx1].ravel(), y_pred_test1)
      acc_test2 = accuracy_score(outputData2[test_idx2].ravel(), y_pred_test2)
      pre_train1 = precision_score(outputData1[train_idx1].ravel(), y_pred_train1, average='macro')
      pre_test1 = precision_score(outputData1[test_idx1].ravel(), y_pred_test1, average='macro')
      pre_train2 = precision_score(outputData2[train_idx2].ravel(), y_pred_train2, average='macro')
      pre_test2 = precision_score(outputData2[test_idx2].ravel(), y_pred_test2, average='macro')
      rec_train1 = recall_score(outputData1[train_idx1].ravel(), y_pred_train1, average='macro')
      rec_test1 = recall_score(outputData1[test_idx1].ravel(), y_pred_test1, average='macro')
      rec_train2 = recall_score(outputData2[train_idx2].ravel(), y_pred_train2, average='macro')
      rec_test2 = recall_score(outputData2[test_idx2].ravel(), y_pred_test2, average='macro')
      f1_train1 = f1_score(outputData1[train_idx1].ravel(), y_pred_train1, average='macro')
      f1_test1 = f1_score(outputData1[test_idx1].ravel(), y_pred_test1, average='macro')
      f1_train2 = f1_score(outputData2[train_idx2].ravel(), y_pred_train2, average='macro')
      f1_test2 = f1_score(outputData2[test_idx2].ravel(), y_pred_test2, average='macro')

      # print the scores
      print('Accuracy scores of K-NN classifier 1 are:',
            'train: {:.2f}'.format(acc_train1), 'and test: {:.2f}.'.format(acc_test1))
      print('Accuracy scores of K-NN classifier 2 are:',
            'train: {:.2f}'.format(acc_train2), 'and test: {:.2f}.'.format(acc_test2))
      print('Precision scores of K-NN classifier 1 are:',
            'train: {:.2f}'.format(pre_train1), 'and test: {:.2f}.'.format(pre_test1))
      print('Precision scores of K-NN classifier 2 are:',
            'train: {:.2f}'.format(pre_train2), 'and test: {:.2f}.'.format(pre_test2))
      print('Recall scores of K-NN classifier 1 are:',
            'train: {:.2f}'.format(rec_train1), 'and test: {:.2f}.'.format(rec_test1))
      print('Recall scores of K-NN classifier 2 are:',
            'train: {:.2f}'.format(rec_train2), 'and test: {:.2f}.'.format(rec_test2))
      print('F1 scores of K-NN classifier 1 are:',
            'train: {:.2f}'.format(f1_train1), 'and test: {:.2f}.'.format(f1_test1))
      print('F1 scores of K-NN classifier 2 are:',
            'train: {:.2f}'.format(f1_train2), 'and test: {:.2f}.'.format(f1_test2))
      print('')

      # classification tree
      # predict outcomes for test data and calculate the test scores
      y_pred_train1 = clf1.predict(inputData1[train_idx1, :])
      y_pred_test1 = clf1.predict(inputData1[test_idx1, :])
      y_pred_train2 = clf2.predict(inputData2[train_idx2, :])
      y_pred_test2 = clf2.predict(inputData2[test_idx2, :])

      # calculate the accuracy, precision, recall and f1 scores
      acc_train1 = accuracy_score(outputData1[train_idx1].ravel(), y_pred_train1)
      acc_train2 = accuracy_score(outputData2[train_idx2].ravel(), y_pred_train2)
      acc_test1 = accuracy_score(outputData1[test_idx1].ravel(), y_pred_test1)
      acc_test2 = accuracy_score(outputData2[test_idx2].ravel(), y_pred_test2)
      pre_train1 = precision_score(outputData1[train_idx1].ravel(), y_pred_train1, average='macro')
      pre_test1 = precision_score(outputData1[test_idx1].ravel(), y_pred_test1, average='macro')
      pre_train2 = precision_score(outputData2[train_idx2].ravel(), y_pred_train2, average='macro')
      pre_test2 = precision_score(outputData2[test_idx2].ravel(), y_pred_test2, average='macro')
      rec_train1 = recall_score(outputData1[train_idx1].ravel(), y_pred_train1, average='macro')
      rec_test1 = recall_score(outputData1[test_idx1].ravel(), y_pred_test1, average='macro')
      rec_train2 = recall_score(outputData2[train_idx2].ravel(), y_pred_train2, average='macro')
      rec_test2 = recall_score(outputData2[test_idx2].ravel(), y_pred_test2, average='macro')
      f1_train1 = f1_score(outputData1[train_idx1].ravel(), y_pred_train1, average='macro')
      f1_test1 = f1_score(outputData1[test_idx1].ravel(), y_pred_test1, average='macro')
      f1_train2 = f1_score(outputData2[train_idx2].ravel(), y_pred_train2, average='macro')
      f1_test2 = f1_score(outputData2[test_idx2].ravel(), y_pred_test2, average='macro')

      # print the scores
      print('Accuracy scores of Decision Tree classifier 1 are:',
            'train: {:.2f}'.format(acc_train1), 'and test: {:.2f}.'.format(acc_test1))
      print('Accuracy scores of Decision Tree classifier 2 are:',
            'train: {:.2f}'.format(acc_train2), 'and test: {:.2f}.'.format(acc_test2))
      print('Precision scores of Decision Tree classifier 1 are:',
            'train: {:.2f}'.format(pre_train1), 'and test: {:.2f}.'.format(pre_test1))
      print('Precision scores of Decision Tree classifier 2 are:',
            'train: {:.2f}'.format(pre_train2), 'and test: {:.2f}.'.format(pre_test2))
      print('Recall scores of Decision Tree classifier 1 are:',
            'train: {:.2f}'.format(rec_train1), 'and test: {:.2f}.'.format(rec_test1))
      print('Recall scores of Decision Tree classifier 2 are:',
            'train: {:.2f}'.format(rec_train2), 'and test: {:.2f}.'.format(rec_test2))
      print('F1 scores of Decision Tree classifier 1 are:',
            'train: {:.2f}'.format(f1_train1), 'and test: {:.2f}.'.format(f1_test1))
      print('F1 scores of Decision Tree classifier 2 are:',
            'train: {:.2f}'.format(f1_train2), 'and test: {:.2f}.'.format(f1_test2))
      print('')

      # naive Bayes
      # now check for both train and test data, how well the model learned the patterns
      y_pred_train1 = gnb1.predict(inputData1[train_idx1, :])
      y_pred_test1 = gnb1.predict(inputData1[test_idx1, :])
      y_pred_train2 = gnb2.predict(inputData2[train_idx2, :])
      y_pred_test2 = gnb2.predict(inputData2[test_idx2, :])

      # calculate the accuracy, precision, recall and f1 scores
      acc_train1 = accuracy_score(outputData1[train_idx1].ravel(), y_pred_train1)
      acc_train2 = accuracy_score(outputData2[train_idx2].ravel(), y_pred_train2)
      acc_test1 = accuracy_score(outputData1[test_idx1].ravel(), y_pred_test1)
      acc_test2 = accuracy_score(outputData2[test_idx2].ravel(), y_pred_test2)
      pre_train1 = precision_score(outputData1[train_idx1].ravel(), y_pred_train1, average='macro')
      pre_test1 = precision_score(outputData1[test_idx1].ravel(), y_pred_test1, average='macro')
      pre_train2 = precision_score(outputData2[train_idx2].ravel(), y_pred_train2, average='macro')
      pre_test2 = precision_score(outputData2[test_idx2].ravel(), y_pred_test2, average='macro')
      rec_train1 = recall_score(outputData1[train_idx1].ravel(), y_pred_train1, average='macro')
      rec_test1 = recall_score(outputData1[test_idx1].ravel(), y_pred_test1, average='macro')
      rec_train2 = recall_score(outputData2[train_idx2].ravel(), y_pred_train2, average='macro')
      rec_test2 = recall_score(outputData2[test_idx2].ravel(), y_pred_test2, average='macro')
      f1_train1 = f1_score(outputData1[train_idx1].ravel(), y_pred_train1, average='macro')
      f1_test1 = f1_score(outputData1[test_idx1].ravel(), y_pred_test1, average='macro')
      f1_train2 = f1_score(outputData2[train_idx2].ravel(), y_pred_train2, average='macro')
      f1_test2 = f1_score(outputData2[test_idx2].ravel(), y_pred_test2, average='macro')

      # print the scores
      print('Accuracy scores of GNB classifier 1 are:',
            'train: {:.2f}'.format(acc_train1), 'and test: {:.2f}.'.format(acc_test1))
      print('Accuracy scores of GNB classifier 2 are:',
            'train: {:.2f}'.format(acc_train2), 'and test: {:.2f}.'.format(acc_test2))
      print('Precision scores of GNB classifier 1 are:',
            'train: {:.2f}'.format(pre_train1), 'and test: {:.2f}.'.format(pre_test1))
      print('Precision scores of GNB classifier 2 are:',
            'train: {:.2f}'.format(pre_train2), 'and test: {:.2f}.'.format(pre_test2))
      print('Recall scores of GNB classifier 1 are:',
            'train: {:.2f}'.format(rec_train1), 'and test: {:.2f}.'.format(rec_test1))
      print('Recall scores of GNB classifier 2 are:',
            'train: {:.2f}'.format(rec_train2), 'and test: {:.2f}.'.format(rec_test2))
      print('F1 scores of GNB classifier 1 are:',
            'train: {:.2f}'.format(f1_train1), 'and test: {:.2f}.'.format(f1_test1))
      print('F1 scores of GNB classifier 2 are:',
            'train: {:.2f}'.format(f1_train2), 'and test: {:.2f}.'.format(f1_test2))
      print('')

      for i in range(3): #Loop for every classifier creating the plots

            # run it on the image
            if i==0:   #KNN predictions
                  fullImgPred1 = knn1.predict(inputData1)
                  fullImgPred2 = knn2.predict(inputData2)
                  msg='KNN predictions'

            elif i==1: #Decision Tree predictions
                  fullImgPred1 = clf1.predict(inputData1)
                  fullImgPred2 = clf2.predict(inputData2)
                  msg='Decision Tree predictions'

            else:      #Naive Bayes predictions
                  fullImgPred1 = gnb1.predict(inputData1)
                  fullImgPred2 = gnb2.predict(inputData2)
                  msg = 'Naive Bayes predictions'

            #Reshaping segmented images depends on inputFeatureVals
            if inputFeatureVals==0: #Only RGB values
                  segmentedImg1 = np.reshape(fullImgPred1,testannotatedImg1.shape[:2])
                  segmentedImg2 = np.reshape(fullImgPred2,testannotatedImg2.shape[:2])

            elif inputFeatureVals==1: #Edge, corner, entropy and equalization values
                  segmentedImg1 = np.reshape(fullImgPred1, testannotatedImg3.shape[:2])
                  segmentedImg2 = np.reshape(fullImgPred2, testannotatedImg4.shape[:2])

            else: #RGB, edge, corner, entropy and equalization values
                  segmentedImg1 = np.reshape(fullImgPred1, testannotatedImg5.shape[:2])
                  segmentedImg2 = np.reshape(fullImgPred2, testannotatedImg6.shape[:2])

            # display the outcome
            fig1, ax1 = plt.subplots(ncols=2, figsize=(10, 5))
            ax1[0].imshow(segmentedImg1, cmap='gray')
            ax1[0].axis('off')
            ax1[0].set_title(msg)

            ax1[1].imshow(annotatedImg1, cmap='gray')
            ax1[1].axis('off')
            ax1[1].set_title('Actual labels')

            plt.tight_layout()

            np.unique(segmentedImg1)

            fig2, ax2 = plt.subplots(ncols=2, figsize=(10, 5))
            ax2[0].imshow(segmentedImg2, cmap='gray')
            ax2[0].axis('off')
            ax2[0].set_title(msg)

            ax2[1].imshow(annotatedImg2, cmap='gray')
            ax2[1].axis('off')
            ax2[1].set_title('Actual labels')

            plt.tight_layout()

            np.unique(segmentedImg2)