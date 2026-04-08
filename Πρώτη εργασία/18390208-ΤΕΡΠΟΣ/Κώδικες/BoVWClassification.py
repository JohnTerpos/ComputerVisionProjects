# import the required libraries
import cv2
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans, MeanShift  # KMeans and MeanShift
from sklearn.svm import SVC #support vector machines
# Convert Categorical Data For Scikit-Learn
from sklearn import preprocessing
# import libraries for accuracy, precision, recall and f1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# return a dictionary that holds all images category by category.
def load_images_from_folder(folder, inputImageSize):
    images = {}
    for filename in os.listdir(folder):
        category = [] #Initialize the list
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat)
            #print(' .. parsing image', cat)
            if img is not None:
                # grayscale it
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # resize it, if necessary
                img = cv2.resize(img, (inputImageSize[0], inputImageSize[1]))

                category.append(img)
        images[filename] = category
        print(' . Finished parsing images from:', folder)
    return images

# Creation of the histograms. To create our each image by a histogram. We will create a vector of k values for each
# image. For each keypoints in an image, we will find the nearest center, defined using training set
# and increase by one its value
def mapFeatureValsToHistogram (DataFeaturesByClass, visualWords, TrainedModel):
    # depending on the training model created by clustering algorithms we may not use all the inputs
    histogramsList = []
    targetClassList = []
    numberOfBinsPerHistogram = visualWords.shape[0]

    for categoryIdx, featureValues in DataFeaturesByClass.items():
        for tmpImageFeatures in featureValues: # check one by one the values in each image for all images
            tmpImageHistogram = np.zeros(numberOfBinsPerHistogram)
            tmpIdx = list(TrainedModel.predict(tmpImageFeatures.astype('float')))
            clustervalue, visualWordMatchCounts = np.unique(tmpIdx, return_counts=True)
            tmpImageHistogram[clustervalue] = visualWordMatchCounts
            # normalize the histogram values
            numberOfDetectedPointsInThisImage = tmpIdx.__len__()
            tmpImageHistogram = tmpImageHistogram/numberOfDetectedPointsInThisImage

            # update the input and output corresponding lists
            histogramsList.append(tmpImageHistogram)
            targetClassList.append(categoryIdx)

    return histogramsList, targetClassList

# Creates descriptors using SIFT detector algorithm
# Takes one parameter that is images dictionary
# Return an array whose first index holds the descriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
def detector_features_SIFT(images):
    print(' . start detecting points and calculating features for a given image set with algorithm SIFT')
    detector_vectors = {}
    descriptor_list = []
    detectorToUse = cv2.xfeatures2d.SIFT_create() #initialize the SIFT detector algorithm
    for nameOfCategory, availableImages in images.items():
        print(' . we are in category:', nameOfCategory )
        features = [] #Initialize the list
        tmpImgCounter = 1
        for img in availableImages:
            kp, des = detectorToUse.detectAndCompute(img, None) # compute the descriptors and keypoints with SIFT
            tmpImgCounter += 1
            if des is None:
                print(' .. WARNING: image {:d} cannot be used'.format(tmpImgCounter))
            else:
                descriptor_list.extend(des)
                features.append(des)
        detector_vectors[nameOfCategory] = features
        print(' . finished detecting points and calculating features for a given image set with algorithm SIFT')
    return [descriptor_list, detector_vectors]

# Creates descriptors using BRISK detector algorithm
# Takes one parameter that is images dictionary
# Return an array whose first index holds the descriptor_list without an order
# And the second index holds the brisk_vectors dictionary which holds the descriptors but this is seperated class by class
def detector_features_BRISK(images):
    print(' . start detecting points and calculating features for a given image set with algorithm BRISK')
    detector_vectors = {}
    descriptor_list = [] #Initialize the list
    detectorToUse = cv2.BRISK_create() #initialize the BRISK detector algorithm
    for nameOfCategory, availableImages in images.items():
        print(' . we are in category:', nameOfCategory )
        features = [] #Initialize the list
        tmpImgCounter = 1
        for img in availableImages:
            kp, des = detectorToUse.detectAndCompute(img, None) # compute the descriptors and keypoints with BRISK
            tmpImgCounter += 1
            if des is None:
                print(' .. WARNING: image {:d} cannot be used'.format(tmpImgCounter))
            else:
                descriptor_list.extend(des)
                features.append(des)
        detector_vectors[nameOfCategory] = features
        print(' . finished detecting points and calculating features for a given image set with algorithm BRISK')
    return [descriptor_list, detector_vectors]

# Creates descriptors using BRIEF detector algorithm
# Takes one parameter that is images dictionary
# Return an array whose first index holds the descriptor_list without an order
# And the second index holds the brief_vectors dictionary which holds the descriptors but this is seperated class by class
def detector_features_BRIEF(images):
    print(' . start detecting points and calculating features for a given image set with algorithm BRIEF')
    detector_vectors = {}
    descriptor_list = []
    fast = cv2.xfeatures2d.StarDetector_create() # initialize FAST detector
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() # Initiate BRIEF extractor
    for nameOfCategory, availableImages in images.items():
        print(' . we are in category:', nameOfCategory )
        features = [] #Initialize the list
        tmpImgCounter = 1
        for img in availableImages:
            # find the keypoints with STAR
            kp = fast.detect(img, None)

            # compute the descriptors with BRIEF
            kp, des = brief.compute(img, kp)
            tmpImgCounter += 1
            descriptor_list.extend(des)
            features.append(des)

        detector_vectors[nameOfCategory] = features
        print(' . finished detecting points and calculating features for a given image set with algorithm BRIEF,')
    return [descriptor_list, detector_vectors]

# A k-means clustering algorithm who takes 2 parameter which is number
# of cluster(k) and the other is descriptors list(unordered 1d array)
# Returns an array that holds central points.
def kmeansVisualWordsCreation(k, descriptor_list):
    print(' . calculating central points for the existing feature values with Kmeans.')
    batchSize = np.ceil(descriptor_list.__len__()/50).astype('int')
    kmeansModel = MiniBatchKMeans(n_clusters=k, batch_size=batchSize, verbose=0) # Use k-means algorithm to create the model
    kmeansModel.fit(descriptor_list)  # train the model
    visualWords = kmeansModel.cluster_centers_ # centers of reference
    print(' . done calculating central points for the given feature set with Kmeans.')
    return visualWords, kmeansModel # Return the Bag of Visual Words and the K-Means model.

# A MeanShift clustering algorithm who takes 1 parameter which is
# descriptors list(unordered 1d array) and returns an array
# that holds centers by looking at the density of points
def MeanShiftVisualWordsCreation(descriptor_list):
    print('Calculating central points for the existing feature values with Mean Shift.')
    ShiftModel = MeanShift(bandwidth=10)  # Use MeanShift to create the model.
    ShiftModel.fit(descriptor_list) # train the model
    VisualWords = ShiftModel.labels_  # centers of reference.
    print('Done calculating central points for the given feature set with Mean Shift.')
    return VisualWords, ShiftModel  # Return the Bag of Visual Words and the MeanShift model.

# here we run the code

if __name__ == '__main__':
    #define a fixed image size to work with
    inputImageSize = [200, 200, 3] #define the FIXED size that CNN will have as input

    #define the path to train and test files
    TrainImagesFilePath= 'C:/Users/user/PycharmProjects/ExercisesComputerVision/Exercise1/TrainImages'
    TestImagesFilePath=  'C:/Users/user/PycharmProjects/ExercisesComputerVision/Exercise1/TestImages'
    for Detector_Method in range(3):  # Loop for every Detector method.
        for Histogram_Method in range(2):  # Loop for every Histogram method.
            # load the train images
            trainImages = load_images_from_folder(TrainImagesFilePath, inputImageSize)# take all images category by category for train set
            #calculate points and descriptor values per image using SIFT, BRISK and BRIEF algorithms
            if Detector_Method==0:
                trainDataFeatures = detector_features_SIFT(trainImages) #SIFT algortihm
            elif Detector_Method==1:
                trainDataFeatures = detector_features_BRISK(trainImages) #BRISK algorithm
            else:
                trainDataFeatures = detector_features_BRIEF(trainImages) #BRIEF algorithm

            # Takes the descriptor list which is unordered one
            TrainDescriptorList = trainDataFeatures[0]
            #print(TrainDescriptorList)

            if Histogram_Method==0:    #Create histogram with algorithm k-means
                numberOfClasses = trainImages.__len__()  # retrieve num of classes from dictionary
                possibleNumOfCentersToUse = 10 * numberOfClasses #Set number of clusters
                visualWords, TrainedKmeansModel = kmeansVisualWordsCreation(possibleNumOfCentersToUse, TrainDescriptorList)
                # Takes the detector algorithm feature values that is seperated class by class for train data, we need this to calculate the histograms
                trainBoVWFeatureVals = trainDataFeatures[1]
                # create the train input train output format
                trainHistogramsList, trainTargetsList = mapFeatureValsToHistogram(trainBoVWFeatureVals, visualWords,
                                                                              TrainedKmeansModel)
            else: #Create histogram with algorithm MeanShift
                visualWords, TrainedMeanShiftModel = MeanShiftVisualWordsCreation(TrainDescriptorList)
                # Takes the detector algorithm feature values that is seperated class by class for train data, we need this to calculate the histograms
                trainBoVWFeatureVals = trainDataFeatures[1]
                # create the train input train output format
                trainHistogramsList, trainTargetsList = mapFeatureValsToHistogram(trainBoVWFeatureVals, visualWords,
                                                                       TrainedMeanShiftModel)
            X_train = np.stack(trainHistogramsList, axis=0) #set X_train
            # Create a label (category) encoder object
            labelEncoder = preprocessing.LabelEncoder()
            labelEncoder.fit(trainTargetsList)
            # convert the categories from strings to names
            y_train = labelEncoder.transform(trainTargetsList)

            svm = SVC() #Create support vector machine Classifier
            svm.fit(X_train, y_train) # Train the model
            print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))

            # load the test images
            testImages = load_images_from_folder(TestImagesFilePath,
                                                 inputImageSize)  # take all images category by category for test set

            # calculate points and descriptor values per image using SIFT, BRISK and BRIEF algorithms
            if Detector_Method == 0:
                testDataFeatures = detector_features_SIFT(testImages)  # SIFT algorithm
            elif Detector_Method == 1:
                testDataFeatures = detector_features_BRISK(testImages)  # BRISK algorithm
            else:
                testDataFeatures = detector_features_BRIEF(testImages)  # BRIEF algorithm

            # Takes the detector algorithm feature values that is seperated class by class for test data, we need this to calculate the histograms
            testBoVWFeatureVals = testDataFeatures[1]

            if Histogram_Method == 0:  # For the K Means method.
                TestHistogramsList, TestTargetsList = mapFeatureValsToHistogram(testBoVWFeatureVals, visualWords,
                                                                         TrainedKmeansModel)
            else:  # For the MeanShift method.
                TestHistogramsList, TestTargetsList = mapFeatureValsToHistogram(testBoVWFeatureVals, visualWords,
                                                                         TrainedMeanShiftModel)
            #create the test input / test output format
            X_test = np.array(TestHistogramsList)
            y_test = labelEncoder.transform(TestTargetsList)

            # support vector machines
            # now check for both train and test data, how well the model learned the patterns
            y_pred_train = svm.predict(X_train)
            y_pred_test = svm.predict(X_test)
            # calculate the scores
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            pre_train = precision_score(y_train, y_pred_train, average='macro')
            pre_test = precision_score(y_test, y_pred_test, average='macro')
            rec_train = recall_score(y_train, y_pred_train, average='macro')
            rec_test = recall_score(y_test, y_pred_test, average='macro')
            f1_train = f1_score(y_train, y_pred_train, average='macro')
            f1_test = f1_score(y_test, y_pred_test, average='macro')

            # print the scores
            print('Accuracy scores of SVM classifier are:',
                  'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
            print('Precision scores of SVM classifier are:',
                  'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
            print('Recall scores of SVM classifier are:',
                  'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
            print('F1 scores of SVM classifier are:',
                  'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
            print('')