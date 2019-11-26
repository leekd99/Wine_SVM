'''
Add the code below the TO-DO statements to finish the assignment. Keep the interfaces
of the provided functions unchanged. Change the returned values of these functions
so that they are consistent with the assignment instructions. Include additional import
statements and functions if necessary.
'''

import csv
import numpy as np
import matplotlib.pyplot as plt

'''
The loss functions shall return a scalar, which is the *average* loss of all the examples
'''

'''
For instance, the square loss of all the training examples is computed as below:

def squared_loss(train_y, pred_y):

    loss = np.mean(np.square(train_y - pred_y))

    return loss
'''


def logistic_loss(train_y, pred_y):
    # L(y,f(x)) = log(1+exp(-y*f(x))) where y is your label and f(x) is your predicted value
    # argmin sum(i,n)(yi,f(xi))

    # multiply train_y and pred_y component wise then multiply each component by minus 1
    # exponentiate the results and add 1
    # take the log then sum, then divide by the number of elements

    return np.sum(np.log(1 + np.exp((-1)*np.multiply(train_y, pred_y)))) / len(train_y)


def hinge_loss(train_y, pred_y):
    # take predicted y and training label y and multiply them subtracting them from 1
    # take the max of new values and 0
    # add up the values and divide by how many values there were
    return np.sum(np.maximum(np.zeros(len(train_y)), 1-np.multiply(train_y, pred_y))) / len(train_y)

'''
The regularizers shall compute the loss without considering the bias term in the weights
'''


def l1_reg(w):
    # Remove the bias term from w vector and calculate l1 norm
    return np.linalg.norm(w[1:], 1)


def l2_reg(w):
    # Remove the bias term from w vector and calculate l1 norm
    return np.dot(w[1:], w[1:])


def train_classifier(train_x, train_y, learn_rate, loss, lambda_val=None, regularizer=None):

    # if no lambda value was given default to 1
    if lambda_val is None:
        lambda_val = 1
    # if no regularizer function was given default to 0
    if regularizer is None:
        regularizer = 0

    # small h to change our input slightly
    h = 0.0001
    # number fo iterations we should do to find the new wegiths vector
    epoch = 1000

    # for graphing
    x_axis = np.zeros(epoch)
    y_axis = np.zeros(epoch)

    # initialize the weights with bias term
    w_vector = np.array(np.random.normal(0, 0.01, 11))
    # initialize bias with all 1s
    bias = np.ones(np.size(train_x, 0))
    # add the bias column to the front of x
    train_x = np.column_stack((bias, train_x))

    # run for however many times epoch was set to
    for iteration in range(epoch):
        # allows us to skip bias
        bFirst_elt = True
        # initialize the gradient vector
        gradient_of_loss = np.zeros(11)
        # iterate through each value in w and add a tiny h
        for val in range(len(w_vector)):
            # make temp weights vector so the changes don't carry over
            w_vector_delta = np.array(w_vector)
            # skip the bias
            if bFirst_elt:
                bFirst_elt = False
                continue

            # add the small term h to a single weight in the vector
            w_vector_delta[val] += h

            # using the small changed weights predict the label sign(f(x)) with f(x) = x*w
            pred_y = np.sign(train_x.dot(w_vector))
            pred_y_h = np.sign(train_x.dot(w_vector_delta))

            # if no regularizer was given set it to 0
            if regularizer == 0:
                gradient_of_loss[val] += ((lambda_val * regularizer) + loss(train_y, pred_y_h))
                gradient_of_loss[val] -= ((lambda_val * regularizer) + loss(train_y, pred_y))
                gradient_of_loss[val] = gradient_of_loss[val]/h
            # continue on
            else:
                gradient_of_loss[val] += ((lambda_val * regularizer(w_vector_delta)) + loss(train_y, pred_y_h))
                gradient_of_loss[val] -= ((lambda_val * regularizer(w_vector)) + loss(train_y, pred_y))
                gradient_of_loss[val] = gradient_of_loss[val]/h
                # save data for plot
                x_axis[iteration] = iteration + 1
                y_axis[iteration] = (lambda_val * regularizer(w_vector)) + loss(train_y, np.sign(np.dot(train_x, w_vector)))

        # update the weights
        w_vector = w_vector - (learn_rate*gradient_of_loss)

    ''' uncomment to plot graph
    # for graph
    plt.plot(x_axis, y_axis)
    plt.xlabel('number of iterations')
    plt.ylabel('loss')
    plt.title('loss vs. number of iterations')
    plt.show()
    '''

    return w_vector


def test_classifier(w, test_x):
    # initialize bias terms
    bias = np.ones(np.size(test_x, 0))
    # add bias term to test data
    test_x = np.column_stack((bias, test_x))
    # return the predicted y value
    return test_x.dot(w)

def accuracy(pred_y, test_y):

    # iterate through the results and classify them
    for i in range(len(pred_y)):
        if pred_y[i] >= 0:
            pred_y[i] = 1
        else:
            pred_y[i] = -1

    # compare the two results by multiplying them
    acc = np.multiply(pred_y, test_y)
    for i in range(len(acc)):
        # inaccurate results will have negative sign
        if acc[i] < 0:
            acc[i] = 0
    # sum up correctly classified points and divide by all points
    return np.sum(acc)/len(acc)

def main():
    # Read the training data file
    szDatasetPath = 'winequality-white.csv'
    listClasses = []
    listAttrs = []
    bFirstRow = True
    with open(szDatasetPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            if bFirstRow:
                bFirstRow = False
                continue
            if int(row[-1]) < 6:
                listClasses.append(-1)
                listAttrs.append(list(map(float, row[1:len(row) - 1])))
            elif int(row[-1]) > 6:
                listClasses.append(+1)
                listAttrs.append(list(map(float, row[1:len(row) - 1])))

    dataX = np.array(listAttrs)
    dataY = np.array(listClasses)

    '''
    Begin splitting data into 5 sets
    '''

    # get the split index for the whole data set
    splitIndex = (np.size(dataX, 0)/5)

    # set left and right indices of the splits
    left = 0
    right = int(splitIndex)

    # initializing array to hold all 5 splits
    setOfSplits_X = []
    setOfSplits_Y = []

    # begin splitting
    for i in range(5):
        # temporarily store the split of data
        tempX = dataX[left:right, :]
        tempY = dataY[left:right,]

        # append the split data to an array so that the 5 splits can be referenced by indices (0-4)
        setOfSplits_X.append(tempX)
        setOfSplits_Y.append(tempY)

        # increment the left and right indices so that it can slice off the correct chunk of data
        left += int(splitIndex)
        right = int(left+splitIndex)

    #Convert to numpy array for easy access
    setOfSplits_X = np.array(setOfSplits_X)
    setOfSplits_Y = np.array(setOfSplits_Y)

    '''
    Combine splits into larger sets for training and pick set for test
    '''

    # initialize list to hold train and test x/y
    setof_trainX = []
    setof_trainY = []
    setof_testX = []
    setof_testY = []

    # iterate through 5 cases for 5 fold cross validation
    for rowofSet in range(5):
        # initialize numpy array to hold temporary train and test sets
        temp_trainX = np.array([])
        temp_trainY = np.array([])
        temp_testX = np.array([])
        temp_testY = np.array([])

        # go through each data set split and split test off and combine rest to be train
        for rowofSplit in range(5):
            # if the index of the current data split is equal to case number make that the test set
            if (rowofSplit == rowofSet):
                temp_testX = setOfSplits_X[rowofSplit]
                temp_testY = setOfSplits_Y[rowofSplit]
            # if it's the first time adding to training array just let the train set equal the temp set
            elif (np.size(temp_trainX, 0) == 0):
                temp_trainX = setOfSplits_X[rowofSplit]
                temp_trainY = setOfSplits_Y[rowofSplit]
            # append the rest of the train set to the train array
            else:
                temp_trainX = np.concatenate((temp_trainX, setOfSplits_X[rowofSplit]))
                temp_trainY = np.concatenate((temp_trainY, setOfSplits_Y[rowofSplit]))

        # append the train test split to respective x and y list
        setof_trainX.append(temp_trainX)
        setof_trainY.append(temp_trainY)
        setof_testX.append(temp_testX)
        setof_testY.append(temp_testY)



    # convert to numpy array for easy access
    setof_trainX = np.array(setof_trainX)
    setof_trainY = np.array(setof_trainY)
    setof_testX = np.array(setof_testX)
    setof_testY = np.array(setof_testY)



    '''
    normalize the data
    '''
    # iterate through each split train data set
    for row in range(np.size(setof_trainX, 0)):
        # iterate through each column of data set
        for column in range(np.size(setof_trainX[row], 1)):
            # calculate mean of current feature column using train data
            mean = np.mean(setof_trainX[row][:, column])
            # calculate std of current feature column using train data
            std = np.std(setof_trainX[row][:, column])
            # normalize train feature vector using mean and std of train column
            setof_trainX[row][:, column] = (setof_trainX[row][:, column] - mean) / std
            # normalize test feature vector using mean and std of train column
            setof_testX[row][:, column] = (setof_testX[row][:, column] - mean) / std

    '''
    Train classifier using linear regression and svm
    '''
    # logistical regression = logistic loss no regularizer
    # svm hinge loss with l2 reg
    logistical_acc = np.zeros(np.size(setof_testX,0))
    svm_acc = np.zeros(np.size(setof_testX,0))

    for i in range(np.size(setof_testX,0)):
        # train weights
        w_logistical = train_classifier(setof_trainX[i], setof_trainY[i], 0.01, logistic_loss)
        w_svm = train_classifier(setof_trainX[i], setof_trainY[i], 0.1, hinge_loss, 0.001, l1_reg)
        # predict class
        pred_y_log = test_classifier(w_logistical, setof_testX[i])
        pred_y_svm = test_classifier(w_svm, setof_testX[i])
        # calculate accuracy of each data split
        log_accuracy = accuracy(pred_y_log, setof_testY[i])
        svm_accuracy = accuracy(pred_y_svm, setof_testY[i])
        # keep track of all accuracy
        logistical_acc[i] = log_accuracy
        svm_acc[i] = svm_accuracy

    # calculate 5-fold accuracy
    fiveFoldLog = np.sum(logistical_acc)/len(logistical_acc)
    fiveFoldSVM = np.sum(svm_acc)/len(svm_acc)

    # open file
    filePerc = open("results.txt", "a")

    # print results to scree
    print('5-fold cross validation for Logistical Regression')
    print(logistical_acc)
    print(fiveFoldLog)
    print('5-fold cross validation for SVM')
    print(svm_acc)
    print(fiveFoldSVM)

    # write results to file
    filePerc.write('5-fold cross validation for Logistical Regression\n')
    filePerc.write('The results for each test: ' + str(logistical_acc) + '\n')
    filePerc.write('5-fold validation result: ' + str(fiveFoldLog) + '\n')
    filePerc.write('\n')
    filePerc.write('5-fold cross validation for SVM\n')
    filePerc.write('The results for each test: ' + str(svm_acc) + '\n')
    filePerc.write('5-fold validation result: ' + str(fiveFoldSVM) + '\n')
    filePerc.write('\n')
    filePerc.close()

    # 5-fold cross-validation
    # Note: in this assignment, preprocessing the feature values will make
    # a big difference on the accuracy. Perform feature normalization after
    # spliting the data to training and validation set. The statistics for
    # normalization would be computed on the training set and applied on
    # training and validation set afterwards.
    # TO-DO: Add your code here

    return None


if __name__ == "__main__":
    main()
