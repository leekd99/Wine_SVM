'''
Add the code below the TO-DO statements to finish the assignment. Keep the interfaces
of the provided functions unchanged. Change the returned values of these functions
so that they are consistent with the assignment instructions. Include additional import
statements and functions if necessary.
'''

import csv
import numpy as np

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
    # TO-DO: Add your code here
    # L(y,f(x)) = log(1+exp(-y*f(x))) where y is your label and f(x) is your predicted value
    # argmin sum(i,n)(yi,f(xi))

    # multiply train_y and pred_y component wise then multiply each component by minus 1

    return np.sum(np.log(1 + np.exp(-np.multiply(train_y, pred_y)))) / len(train_y)


def hinge_loss(train_y, pred_y):
    # TO-DO: Add your code here
    # np.sum(np.maximum(np.zeros(len(train_y)), 1-np.multiply(train_y, pred_y)))/len(train_y)
    print(1-np.multiply(train_y, pred_y))


    return np.sum(np.maximum(np.zeros(len(train_y)), 1-np.multiply(train_y, pred_y)))/len(train_y)

'''
The regularizers shall compute the loss without considering the bias term in the weights
'''


def l1_reg(w):
    # TO-DO: Add your code here
    # Remove the bias term from w vector and calculate l1 norm

    return np.linalg.norm(w[1:], 1)


def l2_reg(w):
    # TO-DO: Add your code here

    return np.dot(w[1:], w[1:])


def train_classifier(train_x, train_y, learn_rate, loss, lambda_val=None, regularizer=None):
    # TO-DO: Add your code here

    return None


def test_classifier(w, test_x):
    # TO-DO: Add your code here

    return None


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

    # print(dataX)
    # print('#####################')
    # print(dataY)

    x = np.array([2, -3, 4, 2])
    y = np.array([-2, 3, -4, 5])
    test = np.array([3, 4])

    for i in range(len(x)):
        break
        aug = np.array(x)
        aug[i] += 1
        print('aug')
        print(aug)
        print('x')
        print(x)

    # print(np.multiply(x,y))
    # print(np.log(np.exp(-(1+x))))
    # print(dataX)

    # print(l1_reg(x))
    # print(l2_reg(x))

    # print(np.sum(np.log(1 + np.exp(-np.multiply(x, y))))/len(y))
    # print(dataX)
    # print(dataY)

    # print(np.maximum(np.zeros(len(x)), 1-x))

    # print(1-x)

    print(hinge_loss(x, y))

    # print(x[1:])

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
