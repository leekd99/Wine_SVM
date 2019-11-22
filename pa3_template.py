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
    myLoss = np.log(1 + np.exp(-np.multiply(train_y, pred_y)))

    return None

def hinge_loss(train_y, pred_y):
    
	# TO-DO: Add your code here

    return None

'''
The regularizers shall compute the loss without considering the bias term in the weights
'''
def l1_reg(w):

    # TO-DO: Add your code here

    # TO-DO remove bias term
    myL1 = np.mean(np.absolute(w))

    return myL1

def l2_reg(w):

    # TO-DO: Add your code here
    myL2 = np.mean(np.square(w))

    return myL2

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

    #print(dataX)
    #print('#####################')
    #print(dataY)

    x = np.array([2,3,4,2])
    y = np.array([2,3,4,5])

    #print(np.multiply(x,y))
    print(np.log(np.exp(-(1+x))))
    print(dataX)

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
