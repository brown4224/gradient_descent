#Sean McGlincy
# Machine Learning
# HW 2
import numpy as np
# Requires
# Linux:  sudo yum install tkinter python36u-tkinter

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def CalculateResults(X, Y, b, threshold):
    result_opt = Classify_Prediction(X, b, threshold)  # Make prediction
    return sum(result_opt == Y) / len(Y)  # Classify accuracy

def Classify_Prediction(X, b, threashold):
    return np.array(np.dot(X, b) > threashold)

def NormalizeData(data):
    # arr = [[255] * len(data[0])] * len(data)
    return data / 255  #image data 255 max.  Using min-max normailization

# Use equation from class, Don't re-invent wheel.  Code from Dr. Kang's Example:  http://ksuweb.kennesaw.edu/~mkang9/teaching/CS7267/code3/Linear_Regression.html
def LinearRegression(X, y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), y)

def GradDecent(X, y, b):
    return  -np.dot(X.transpose(), y) + np.dot(np.dot(X.transpose(), X) ,b)

def Cost(X, y, b):
    return np.sum(np.dot(X, b) - y)**2



def DisplayGraph(plotter):
    # Output Graph.  Flags for PyCharm
    plt.plot(plotter)
    plt.interactive(False)
    plt.show(block=True)

#####################################################################################################
#######################################  Main   #####################################################
#####################################################################################################
#  Main Start Program HERE
# Import data as Integers and remove labels
training_data = np.genfromtxt('MNIST_training.csv', delimiter=',', dtype=int, skip_header=1)
test_data = np.genfromtxt('MNIST_test.csv', delimiter=',', dtype=int, skip_header=1)

# Make array of labels
y_training = np.array(training_data[:, 0])
y_test = np.array(test_data[:, 0])

# Make array of data
x_training = np.array(training_data[:, 1:])
x_test = np.array(test_data[:, 1:])


# Normalize Data Min-Max
x_training = NormalizeData(x_training)
x_test = NormalizeData(x_test)

# Threshold for Classes
threshold = 0.5

#  Linear Regression
b_opt = LinearRegression(x_training, y_training)  # Calculate b
accuracy_opt = CalculateResults(x_test,y_test, b_opt, threshold)

#   Gradient Decent
b_est = np.zeros(len(x_training[0])) # Array of zeros for each feature
learn_rate = 1e-4
r = 100

aggregate = []
for i in range(0, r):
    # Iterativly Calculates b_est based on learning rate and gradient decent
    b_est  = b_est - learn_rate * GradDecent(x_training, y_training, b_est)

    # Calculate cost for graph
    cost = Cost(x_training, y_training, b_est)
    aggregate.append(cost)


accuracy_est = CalculateResults(x_test,y_test, b_est, threshold)

# Display Accuracy
bdiff = sum(abs(b_opt - b_est))
print("b_opt: ", b_opt)
print("b_est: ", b_est)
print("b_opt accuracy", accuracy_opt)
print("b_est accuracy", accuracy_est)
print("B Diff: ", bdiff)

DisplayGraph(aggregate)