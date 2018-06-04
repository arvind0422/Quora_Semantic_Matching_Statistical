"""
Maximum Likelihood Estimate with Multivariate Gaussian Distribution
Parameters Estimated: Mean Vector, Covariance Matrix.

"""

import numpy as np
import pandas
from scipy.stats import multivariate_normal


def MLE(x):

    mean_vector=np.mean(x, axis=0)

    mat=np.zeros([len(mean_vector), len(mean_vector)])
    for i in range(len(x)):
        temp=x[i]-mean_vector
        mat=mat+(temp*temp.reshape([len(mean_vector), 1]))
    covariance_matrix=mat/len(x)

    return mean_vector, covariance_matrix


def Classify(mean0, mean1, cov0, cov1, p0, p1, test):

    f0 = multivariate_normal.pdf(x=test, mean=mean0, cov=cov0, allow_singular=True)
    f1 = multivariate_normal.pdf(x=test, mean=mean1, cov=cov1, allow_singular=True)

    q0 = f0 * p0
    q1 = f1 * p1

    if q0 > q1:
        return 0
    else:
        return 1


if __name__ == "__main__":
    dataset = pandas.read_csv("train01.csv")

    num = len(dataset)
    dim = len(dataset.values[0]) - 1
    split = int(num*0.9)

    data = dataset.values[:,1:dim]
    train_data = data[0:split,:]
    test_data = data[split:num, :]
    answers = dataset.values[:,0]

    class0 = list()
    for i in range(len(train_data)):
        if answers[i] == 0:
            class0.append(train_data[i, :])
    class0 = np.array(class0)
    prior0 = len(class0)/split

    class1 = list()
    for i in range(len(train_data)):
        if answers[i] == 1:
            class1.append(train_data[i, :])
    class1 = np.array(class1)
    prior1 = len(class1)/split

    """ TRAINING """

    mean0, cov0 = MLE(class0)
    mean1, cov1 = MLE(class1)

    n = 1
    m = 0
    mu0 = np.abs(np.random.randn(len(mean0)))

    newmean0 = ((n*mean0) + (m*mu0))/(n+m)
    newcov0 = cov0/(n+m)

    newmean1 = ((n*mean1) + (m*mu0))/(n+m)
    newcov1 = cov1/(n+m)

    """ CLASSIFYING """

    classified = list()
    for i in range(len(test_data)):
        classified.append(Classify(newmean0, newmean1, newcov0, newcov1, prior0, prior1, test_data[i,:]))
    scores = np.array(classified) - dataset.values[split:num, 0]
    x = np.count_nonzero(scores)
    accuracy = 1 - (x/len(scores))
    print("Test Accuracy = "+str(round(accuracy*100, 3))+ "%")

    classified = list()
    for i in range(len(train_data)):
        classified.append(Classify(mean0, mean1, cov0, cov1, prior0, prior1, train_data[i, :]))
    scores = np.array(classified) - answers[0:split]
    x = np.count_nonzero(scores)
    accuracy = 1 - (x/len(scores))
    print("Training Accuracy = "+str(round(accuracy*100, 3))+ "%")
