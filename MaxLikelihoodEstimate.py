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
    covariance_matrix=mat/(len(x)-1)

    return mean_vector, covariance_matrix


def Classify(mean0, mean1, cov0, cov1, p0, p1, test):

    f0 = multivariate_normal.pdf(x=test, mean=mean0, cov=cov0, allow_singular=True)
    f1 = multivariate_normal.pdf(x=test, mean=mean1, cov=cov1, allow_singular=True)

    # print(f0, f1)

    # u,s,v = np.linalg.svd(cov0, full_matrices=False)
    # cov0d = np.diag(s)
    # den0 = math.log(math.sqrt(math.pow(2 * math.pi, len(test)) * np.linalg.det(cov0d)))
    # cov0in = np.matmul(np.matmul(v.T, np.linalg.pinv(cov0d)), u.T)
    # eigen0 = np.linalg.eig(cov0)[0]
    # pseudodet0 = np.product(eigen0[eigen0 > 1e-12])
    # den0 = math.log(math.sqrt(math.pow(2 * math.pi, len(test)) * pseudodet0))
    # pseudoinv0 = np.linalg.pinv(cov0)
    # diff0 = np.atleast_2d(test-mean0)
    # e0 = (-0.5)*np.matmul(np.matmul(diff0, pseudoinv0), diff0.T)[0][0]
    # f0l = e0 - den0
    # f0 = math.exp(f0l)
    #
    # u,s,v = np.linalg.svd(cov1, full_matrices=False)
    # cov1d = np.diag(s)
    # den1 = math.log(math.sqrt(math.pow(2 * math.pi, len(test)) * np.linalg.det(cov1d)))
    # cov1in = np.matmul(np.matmul(v.T, np.linalg.pinv(cov1d)), u.T)
    # eigen1 = np.linalg.eig(cov1)[0]
    # pseudodet1 = np.product(eigen1[eigen1 > 1e-12])
    # den1 = math.log(math.sqrt(math.pow(2 * math.pi, len(test)) * pseudodet1))
    # pseudoinv1 = np.linalg.pinv(cov1)
    # diff1 = np.atleast_2d(test-mean1)
    # e1 = (-0.5)*np.matmul(np.matmul(diff1, pseudoinv1), diff1.T)[0][0]
    # f1l = e1 - den1
    # f1 = math.exp(f1l)

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
    split = int(num*0.75)

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

    print("Class 0")
    print("Mean")
    print(mean0)
    print("Class 0")
    print("Covariance Matrix")
    print(cov0)
    print("Class 1")
    print("Mean")
    print(mean1)
    print("Class 1")
    print("Covariance Matrix")
    print(cov1)

    """ CLASSIFYING """

    classified = list()
    for i in range(len(test_data)):
        classified.append(Classify(mean0, mean1, cov0, cov1, prior0, prior1, test_data[i,:]))
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
