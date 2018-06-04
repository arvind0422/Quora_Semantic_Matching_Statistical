"""
EM Algorithm: Mixture of 2 Gaussians.

"""

import pandas
import numpy as np
from scipy.stats import multivariate_normal
from sys import maxsize

def distance(old_params, new_params):
  dist = 0
  for param in ['mean0', 'mean1']:
    for i in range(len(old_params)):
      dist += (old_params[param][i] - new_params[param][i]) ** 2
  return dist ** 0.5

def MeanCov(x):

    mean_vector=np.mean(x, axis=0)

    mat=np.zeros([len(mean_vector), len(mean_vector)])
    for i in range(len(x)):
        temp=x[i]-mean_vector
        mat=mat+(temp*temp.reshape([len(mean_vector), 1]))
    covariance_matrix=mat/(len(x)-1)

    return mean_vector, covariance_matrix


def Classify(test, PARAMETERS0, PARAMETERS1):
    # TODO

    f0 = multivariate_normal.pdf(x=test, mean=PARAMETERS0["mean0"], cov=PARAMETERS0["cov0"], allow_singular=True)
    f1 = multivariate_normal.pdf(x=test, mean=PARAMETERS["mean1"], cov=PARAMETERS0["cov1"], allow_singular=True)
    k0 =  f0*PARAMETERS0["lambda0"]
    k1 = f1*PARAMETERS0["lambda1"]
    q0 = k0+k1

    f0 = multivariate_normal.pdf(x=test, mean=PARAMETERS1["mean0"], cov=PARAMETERS1["cov0"], allow_singular=True)
    f1 = multivariate_normal.pdf(x=test, mean=PARAMETERS1["mean1"], cov=PARAMETERS1["cov1"], allow_singular=True)
    k0 =  f0*PARAMETERS1["lambda0"]
    k1 = f1*PARAMETERS1["lambda1"]
    q1 = k0+k1

    if q0>q1:
        return 0
    else:
        return 1

def expectation(data, answers, parameters):
  for i in range(data.shape[0]):
    x = data[i,:]
    # print(x.shape)

    # print(parameters)
    q0 = multivariate_normal.pdf(x, parameters["mean0"], parameters["cov0"], allow_singular=True) * parameters["lambda0"]
    q1 = multivariate_normal.pdf(x, parameters["mean1"], parameters["cov1"], allow_singular=True) * parameters["lambda1"]

    if q0 > q1:
      answers[i] = 0
    else:
      answers[i] = 1

  return answers

def maximization(data, answers, parameters):

    count = np.count_nonzero(answers) / len(data)
    parameters["lambda1"] = count
    parameters["lambda0"] = 1 - count
    data0 = list()
    data1 = list()
    for i in range(len(answers)):
        if answers[i]==0:
            data0.append(data[i])
        else:
            data1.append(data[i])
    data0 = np.array(data0)
    data1 = np.array(data1)
    parameters['mean0'], parameters['cov0'] = MeanCov(data0)
    parameters['mean1'], parameters['cov1'] = MeanCov(data1)
    return parameters


if __name__=="__main__":
    dataset = pandas.read_csv("train01.csv")

    num = len(dataset)
    dim = len(dataset.values[0]) - 1
    split = int(num*0.9)

    data = dataset.values[:,1:]
    train_data = data[0:split,:]

    test_data = data[split:num, :]
    answers = dataset.values[:,0]
    train_answers = answers[0:split]
    test_answers = answers[split:]

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

    EPSILON = 0.004

    CHANGE = maxsize
    ITERATIONS = 0

    PARAMETERS = {"lambda0": 0.5, "lambda1": 0.5, "mean0": np.abs(np.random.rand(dim)), "mean1": np.abs(np.random.rand(dim)), "cov0": np.diag(np.abs(np.random.rand(dim))), "cov1": np.diag(np.abs(np.random.rand(dim)))}
    RANDP = PARAMETERS.copy()
    # print(PARAMETERS)
    train_answers=np.zeros(len(class0))

    while CHANGE > EPSILON:
        ITERATIONS+=1
        train_answers = expectation(class0, train_answers.copy(), PARAMETERS)

        NEW_PARAMETERS = maximization(class0, train_answers, PARAMETERS.copy())

        CHANGE = distance(PARAMETERS, NEW_PARAMETERS)

        PARAMETERS = NEW_PARAMETERS.copy()

        print("Iteration: {}, Change: {}".format(ITERATIONS, CHANGE))

    PARAMETERS0 = PARAMETERS.copy()

    ITERATIONS=0
    CHANGE = maxsize
    PARAMETERS = RANDP.copy()
    train_answers=np.zeros(len(class1))

    while CHANGE > EPSILON:
        ITERATIONS+=1

        train_answers = expectation(class1, train_answers.copy(), PARAMETERS)

        NEW_PARAMETERS = maximization(class1, train_answers, PARAMETERS.copy())

        CHANGE = distance(PARAMETERS, NEW_PARAMETERS)

        PARAMETERS = NEW_PARAMETERS.copy()

        print("Iteration: {}, Change: {}".format(ITERATIONS, CHANGE))

    PARAMETERS1 = PARAMETERS.copy()

    """ CLASSIFICATION """

    classified = list()
    for i in range(len(test_data)):
        classified.append(Classify(test_data[i,:], PARAMETERS0, PARAMETERS1))  # TODO
    scores = np.array(classified) - dataset.values[split:num, 0]
    x = np.count_nonzero(scores)
    accuracy = 1 - (x/len(scores))
    print("Test Accuracy = "+str(round(accuracy*100, 3))+ "%")
