'''
    Creating classifiers, comparing their effectiveness at classifying data, and observing how their performance changes with different parameters
    Created on April 7th, 2017
    @author: Ashka Stephen
'''
import numpy
from classifiers import create_decision_tree, create_random_forest, calculate_model_accuracy, calculate_confusion_matrix
from data import get_minecraft, get_first_n_samples
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression

def p0(featuretype='histogram'):
    data_train, data_test, target_train, target_test = get_minecraft(featuretype)
    model = create_decision_tree()
    # using training set for fitting data
    model.fit(data_train, target_train)
    # Use the model's predict method to predict labels for the training and test sets
    predict_train = model.predict(data_train)
    predict_test = model.predict(data_test)

    accuracy_train, accuracy_test = calculate_model_accuracy(predict_train, predict_test, target_train, target_test)
    print('Training accuracy: {0:3f}, Test accuracy: {1:3f}'.format(accuracy_train, accuracy_test))

    cfm = calculate_confusion_matrix(predict_test,target_test)
    print "Confusion matrix"
    print cfm


    for q in range(1,3):
        for p in range(0,q):
            #compute confusion between classes p and q
            index_pq = [i for i,v in enumerate(target_train) if v in [p,q]]
            modelpq = create_decision_tree()
            #fit model to the data only involving classes p and q
            data_trainPQ =[]
            data_testPQ =[]
            target_trainPQ =[]
            target_testPQ =[]
            for each in index_pq:
            	if each < len(data_train):
            		data_trainPQ.append(data_train[each])
            	if each < len(target_train):
            		target_trainPQ.append(target_train[each])
            model.fit(data_trainPQ, target_trainPQ)

            testindex_pq = [i for i,v in enumerate(target_test) if v in [p,q]]
            for each in testindex_pq:
            	if each < len(data_test):
            		data_testPQ.append(data_test[each])
            	if each < len(target_test):
            		target_testPQ.append(target_test[each])
            modelpq.fit(data_testPQ, target_testPQ)


            predict_trainPQ = model.predict(data_trainPQ)
            predict_testPQ = model.predict(data_testPQ)

            accuracy_trainPQ, accuracy_pq = calculate_model_accuracy(predict_trainPQ, predict_testPQ, target_trainPQ, target_testPQ)
            print "One-vs-one accuracy between classes",p,"and",q,":",accuracy_pq

    return model, predict_train, predict_test, accuracy_train, accuracy_test


def p1():
    #ompare different feature types
    #m,ptrain,ptest,atrain,atest = p0('histogram')
    #m,ptrain,ptest,atrain,atest = p0('rgb')
    m,ptrain,ptest,atrain,atest = p0('gray')

def p2():
    results = []
    model = create_decision_tree()

    #Get the Minecraft dataset using get_minecraft() and create a decision tree
    data_train, data_test, target_train, target_test = get_minecraft('histogram')
    dataTrain = []
    targetTrain = []
    dataTest = []
    targetTest = []

    for n in [50, 100, 150, 200, 250]:
    	dataTrain, targetTrain = get_first_n_samples(data_train, target_train, n)
    	dataTest, targetTest = get_first_n_samples(data_test, target_test, n)
    	model.fit(dataTrain, targetTrain)
        predict_trainz = model.predict(dataTrain)
        predict_testz = model.predict(dataTest)
    	accuracy_train_n, accuracy_test = calculate_model_accuracy(predict_trainz, predict_testz, targetTrain, targetTest)
        results.append((n, accuracy_train_n, accuracy_test))

    print(results)
    return model, results


def p3():
    results = []
    # Get the Minecraft dataset
    data_train, data_test, target_train, target_test = get_minecraft('histogram')

    for n_estimators in [2, 5, 10, 20, 30]:
        # create a random forest classifier with n_estimators estimators
        model = create_random_forest(n_estimators)
        #use the model to fit the training data and predict labels for the training and test data
    	model.fit(data_train, target_train)
        predict_trainz = model.predict(data_train)
        predict_testz = model.predict(data_test)
        # calculate the accuracies of the models and add them to the results
        accuracy_train, accuracy_test = calculate_model_accuracy(predict_trainz, predict_testz, target_train, target_test)
        results.append((n_estimators, accuracy_train, accuracy_test))
    print(results)
    return model, results


def bonus():
    results = []
    data_train, data_test, target_train, target_test = get_minecraft("histogram")
    model = LogisticRegression()

    model.fit(data_train, target_train)
    predict_train = model.predict(data_train)
    predict_test = model.predict(data_test)

    accuracy_train, accuracy_test = calculate_model_accuracy(predict_train, predict_test, target_train, target_test)
    print('Training accuracy: {0:3f}, Test accuracy: {1:3f}'.format(accuracy_train, accuracy_test))

    cfm = calculate_confusion_matrix(predict_test,target_test)
    for q in range(1,3):
        for p in range(0,q):
            index_pq = [i for i,v in enumerate(target_train) if v in [p,q]]
            modelpq = create_decision_tree()
            data_trainPQ =[]
            data_testPQ =[]
            target_trainPQ =[]
            target_testPQ =[]
            for each in index_pq:
            	if each < len(data_train):
            		data_trainPQ.append(data_train[each])
            	if each < len(target_train):
            		target_trainPQ.append(target_train[each])
            model.fit(data_trainPQ, target_trainPQ)

            testindex_pq = [i for i,v in enumerate(target_test) if v in [p,q]]
            for each in testindex_pq:
            	if each < len(data_test):
            		data_testPQ.append(data_test[each])
            	if each < len(target_test):
            		target_testPQ.append(target_test[each])
            modelpq.fit(data_testPQ, target_testPQ)


            predict_trainPQ = model.predict(data_trainPQ)
            predict_testPQ = model.predict(data_testPQ)

            accuracy_trainPQ, accuracy_pq = calculate_model_accuracy(predict_trainPQ, predict_testPQ, target_trainPQ, target_testPQ)
            print "One-vs-one accuracy between classes",p,"and",q,":",accuracy_pq
    return model, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", type=str, choices=['p0', 'p1', 'p2', 'p3', 'bonus'], help="The problem to run")
    args = parser.parse_args()

    if args.problem == 'p0':
        p0()
    elif args.problem == 'p1':
        p1()
    elif args.problem == 'p2':
        p2()
    elif args.problem == 'p3':
        p3()
    elif args.problem == 'bonus':
        bonus()
