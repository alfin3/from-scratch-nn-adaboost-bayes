#!/usr/bin/env python
"""
bn_run.py

Implements functions for evaluating and running Bayesian network experiments.

Python 2.7.12
"""

import bn
import csv
import data_process
import graphs_metaxl

def count_params(bnet, cValues):
    """Returns the number of parameters necessary to represent the joint
    probabilty distribution of the data, according to the conditional independence
    assumtions of the Bayes net bn. cValues is the number of values that each
    random variable in bn can take.
    bnet = bn.bn(graphs_metaxl.NAIVE_BAYES, [0,1])
    count_params(bnet, len([0,1]))
    19
    bnet = bn.bn(graphs_metaxl.MET_AKT_BAYES, [0, 1, 2])
    count_params(bnet, len([0,1,2]))
    104
    """
    dictProbTable = bnet.dictProbTable
    cStoredParameters = sum(map(lambda ixTuple: len(dictProbTable[ixTuple].keys()),
                         dictProbTable.keys()))
    return cStoredParameters - cStoredParameters / cValues
                         
def fold_size(listInstances, iFold):
    """Returns the size of the regular fold and the size of the last fold. 
    iFold must be larger than 1 for cross-validation purposes.
    data = data_process.parse_input(open('metaxl_bin_mix.csv', "r"), int, 112)
    fold_size(data, 10)
    (11, 13)
    """
    if iFold <= 1:
        raise ValueError("The number of folds must be larger than 1 for cross-validation.")
    cInstances = len(listInstances)
    cResInstances = cInstances % iFold
    iFoldSize = (cInstances  - cResInstances) / iFold
    iFoldSizeLast = iFoldSize + cResInstances
    return iFoldSize, iFoldSizeLast
    
def test_set_indeces(iFold, iFoldSize, iFoldSizeLast):
    """Returns the list of index pairs for starting and ending instances
    of testing sets in a iFold-fold cross-validation experiment.
    data = data_process.parse_input(open('metaxl_bin_mix.csv', "r"), int, 112)
    iFoldSize, iFoldSizeLast = fold_size(data, 10)
    test_set_indeces(10, iFoldSize, iFoldSizeLast)
    [[0, 10], [11, 21], [22, 32], [33, 43], [44, 54], [55, 65], [66, 76],
    [77, 87], [88, 98], [99, 111]]
    """
    startIndex = 0
    endIndex = startIndex + iFoldSize - 1
    listIndexPairs = [[startIndex, endIndex]]
    for ixFold in range(1, iFold - 1):
        startIndex = ixFold * iFoldSize
        endIndex = startIndex + iFoldSize - 1
        listIndexPairs.append([startIndex, endIndex])  
    listIndexPairs.append([endIndex + 1, endIndex + iFoldSizeLast])
    return listIndexPairs

def class_accuracy(bnet, listInstances, ixClass):
    """Returns the classification accuracy of the bnet on listInstances.
    bnet = bn.bn(graphs_metaxl.NAIVE_BAYES, [0,1])
    data = data_process.parse_input(open('metaxl_bin_mix.csv', "r"), int, 112)
    bnet.learn_ML(data)
    class_accuracy(bnet, data, graphs_metaxl.IX_CLASS)
    0.7321428571428571
    """
    cCorrect = 0
    cInstances = len(listInstances)       
    for inst in listInstances:
        if inst[ixClass] == bnet.classify(inst[: -1]):
             cCorrect += 1
    return cCorrect / float(cInstances)

def cross_validation(bnet, listInstances, iFold, ixClass):
    """Performs an iFold-fold cross-validation experiment on the Bayes net bnet
    and the dataset listInstances. Returns the average classification accuracy 
    of the bnet classifier. The data set listInstances must already pre-processed.
    bnet = bn.bn(graphs_metaxl.NAIVE_BAYES, [0,1])
    data = data_process.parse_input(open('metaxl_bin_mix.csv', "r"), int, 112)
    cross_validation(bnet, data, 3, graphs_metaxl.IX_CLASS)
    (0.6963015647226175, 0.7366366366366366)
    """
    if iFold <= 1:
        raise ValueError("Need at least two folds for cross-validation.")
    iFoldSize, iFoldSizeLast = fold_size(listInstances, iFold)
    listIndexPairs = test_set_indeces(iFold, iFoldSize, iFoldSizeLast)
    dblTestCumAccuracy = 0.0
    dblTrainCumAccuracy = 0.0
    for ixTestStart, ixTestEnd in listIndexPairs:
        bnet.clean_dict()
        testData = listInstances[ixTestStart : (ixTestEnd + 1)]
        trainData = listInstances[: ixTestStart] + listInstances[(ixTestEnd + 1) :]
        bnet.learn_ML(trainData)
        #print class_accuracy(bnet, testData, ixClass)
        dblTestCumAccuracy += class_accuracy(bnet, testData, ixClass)
        dblTrainCumAccuracy += class_accuracy(bnet, trainData, ixClass)   
    return (dblTestCumAccuracy/iFold, dblTrainCumAccuracy/iFold)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
#++++++++++++++++++++++++++++++Run Bayes net experiments+++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
def experiment():
    """Run Bayes net experiments."""
    iFold = 10
    listValues = [0,1]
    ixClass = graphs_metaxl.IX_CLASS
    writer = csv.writer(open("bnet_results2.csv", "wb"))
    listInstances = data_process.parse_input(open('metaxl_bin_mix.csv', "r"), 
                                             int, 112)
    for listTuplesRvs in graphs_metaxl.LIST_NETS:
        bnet = bn.bn(listTuplesRvs, listValues)
        cParameters = count_params(bnet, len(listValues))
        dblTestAcc, dblTrainAcc = cross_validation(bnet, listInstances, 
                                                   iFold, ixClass)
        writer.writerow([cParameters, dblTestAcc, dblTrainAcc])                                        
    return


    


    
    
            
        
