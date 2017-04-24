#!/usr/bin/env python
"""
bn.py

Implements Bayesian networks to represent joint distributions of
discrete random variables, where all random variable can take the same set 
of values (current version).

Python 2.7.12
A 'from scratch' implementation.
"""

import graphs_metaxl
import data_process

class bn(object):
    """"""
    def __init__(self, listTuplesRvs, listValues):
        """Arguments:
        listTuplesRvs: the list of tuples of random variables. Random variable 
        names correspond to the indexes of the columns of the data set.
        In each tuple, the last entry is the conditioned random variable. 
        All previous entries are conditioning random variables. 
        For example: P(Y| X2, X1, X0) is represented as (0,1,2,9), where the 
        index of Y is 9.
        listValues: the list of all discrete values that each random variable 
        can take. In this version, these values must be the same across all 
        random variables of the network, e.g.[0,1] or [0,1,2] etc...
        bnet = bn(graphs_metaxl.NAIVE_BAYES, [0,1])
        data = data_process.parse_input(open('metaxl_bin_mix.csv', "r"), int, 112)
        bnet.learn_ML(data)
        bnet.get_prob((9,),(0,))
        0.5
        bnet.get_prob((9,0),(0,1))
        0.5892857142857143
        """
        self.MLincomplete = True
        self.dictProbTable = initialize_bn(listTuplesRvs, listValues)
        self.listTuplesRvs = listTuplesRvs
        self.listValues = listValues
        
    def get_prob(self, tuplesRvs, tupleValues):
        """Returns the probability stored in the corresponding conditional 
        probability table, if Maximum Likelihood is completed.
        """
        if self.MLincomplete:
            raise ValueError("ML incomplete")
        return self.dictProbTable[tuplesRvs][tupleValues]

    def learn_ML(self, listInstances):
        """Runs the Maxumum Likelihood algorithm to learn the conditional 
        probability tables."""
        self.dictProbTable = learn_ML(self.dictProbTable, listInstances)
        self.MLincomplete = False
        return

    def classify(self, listFeatures):
        """Returns the most likely class value corresponding to the values
        of input features."""
        if self.MLincomplete:
            raise ValueError("ML incomplete")
        return classify(self.dictProbTable, listFeatures, self.listValues)
    
    def class_belief_state(self, listFeatures):
        """Returns the class belief state corresponding to the values
        of input features."""
        if self.MLincomplete:
            raise ValueError("ML incomplete")
        return class_belief_state(self.dictProbTable, listFeatures, self.listValues)
    
    def save_dict(self, strFileName):
        """Saves the dictionary of conditional probability tables to file.
        """
        file(strFileName, 'w').write(repr(self.dictProbTable))
        return
        
    def load_dict(self, strFileName):
        """Loads a dictionary of conditioanl probability tables from file. 
        Must be compatible with the initialized Bayes net"""
        temp = file(strFileName, 'r').read()
        self.dictProbTable = eval(temp)
        self.MLincomplete = False
        return
    
    def clean_dict(self):
        """Re-initializes all probability values in the dictionary of conditional
        probability talbes to 0.0"""
        self.dictProbTable = initialize_bn(self.listTuplesRvs, self.listValues)
        self.MLincomplete = True

def get_val_tuples(listRvs, listValues):
    """Returns a list of tuples of all possible values of the random variables 
    in listRvs.
    get_val_tuples([1], [0,1])
    [(0,), (1,)]
    get_val_tuples([1,2,3], [0,1])
    [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    get_val_tuples([1,2,5], [0,3])
    [(0, 0, 0), (0, 0, 3), (0, 3, 0), (0, 3, 3), 
    (3, 0, 0), (3, 0, 3), (3, 3, 0), (3, 3, 3)]
    """
    if len(listRvs) == 0 or len(listValues) == 0:
        raise ValueError('Cannot be length 0')
    listCurrent = map(lambda val: [val], listValues)
    for _ in range(len(listRvs)-1):
        listCurrent = reduce(lambda h, t: h + t, 
                             map(lambda cur: map(lambda val: cur + [val], 
                                                 listValues), listCurrent), [])
    return map(tuple, listCurrent)

def initialize_bn(listTuplesRvs, listValues):
    """
    Returns the dictionary of the prior and conditional probabilities of the
    Bayes net, encoded in listTuplesRvs. Each random variable can take values in
    listValues. All probabilities are initialized to 0.0.
    initialize_bn(graphs_metaxl.NAIVE_BAYES, [0,1,2])
    initialize_bn(graphs_metaxl.NAIVE_BAYES, [0])
    """
    dictProbTable = {}
    for tupleRvs in listTuplesRvs:
        dictProbTable[tupleRvs] = {}
        listRvs = list(tupleRvs)
        for tupleValues in get_val_tuples(listRvs, listValues):
            dictProbTable[tupleRvs][tupleValues] = 0.0
    return dictProbTable

def get_all_rvs(listTuplesRvs):
    """Gets all distinct random variables from the list of tuples that defines 
    a Bayes net.
    get_all_rvs(graphs_metaxl.NAIVE_BAYES)
    [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    """
    listDistinctRvs = []
    for tupleRvs in listTuplesRvs:
        for rv in tupleRvs:
            if not(rv in listDistinctRvs):
                listDistinctRvs.append(rv)
    return listDistinctRvs 

def count_matches(listRvs, listValuesRvs, listInstances):
    """Counts the number of instances containing the given random variables and 
    at given values and returns the resulting integer count.
    data = data_process.parse_input(open('metaxl_bin_mix.csv', "r"), int, 112)
    count_matches([0], [0], data)
    56
    count_matches([0,1], [0,1], data)
    10
    """
    cMatches = 0
    for inst in  listInstances:
        if reduce (lambda h, t: h and t,
               map(lambda rv, valueRv: inst[rv] == valueRv,
                   listRvs, listValuesRvs), True):
           cMatches += 1
    return cMatches             

def learn_ML(dictProbTable, listInstances):
    """Learns the conditional probability tables from data with the 
    Maximum Likelihood method. Returns the resulting dictionary of conditional 
    probability tables. 
    listInstances = data_process.parse_input(open('metaxl_bin_mix.csv', "r"), int, 112)
    dictProbTable = initialize_bn(graphs_metaxl.NAIVE_BAYES, [0,1])
    learn_ML(dictProbTable , listInstances)
    {(9,): {(0,): 0.5, (1,): 0.5},
     (9, 0): {(0, 0): 0.4107142857142857,
              (0, 1): 0.5892857142857143,
              (1, 0): 0.5892857142857143,
              (1, 1): 0.4107142857142857},
     (9, 1): {(0, 0): 0.48214285714285715,
              (0, 1): 0.5178571428571429,
              (1, 0): 0.5178571428571429,
              (1, 1): 0.48214285714285715},
    etc...
    """
    #check for inconsistencies
    if len(get_all_rvs(dictProbTable.keys())) > len(listInstances[0]):
        raise ValueError("Data must have no latent varialbes.")    
    maxIx = max(map(max,dictProbTable.keys()))
    if maxIx > len(listInstances[0]):\
       raise ValueError("Indices larger than dimensionality of data")
    #ML Algorithm
    for inst in listInstances:
        for tupleRvs in dictProbTable.keys():
            listRvs =list(tupleRvs)
            listValuesRvs = map(lambda rv: inst[rv], listRvs)
            tupleValues = tuple(listValuesRvs)   
            #compute probability 
            if not(dictProbTable[tupleRvs][tupleValues] > 0.0):
                cTargets = count_matches(listRvs, listValuesRvs, listInstances)
                cConditionals = count_matches(listRvs[: -1], 
                                              listValuesRvs[: -1], listInstances)
                dictProbTable[tupleRvs][tupleValues] = cTargets / float(cConditionals)
    return dictProbTable
        
def joint_prob(dictProbTable, inst):
    """Computes the joint probability of the instance.
    listInstances = data_process.parse_input(open('metaxl_bin_mix.csv', "r"), int, 112)
    dictProbTable = initialize_bn(graphs_metaxl.NAIVE_BAYES, [0,1])
    dictLearnedProbTable = learn_ML(dictProbTable , listInstances)
    joint_prob(dictProbTable, listInstances[0])
    0.0017286523043710963
    """
    dblJointProb = 1.0
    for tupleRvs in dictProbTable.keys():
        listRvs =list(tupleRvs)
        listValuesRvs = map(lambda rv: inst[rv], listRvs)
        tupleValues = tuple(listValuesRvs)
        dblJointProb *= dictProbTable[tupleRvs][tupleValues]
    return dblJointProb

def class_belief_state(dictProbTable, listFeatures, listClassValues):
    """computes a dictionary that maps class values to conditional 
    probabilities P(ClassValues|Features).
    listInstances = data_process.parse_input(open('metaxl_bin_mix.csv', "r"), int, 112)
    dictProbTable = initialize_bn(graphs_metaxl.NAIVE_BAYES, [0,1])
    dictLearnedProbTable = learn_ML(dictProbTable , listInstances)
    class_belief_state(dictProbTable, listInstances[0][: -1], [0,1])
    {0: 0.11884648995048058, 1: 0.8811535100495195}
    """
    b = {}
    for classVal in listClassValues:
        inst = listFeatures + [classVal]
        b[classVal] = joint_prob(dictProbTable, inst)
    dblNormalization = sum(b.values())
    for classVal in listClassValues:
        b[classVal] = b[classVal] / dblNormalization
    return b
        
def classify(dictProbTable, listFeatures, listClassValues):
    """Given a set of feature values, returns the class value 
    of the belief state corresponding to maximal conditional probability."""
    b = class_belief_state(dictProbTable, listFeatures, listClassValues)
    return max([(b[classVal],  classVal) for classVal in b.keys()])[1]

    







            
            
            
                  


                



                               
                    
                               


            



       




