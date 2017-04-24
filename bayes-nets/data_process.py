#!/usr/bin/env python
"""
data_process.py

Implements data pre-processing functions.

Python 2.7.12
"""

import csv

def parse_input(dataFile, fn_type, numExamples):
    """
    Parses the data file and creates a list data set of numExamples instances.
    lstData = parse_input(open('metaxl.csv', "r"), float, 112)
    """
    lstData = []
    ct = 0
    for line in dataFile:
        instance = line.split(",")
        lstData.append(map(lambda x:fn_type(x),instance))
        ct += 1
        if not numExamples is None and ct >= numExamples:
          break
    return lstData

def median(listValues):
    """Calculates the median from the list of values
    listValues = [1, 4, 3, 5, 6]
    median(listValues)
    4
    listValues = [3, 3, 3, 6, 6, 6]
    median(listValues)
    4.5
    """
    length = len(listValues)
    if length < 1:
        return None
    listSortedValues = sorted(listValues)
    if length%2 == 1:
        return listSortedValues[(length - 1) / 2]
    else:
        return (listSortedValues[length / 2 - 1] + listSortedValues[length / 2]) / 2.0

def thresholds(listFloatData, fn_thresh):
    """Returns the list of thresholds for binarization of float data, one 
    threshold value for each column of the data.
    listFloatData = parse_input(open('metaxl.csv', "r"), float, 112)
    listThresholds = thresholds(listFloatData, median)
    """
    listThresholds = []
    for ix in range(len(listFloatData[0])):
        listValues = map(lambda floatInst: floatInst[ix], listFloatData)
        listThresholds.append(fn_thresh(listValues))
    return listThresholds

def bin_data(listFloatData, listThresholds, fileName):
    """Creates a csv file with the binarized data. Boolean values are derived 
    according to the thresholds for each column of the data.
    listFloatData = parse_input(open('metaxl.csv', "r"), float, 112)
    listThresholds = thresholds(listFloatData, median)
    bin_data(listFloatData, listThresholds, 'metaxl_bin.csv')
    """
    writer = csv.writer(open(fileName, "wb"))
    def compare(val, thresh):
        if (val >= thresh):
            return 1
        return 0
    for inst in listFloatData:
        binInst = map(lambda val, thresh: compare(val, thresh), inst, listThresholds)
        writer.writerow(binInst)
    return

def mix_balanced_data(listData, listClassValues, ixClass, fileName):
    """Assumes i)discrete variable ixClass, ii) the data set inlistSourceData 
    is balanced. Creates a data set in fileName with instances of the 
    different classes interchanged for crossvalidation experiments 
    on balanced data.
    listBoolData = parse_input(open('metaxl_bin.csv', "r"), int, 112)
    mix_balanced_data(listBoolData, [1,0], 9, 'metaxl_bin_mix.csv')
    """
    dictClassVal_Instances = {}
    writer = csv.writer(open(fileName, "wb"))
    for classVal in listClassValues:
        dictClassVal_Instances[classVal] = []        
    for inst in listData:
        dictClassVal_Instances[inst[ixClass]].append(inst)
    # here balanced data assumption
    for ixInst in range(len(dictClassVal_Instances[listClassValues[0]])):
        for classVal in listClassValues:
            inst = dictClassVal_Instances[classVal][ixInst]
            writer.writerow(inst)
    return
