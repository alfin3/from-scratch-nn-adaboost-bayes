#!/usr/bin/env python
"""
analysis.py

Implements the plotting of the data from Bayes net experiments.

Python 2.7.12
"""
import matplotlib.pyplot as plt 
import os.path

basepath = os.path.dirname(__file__)
file8bnets = os.path.abspath(os.path.join(basepath, 'bnet_results.csv'))
            
def parse_ix_input(dataFile, fn_type, numExamples, ix):
    """
    Parses the data file and creates a list of values of the index ix across all
    instances of the data set in the dataFile.
    lst = parse_ix_input(open('bnet_results.csv'), int, 8, 1)
    """
    lstData = []
    ct = 0
    for line in dataFile:
        instance = line.split(",")
        lstData.append(fn_type(instance[ix]))
        ct += 1
        if not numExamples is None and ct >= numExamples:
          break
    return lstData

def subtr_from(lstValues, maxVal = 1.0):
    """
    Returns a list with values subtracted from maxVal.
    """
    return map(lambda val: maxVal - val, lstValues)

lstParameters = parse_ix_input(open(file8bnets), int, 8, 0)     
lstTest = parse_ix_input(open(file8bnets), float, 8, 1)  
lstTraining = parse_ix_input(open(file8bnets), float, 8, 2)

labels = lstParameters
t = range(0, 8)
line = plt.plot(t, subtr_from(lstTest), 'r-', marker='8', label='Test')
plt.setp(line, linewidth=0.5)
line = plt.plot(t, subtr_from(lstTraining), 'b-', marker='s', label='Training')
plt.setp(line, linewidth=0.5)
plt.xticks(t, labels)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
leg = plt.legend(fontsize = 'small')
leg.get_frame().set_linewidth(0.0)
plt.axis([0, 7, 0.14, 0.35])
plt.xlabel('# Of Parameters')
plt.ylabel('Average Classification Error In 10-Fold CV')
plt.title('Bayes Nets - Model Selection Experiment')       
plt.savefig('Bnet_Model_Selection', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
plt.show() 

