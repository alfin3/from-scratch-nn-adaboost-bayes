#!/usr/bin/env python
"""
analysis_boost.py

Implements the plotting of the data from boosting experiments.

Python 2.7.12
"""
import matplotlib.pyplot as plt 
import os.path

basepath = os.path.dirname(__file__)
file50n = os.path.abspath(os.path.join(basepath, 'Adaboost+NN 350e 50n', 'adaboost+NN_350e_50n.csv'))
file125n = os.path.abspath(os.path.join(basepath, 'Adaboost+NN 350e 125n', 'adaboost+NN_350e_125n.csv'))
            
def parse_ix_input(dataFile, fn_type, numExamples, ix):
    """
    Parses the data file and creates a list of values of the index ix across all
    instances of the dataset of the dataFile.
    lst = parse_ix_input(open('500epochs_140Nodes_4.csv'), float, 500, 3)
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
    Returns a list with values subtracted from maxVal
    """
    return map(lambda val: maxVal - val, lstValues)

#plotting the boosting experiments with NNs (50 hidden units)      
lstTraining = parse_ix_input(open(file50n), float, 11, 2)  
lstTest = parse_ix_input(open(file50n), float, 11, 3)
t = range(1, 12)
fig = plt.figure()
st = fig.suptitle('AdaBoost + NNs (50 Hidden Units)', fontsize="medium")

plt.subplot(121)
line = plt.plot(t, subtr_from(lstTraining), 'r-', marker='8', label='Training')
plt.setp(line, linewidth=0.5)  
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
leg = plt.legend(fontsize = 'medium')
leg.get_frame().set_linewidth(0.0)
plt.axis([1, 11, -0.01, 0.16])
plt.xlabel('Boosting Rounds')
plt.ylabel('Classification Error')

plt.subplot(122)
line = plt.plot(t, subtr_from(lstTest), 'b-', marker='s', label='Test')
plt.setp(line, linewidth=0.5)        
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
leg = plt.legend(fontsize = 'medium')
leg.get_frame().set_linewidth(0.0)
plt.axis([1, 11, 0.24, 0.36])
plt.xlabel('Boosting Rounds')
       
plt.savefig('AdaBoost+NNs_50', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
plt.show()  

#plotting the boosting experiments with NNs (125 hidden units)  
lstTraining = parse_ix_input(open(file125n), float, 21, 2)  
lstTest = parse_ix_input(open(file125n), float, 21, 3)
t = range(1, 22)
fig = plt.figure()
st = fig.suptitle('AdaBoost + NNs (125 Hidden Units)', fontsize="medium")

plt.subplot(121)
line = plt.plot(t, subtr_from(lstTraining), 'r-', marker='8', label='Training')
plt.setp(line, linewidth=0.5)  
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
leg = plt.legend(fontsize = 'medium')
leg.get_frame().set_linewidth(0.0)
plt.axis([1, 21, -0.01, 0.16])
plt.xlabel('Boosting Rounds')
plt.ylabel('Classification Error')

plt.subplot(122)
line = plt.plot(t, subtr_from(lstTest), 'b-', marker='s', label='Test')
plt.setp(line, linewidth=0.5)        
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
leg = plt.legend(fontsize = 'medium')
leg.get_frame().set_linewidth(0.0)
plt.axis([1, 21, 0.24, 0.36])
plt.xlabel('Boosting Rounds')
       
plt.savefig('AdaBoost+NNs_125', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
plt.show()    

