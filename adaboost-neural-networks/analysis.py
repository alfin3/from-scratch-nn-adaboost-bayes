#!/usr/bin/env python
"""
analysis.py

Implements the plotting of the model selection data.

Python 2.7.12
"""
import matplotlib.pyplot as plt 
import os.path

basepath = os.path.dirname(__file__)

E20_1 = '500epochs_20Nodes_1.csv'
E20_2 = '500epochs_20Nodes_2.csv'
E20_3 = '500epochs_20Nodes_3.csv'
E20_4 = '500epochs_20Nodes_4.csv'

E50_1 = '500epochs_50Nodes_1.csv'
E50_2 = '500epochs_50Nodes_2.csv'
E50_3 = '500epochs_50Nodes_3.csv'
E50_4 = '500epochs_50Nodes_4.csv'

E80_1 = '500epochs_80Nodes_1.csv'
E80_2 = '500epochs_80Nodes_2.csv'
E80_3 = '500epochs_80Nodes_3.csv'
E80_4 = '500epochs_80Nodes_4.csv'

E110_1 = '500epochs_110Nodes_1.csv'
E110_2 = '500epochs_110Nodes_2.csv'
E110_3 = '500epochs_110Nodes_3.csv'
E110_4 = '500epochs_110Nodes_4.csv'

E125_1 = '500epochs_125Nodes_1.csv'
E125_2 = '500epochs_125Nodes_2.csv'
E125_3 = '500epochs_125Nodes_3.csv'
E125_4 = '500epochs_125Nodes_4.csv'

E140_1 = '500epochs_140Nodes_1.csv'
E140_2 = '500epochs_140Nodes_2.csv'
E140_3 = '500epochs_140Nodes_3.csv'
E140_4 = '500epochs_140Nodes_4.csv'

lstFiles = map(lambda fileName: os.path.abspath(os.path.join(basepath,
                                                             'NN 500e 20to140n',fileName)),
                                                [E20_1, E20_2, E20_3, E20_4, 
                                                 E50_1, E50_2, E50_3, E50_4, 
                                                 E80_1, E80_2, E80_3, E80_4,
                                                 E110_1, E110_2, E110_3, E110_4, 
                                                 E125_1, E125_2, E125_3, E125_4, 
                                                 E140_1, E140_2, E140_3, E140_4])
                        
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

def subtr_from(lstValues, maxVal= 1.0):
    """
    Returns a list with value subtracted from maxVal
    """
    return map(lambda val: maxVal-val, lstValues)
      
lstAccData = map(lambda dataFile: parse_ix_input(open(dataFile), float, 500, 3),
                 lstFiles)                
lstLearningRate = parse_ix_input(open(lstFiles[0]), float, 500, 1)
t = range(1, 501)

#plotting 54-20-7 to 54-125-7
line = plt.plot(t, subtr_from(lstAccData[0]), 'r--', label='54-20-7 (n = 4)')
plt.setp(line, linewidth=0.5)
for data in lstAccData[1:3]:
    line = plt.plot(t, subtr_from(data), 'r--')
    plt.setp(line, linewidth=0.5)

line = plt.plot(t, subtr_from(lstAccData[4]), 'b--', label='54-50-7 (n = 4)')
plt.setp(line, linewidth=0.5)
for data in lstAccData[5:7]:
    line = plt.plot(t, subtr_from(data), 'b--')
    plt.setp(line, linewidth=0.5)

line = plt.plot(t, subtr_from(lstAccData[8]), 'g--', label='54-80-7 (n = 4)')
plt.setp(line, linewidth=0.5)
for data in lstAccData[9:11]:
    line = plt.plot(t, subtr_from(data), 'g--')
    plt.setp(line, linewidth=0.5)
    
line = plt.plot(t, subtr_from(lstAccData[12]), 'c--', label='54-110-7 (n = 4)')
plt.setp(line, linewidth=0.5)
for data in lstAccData[13:15]:
    line = plt.plot(t, subtr_from(data), 'c--')
    plt.setp(line, linewidth=0.5)
    
line = plt.plot(t, subtr_from(lstAccData[16]), 'k--', label='54-125-7 (n = 4)')
plt.setp(line, linewidth=0.5)
for data in lstAccData[17:19]:
    line = plt.plot(t, subtr_from(data), 'k--')
    plt.setp(line, linewidth=0.5)
                
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
leg = plt.legend(fontsize = 'small')
leg.get_frame().set_linewidth(0.0)
plt.axis([0, 500, 0.18, 0.7])
plt.xlabel('Epochs')
plt.ylabel('Validation Set Classification Error')
plt.title('Model Selection Experiment')       
plt.savefig('20to125', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
plt.show()  

#plotting 54-125-7 and 54-140-7
line = plt.plot(t, subtr_from(lstAccData[16]), 'k--', label='54-125-7 (n = 4)')
plt.setp(line, linewidth=0.5)
for data in lstAccData[17:19]:
    line = plt.plot(t, subtr_from(data), 'k--')
    plt.setp(line, linewidth=0.5)
    
line = plt.plot(t, subtr_from(lstAccData[20]), 'm--', label='54-140-7 (n = 4)')
plt.setp(line, linewidth=0.5)
for data in lstAccData[21:23]:
    line = plt.plot(t, subtr_from(data), 'm--')
    plt.setp(line, linewidth=0.5)
    
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
leg = plt.legend(fontsize = 'small')
leg.get_frame().set_linewidth(0.0) 
plt.axis([0, 500, 0.18, 0.7])
plt.xlabel('Epochs')
plt.ylabel('Validation Set Classification Error')
plt.title('Model Selection Experiment')        
plt.savefig('125to140', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
plt.show() 

#plotting learning rate
learningLine = plt.plot(t, lstLearningRate, 'g-', mew = 1) 

plt.setp(learningLine, linewidth=1)
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Model Selection Experiment')        
plt.savefig('LearningRate', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
plt.show()  



