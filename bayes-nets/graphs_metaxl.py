#!/usr/bin/env python
"""
graphs_metaxl.py

Implements the graphs for Bayes net experiments with the met-axl data set.

Python 2.7.12
"""

# Random variable names correspond to the column indeces in the data set.
# The index of the class variable is 9. 
                              
IX_CLASS = 9

NAIVE_BAYES = map(tuple,
               [[9], [9,0], [9,1], [9,2], [9,3],
                     [9,4], [9,5], [9,6], [9,7], [9,8]])

MET_BAYES = map(tuple,
               [[9], [9,0], [9,1], [9,2], [9,3],
                     [9,1,4], [9,5], [9,6], [9,7], [9,8]])

MET_AKT_BAYES = map(tuple,
               [[9], [9,0], [9,1], [9,2], [9,3],
                     [9,1,3,4], [9,5], [9,6], [9,7], [9,8]])

AXL_MET_AKT_BAYES = map(tuple,
               [[9], [9,0], [9,0,1], [9,2], [9,3],
                     [9,1,3,4], [9,5], [9,6], [9,7], [9,8]])

AXL_MET_AKT_N1_BAYES = map(tuple,
               [[9], [9,0], [9,0,1], [9,1,2], [9,3],
                     [9,1,3,4], [9,5], [9,6], [9,7], [9,8]])

AXL_MET_AKT_N2_BAYES = map(tuple,
               [[9], [9,0], [9,0,1], [9,1,2], [9,1,3],
                     [9,1,3,4], [9,5], [9,6], [9,7], [9,8]])

AXL_MET_AKT_N3_BAYES = map(tuple,
               [[9], [9,0], [9,0,1], [9,1,2], [9,1,3],
                     [9,1,3,4], [9,1,5], [9,6], [9,7], [9,8]])

FULL_PATHWAY_BAYES = map(tuple,
               [[9], [9,0], [9,0,1], [9,1,2], [9,1,3],
                     [9,1,3,5,4], [9,1,5], [9,4,6], [9,6,7], [9,6,8]])

LIST_NETS = [NAIVE_BAYES, MET_BAYES, MET_AKT_BAYES, AXL_MET_AKT_BAYES,
             AXL_MET_AKT_N1_BAYES, AXL_MET_AKT_N2_BAYES, AXL_MET_AKT_N3_BAYES,
             FULL_PATHWAY_BAYES]
               