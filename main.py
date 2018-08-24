#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:38:29 2018

@author: anthony
"""

import pandas as pd


# NOTE: Z = 1, Y = 0
def main():
    data = loadFile("boolean.csv")
    
    attributes = data[1]
    targetAttributes = data[2]
    
    listOfAttributes = splitAttributes(attributes)
    
    makeID3(attributes, targetAttributes)
    
    
    
def loadFile(fileName):
    data = []
    
    file = pd.read_csv(fileName)
    
    data.append(file.iloc[:, 0])
    data.append(file.iloc[:, 1])
    data.append(file.iloc[:, 2])
    
    for line in data:
        print(line)
    
    return data

#TODO:
#Make a function that looks at all features and picks highest information gained one...




def splitAttributes(attributes):
    
    
    list0 = []
    list1 = []
    
    for row in attributes:
        list0.append(row[0])
        list1.append(row[1])
        
    listOfAttributes = []
    listOfAttributes.append(list0)
    listOfAttributes.append(list1)
    
    print(listOfAttributes)
    
    return listOfAttributes


#def id3(attributes, targetAttributes):
    
    
    
    
    
main()




    
    
    
    
    
    
    
    
    
    






