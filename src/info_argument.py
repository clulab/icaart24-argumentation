from sklearn import preprocessing
import operator
import csv
import itertools
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


# we need a function that gets for each argument its basic weight and the list of its attackers, a list of their corresponding degrees, a list of its supporters and a list of their corresponding degrees
#

def argumentt(id,degrees,graph,j):
    arguments= pd.read_csv(graph+'_arguments.csv')
    entailments=pd.read_csv(graph+'_entailment.csv')
    index=arguments[arguments['Argument']==id].index.values[0]
    basic_weights=arguments[list(arguments)[j]]
    basic_weight = basic_weights[index]
    attackers=[]
    supporters=[]
    deg_attackers=[]
    deg_supporters=[]
    for j in range(len(entailments)):
            if entailments['id_argument2'][j]==id and (entailments['entailment'][j]== 'ENTAILMENT' or entailments['entailment'][j]=='YES' or entailments['entailment'][j]=='SUPPORT'):
                supporters.append(entailments['id_argument1'][j])
                index=arguments[arguments['Argument']==entailments['id_argument1'][j]].index.values[0]
                deg_supporters.append(degrees[1][index])
            elif entailments['id_argument2'][j]==id and (entailments['entailment'][j]== 'NONENTAILMENT' or entailments['entailment'][j]=='NO' or entailments['entailment'][j]=='ATTACK'):
                attackers.append(entailments['id_argument1'][j])
                index=arguments[arguments['Argument']==entailments['id_argument1'][j]].index.values[0]
                deg_attackers.append(degrees[1][index])
    liste = [id,basic_weight, attackers, deg_attackers, supporters, deg_supporters]
    return liste