from info_argument import *
from sklearn import preprocessing
import operator
import csv
import itertools
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


#Quad takes the name of the graph, opens the corresponding arguments and entailements csv files and calculates for each argument the final weight using the Quad semantics    
def Quad(init_graph, output_dir):
    arguments= pd.read_csv(init_graph + '_arguments.csv')
    entailments=pd.read_csv(init_graph + '_entailment.csv')
    # for each random distribution starting from poisson
    for j in range(2,len(arguments.iloc[0])):
        print("Iteration: " + str(j))
        # degrees will contain the list of arguments and the list of their intial weights, starting from a specified distribution, using the Quad semantics. The initial weights in degrees will change until they converge.
        A = arguments.copy()
        degrees=[A.loc[:, ('Argument')],A.loc[:, (list(arguments)[j])]]
        is_changed=[]
        for i in range(len(arguments['Argument'])):
            is_changed.append(True)
        # we take for each j the basic weights for a specified distribution
        # then for each argument, we calculate its degree based on the degrees calculated in the previous step, we stop once the weights don't change anymore, which means that is_changed contains only False values
        while True in is_changed:
            for i in range(len(arguments['Argument'])):
                #argument = argumentt(arguments['Argument'][i],[arguments['Argument'],basic_weights])
                #argument = argumentt(arguments['Argument'][i],graph,j)
                argument = argumentt(arguments['Argument'][i], degrees, init_graph, j)
                if not(argument[2]) and not(argument[5]):
                    deg=argument[1]
                else:
                    deg_attackers1=[]
                    deg_supporters1=[]
                    if len(argument[3])!=0:
                        for h in range(len(argument[3])):
                            deg_attackers1.append(1-argument[3][h])
                        fa= argument[1] * np.prod(deg_attackers1)
                    else:
                        fa=0
                    if len(argument[5])!=0:
                        for h in range(len(argument[5])):
                            deg_supporters1.append(1-argument[5][h])
                        fb = 1-((1-argument[1])*np.prod(deg_supporters1))
                    else:
                        fb=0
                    if fa==0 and fb!=0:
                        deg=fb
                    elif fb==0 and fa!=0:
                        deg=fa
                    else:
                        deg= (fa+fb)/2
                #after we calculate deg for each argument based on degrees, where degrees[1] was initially the basic weights, we then check 
                if deg !=A.loc[i, (list(arguments)[j])]:
                    A.loc[i, (list(arguments)[j])]=deg
                    is_changed[i]=True
                else:
                    is_changed[i]=False
        data = []
        for h in range(len(degrees[0])):
            data.append([degrees[0][h],degrees[1][h]])
        df = pd.DataFrame(data)
        df.to_csv(output_dir + '_weights_' + list(arguments)[j] + '.csv', index=False, header=False)
    return

Quad(init_graph="data/debatepedia/", output_dir="data/debatepedia/quad_debatepedia/gold_labels/original/gold")
Quad(init_graph ="data/angrymen/quad-AngryMen/TwelveAngryMan", output_dir="data/angrymen/quad-AngryMen/gold_labels/original/TwelveAngryMen")
#Quad('Chinaonechildpolicy')
