from sklearn import preprocessing
import operator
import csv
import itertools
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from collections import defaultdict

#Extract info from xml files
tree = ET.parse('data/angrymen/12AngryMen/12AngryMen_final_dataset.xml')
entailmentcorpus = tree.getroot()
l = entailmentcorpus.findall('pair')
h=[]
for i in range(len(l)):
    h.append(l[i].attrib)

j= sorted(h, key=operator.itemgetter("topic"))
outputList=[]
for i,g in itertools.groupby(j, key=operator.itemgetter("topic")):
    outputList.append(list(g))
grandeliste=[]   
for i in range(len(outputList)):
    grandeliste.append([])
    for f in range(len(outputList[i])):
        grandeliste[i].append(outputList[i][f])
m=[]
for i in range(len(outputList)):
    m.append(entailmentcorpus.findall("./pair/[@topic='"+outputList[i][0]['topic']+"']"))
# Abortion = m[0]
# Each pair of Abortion is m[0][i]
# the arguments of each pair of Abortion are m[0][i][0] and m[0][i][1]
# m is the list of the graphs for example here we have a graph for Abortion, a graph for Tablet and a graph for Obesity
# info is the list of lists, where each list represents a graph and contains information about the relations between argumentsof this graph 
info = [] #edges (80)
info_text = dict() #nodes (83)
info_text_act_by_id = dict()
for i in range(len(m)):
    info.append([])
    for j in range(len(m[i])):
        topic = m[i][j].attrib['topic']
        relation = m[i][j].attrib['BAF']
        id_argument1 = m[i][j][0].attrib['id'].lower()
        id_argument2 = m[i][j][1].attrib['id'].lower()
        info[i].append([topic,relation, id_argument1,id_argument2])
        info_text[id_argument1] = m[i][j][0].text
        info_text[id_argument2] = m[i][j][1].text
        info_text_act_by_id[id_argument1] = topic[0:4]
        info_text_act_by_id[id_argument2] = topic[0:4]

info_text
        
#arguments is the list of lists that contain the information about the intial weights of the arguments of each graph
s= []
for i in range(len(info)):
    s.append([])
    for j in range(len(info[i])):
        s[i].append([info[i][j][0],info[i][j][2].lower()])
        s[i].append([info[i][j][0],info[i][j][3].lower()])
arguments=[]
for i in range(len(info)):
    new_k = []
    for elem in s[i]:
        if elem not in new_k:
            new_k.append(elem)
    arguments.append(new_k)

with open("data/angrymen/12AngryMenGraphData/graph_indicator.txt", "w+") as f1:
    for i in range(len(info)):
        for j in range(len(info[i])):
            f1.write(str(i+1)+"\n")
f1.close()
# my_dict=dict()
# k=1
# for i in range(len(arguments)):
#     for j in range(len(arguments[i])):
#         if arguments[i][j][1].lower() not in my_dict:
#             my_dict[arguments[i][j][1].lower()]=k
#             k+=1
# print(len(my_dict))
# my_dict

info_text_dict = dict()
index = 0
with open("data/angrymen/12AngryMenGraphData/nodes_text.txt", "w+") as file:
    for k,v in info_text.items():
        info_text_dict[k] = [index, v]
        file.write(v.replace("\n", "") + "\n")
        index += 1
file.close()
info_text_dict

index = 0
with open("data/angrymen/12AngryMenGraphData/nodes_text_id.txt", "w+") as file:
    for k,v in info_text.items():
        file.write(k + "\t" + v.replace("\n", "") + "\n")
        index += 1
file.close()

with open("data/angrymen/12AngryMenGraphData/nodes_text_by_act.txt", "w+") as file:
    for k,v in info_text.items():
        file.write(info_text_act_by_id[k] + "," + v.replace("\n", "") + "\n")
file.close()


# Create the random basic weights
epsilon=0.0001
Nboftimes=100
for i in range(len(arguments)):
    for r in range(Nboftimes):
        data1 =np.random.poisson(0.3, len(arguments[i]))
        if np.min(data1)!=np.max(data1):
            data1 = (data1 - np.min(data1))/(np.max(data1) - np.min(data1))
        data2= np.random.uniform(low=0.1, high=1.0, size=len(arguments[i]))
        data3= np.random.normal(loc=0.0, scale=1.0, size=len(arguments[i]))
        data3 = (data3 - np.min(data3))/(np.max(data3) - np.min(data3))
        data4= np.random.beta(a=0.5, b=0.5, size=len(arguments[i]))
        data= [data1, data2,data3,data4]
        for h in range(len(data)):
            for j in range(len(arguments[i])):
                if data[h][j] <= epsilon:
                    data[h][j]= epsilon
                if data[h][j]>= 1-epsilon:
                    data[h][j]=1-epsilon
                arguments[i][j].append(data[h][j])

# Create the entailment and arguments csv files
with open('data/angrymen/12AngryMenGraphData/edges.csv', 'w') as file:
    edgesFile = csv.writer(file)
    for i in range(len(info)):
        for j in range(len(info[i])):
            edgesFile.writerow([info_text_dict[info[i][j][-2].lower()][0], info_text_dict[info[i][j][-1].lower()][0]])
data= pd.read_csv('data/angrymen/12AngryMenGraphData/edges.csv')
data.to_csv('data/angrymen/12AngryMenGraphData/edges.csv', index=False)

with open('data/angrymen/12AngryMenGraphData/edge_labels.csv', 'w') as file:
    edgesLabelsFile = csv.writer(file)
    for i in range(len(info)):
        for j in range(len(info[i])):
            edgesLabelsFile.writerow([info[i][j][1]])
data= pd.read_csv('data/angrymen/12AngryMenGraphData/edge_labels.csv')
data.to_csv('data/angrymen/12AngryMenGraphData/edge_labels.csv', index=False)

# # Create the entailment and arguments csv files
# for i in range(len(info)):
#     with open("data/12AngryMen/" + info[i][0][0]+'_entailment.csv', 'w') as file:
#         writer = csv.writer(file)
#         writer.writerow(['','entailment','id_argument1','id_argument2'])
#         for j in range(len(info[i])):
#             writer.writerow(info[i][j])
#     data= pd.read_csv("data/12AngryMen/" + info[i][0][0]+'_entailment.csv')
#     data.pop('Unnamed: 0')
#     data.to_csv("data/12AngryMen/" + info[i][0][0]+'_entailment.csv')
#     with open("data/12AngryMen/" + info[i][0][0]+'_arguments.csv', 'w') as file:
#         writer = csv.writer(file)
#         writer.writerow(['','Argument','poisson', 'uniform', 'normal', 'beta', 'poisson', 'uniform','normal', 'beta','poisson', 'uniform','normal', 'beta',])
#         for n in range(len(arguments[i])):
#             writer.writerow(arguments[i][n])
#     data= pd.read_csv("data/12AngryMen/" + info[i][0][0]+'_arguments.csv')
#     data.pop('Unnamed: 0')
#     data.to_csv("data/12AngryMen/" + info[i][0][0]+'_arguments.csv')


with open("data/angrymen/quad-AngryMen/TwelveAngryMan_entailment.csv", 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['', 'entailment', 'id_argument1', 'id_argument2'])
    for i in range(len(info)):
        for j in range(len(info[i])):
            writer.writerow(info[i][j])
data = pd.read_csv("data/angrymen/quad-AngryMen/TwelveAngryMan_entailment.csv")
data.pop("Unnamed: 0")
data.to_csv("data/angrymen/quad-AngryMen/TwelveAngryMan_entailment.csv")

with open("data/angrymen/quad-AngryMen/TwelveAngryMan_arguments.csv", 'w') as file:
    writer = csv.writer(file)
    header = ['', 'Argument']
    for r in range(Nboftimes):
        header.append('poisson')
        header.append('uniform')
        header.append('normal')
        header.append('beta')
    print(len(header))
    writer.writerow(header)
    for i in range(len(info)):
        for n in range(len(arguments[i])):
            writer.writerow(arguments[i][n])
data= pd.read_csv("data/angrymen/quad-AngryMen/TwelveAngryMan_arguments.csv")
data.pop("Unnamed: 0")
data.to_csv("data/angrymen/quad-AngryMen/TwelveAngryMan_arguments.csv")
   




