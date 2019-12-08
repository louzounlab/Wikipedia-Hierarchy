'''
Use this file at first, to project the graph. dimensions can be chosen by user, changes can be applied in configuration file - conf.ini
'''

import configparser
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pickle
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap

config = configparser.ConfigParser()
config.read('conf.ini')
d_dimensions = config['ARGUMENTS']['dimensions']


g=nx.read_adjlist(open("GraphAsAdjlist.adjlist", 'rb'), create_using=nx.DiGraph())
#
# # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(g, dimensions=d_dimensions, walk_length=10, num_walks=10, workers=4)  # Use temp_folder for big graphs
#
# # Embed nodes
model = node2vec.fit(window=100, min_count=1, batch_words=4)
model.wv.save_word2vec_format("projected")
t=open("projected")
y=t.readlines()

for i in range(1,len(y)):
    print(y[i])
    break
