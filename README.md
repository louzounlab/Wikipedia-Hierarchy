#### Full eaplanation can be found at the attached pdf file 

# Hierarchy Clustering
Building a model based on Wikipedia's hierarchy data to define metric on graph groups based on a connected tree.

Useage:

### download data 
In order to load data, use enwiki.dump - explanation can be found in pdf.

### optional: 
Creating the groups can be done using "create_groups.py", when data is in local mysql-server.
Use create_graph, create_tree in order to create the needed structures.

### project the graph
AfterWards, project the graph using "projections.py" file. dimensions can be changed in configuration "conf.ini"

### label data
Use "labels_data.py" to create dictionary of labels from the tree. This dictionary will be used by the model.

### model
use the model "representaion to distances" to get the results showed in "deviations_hostpgram".

