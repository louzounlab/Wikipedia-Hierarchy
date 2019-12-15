import MySQLdb
import pickle
import networkx as nx
import os


'''
This file creates groups and a graph, Using mysql server based on wikipedia data. tables used: page, pagelinks, categorylinks
'''


def create_names(ids):
    names=[]
    for id in ids:   
        sqlstr = "select page_title from page where page_id = %s"
        cursor.execute(sqlstr, (id,))
        names.append( cursor.fetchone()[0] )
    return names

    
        

def create_groups(file_name, names):
    for name in names:
        sqlstr='select cl_from From categorylinks,page  WHERE categorylinks.cl_from=page.page_id AND  categorylinks.cl_to LIKE %s'        
        cursor.execute(sqlstr, (name,))
        sons = [ x[0] for x in cursor.fetchall()]    
        with open(os.path.join(file_name ,name+'_sons.pkl'), 'wb' ) as handle:
            pickle.dump(sons, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


def create_oneway_graph(file_name):
    all_neighbors=[]; 
    for i in range(len(names)):    
        with open(os.path.join(file_name ,names[i]+'_sons.pkl'), 'rb' ) as f:
            sons = pickle.load(f)
        #for each of the leafs in the tree on the give 1 from 4 groups: (like: for leaf in Abstract Algebra's sons), for example Algebraic_topology is one of 131
        for idx_son, son in enumerate(sons):        
            sqlstr='select * from pagelinks where pl_from=%s'
            cursor.execute(sqlstr, (son,))
            first_neighbors = cursor.fetchall()
            all_neighbors+=first_neighbors
            #for each forst neighbor, find its first neighbor (second neighbor of the leaf) and add it to the all neighbors list.
            for idx, neighbor in enumerate(first_neighbors):
                #find the neighbor id, given the result from pagelinks 
                sqlstr='select page_id from page where page_title LIKE %s AND page_namespace = %s'
                cursor.execute(sqlstr, (neighbor[2],neighbor[1],))
                #the id. fetch only one to aviud errors 
                page_id=cursor.fetchone()
                #same as before for each id - find its second neighbors
                sqlstr='select * from pagelinks where pl_from=%s'
                cursor.execute(sqlstr, (page_id,))
                sec_neighbors = cursor.fetchall()
                all_neighbors+=sec_neighbors
    
    with open('demo_graph'+file_name+'.pkl', 'wb' ) as handle:
            pickle.dump(all_neighbors, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    

def from_list_to_nx(directory_name):
    w1=0;w2=0; nones=0;alreadyIn=0
    s=[]
    All_DiGraph = nx.DiGraph()
    #First, adding nodes of all the directed sons - the groups
    for i in range(len(names)):
        with open(os.path.join(directory_name,names[i]+'_sons.pkl'), 'rb' ) as f:
            sons = pickle.load(f)
        #add nodes of all the sons to the graph
        All_DiGraph.add_nodes_from(sons) 
    with open('demo_graph'+directory_name+'.pkl', 'rb' ) as y:
        graph_lines = pickle.load(y)
    for idx_line,line in enumerate(graph_lines):
        if(idx_line % 100000 ==0 ):
        #Get the id of the pl_to - the node to which the edge comes
        sqlstr='select page_id from page where page_title=%s and page_namespace=%s'
        cursor.execute(sqlstr, (line[2],line[1],))
        p_id = cursor.fetchone()
        #if the page doesnt exist in the page table
        if p_id == None:
            nones+=1
            continue
        if isinstance(p_id, tuple):
            p_id = p_id[0]
        #add the edge
        e_toadd = (line[0],p_id)
        if All_DiGraph.has_edge(line[0],p_id):
            alreadyIn+=1
        All_DiGraph.add_edge(line[0],p_id)
    print("writing to FILE")
    with open('directed_graph_4_afterchange.pkl', 'wb' ) as handle:
            pickle.dump(All_DiGraph, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
        

        
def check_all_sons_exist():
    for i in range(len(names)):
        print("i iteratiojn" , i)
        with open(os.path.join("14_groups",names[i]+'_sons.pkl'), 'rb' ) as f:
            sons = pickle.load(f)
            for son in sons:
                if son not in g:
                    print('error')
                    print(son)


#in order to handle different versions of nx, save the file when on the server (#1) and open them from out (#2)
def transfer_version_nx():
    with open('directed_graph_4_afterchange.pkl', 'rb' ) as f:
            g1 = pickle.load(f)

    #1
    nx.write_edgelist(g1, "directed_graph_4_afterchange.edgelist")
    #2
    g2= nx.read_edgelist("directed_graph_4_afterchange.edgelist",create_using=nx.DiGraph())
    

if __name__ == '__main__':

    conn = MySQLdb.connect('localhost','root','webweb', 'WIKI')
    cursor = conn.cursor()
    ids_14 = [736492,722408,1203410,1516113,874722,729377,1399912,46564979,32187902,24238485,21573294,58114046,12606057,4309562]
    ids_4 = [690672, 690777,34511514,782015]
    names = create_names(ids_4)
    create_groups("4_groups", names)
    create_oneway_graph("4_groups")
    from_list_to_nx("4_groups")
    transfer_version_nx()

    #check_no_sons()         
    #find_leafs()
    #check()
    #transfer_version_nx()


    conn.close()


