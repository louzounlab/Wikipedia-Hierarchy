import MySQLdb
import pickle
import networkx as nx
import os


 

#-----------------------------------------------------------------------------------------------------------------------------------------------------------

def create_names():
    ids = [736492,722408,1203410,1516113,874722,729377,36770192,46564979,32187902,24238485,21573294,58114046,12606057,4309562]
    for id in ids:   
        sqlstr = "select page_title from page where page_id = %s"
        cursor.execute(sqlstr, (id,))
        names.append( cursor.fetchone()[0] )
    print(names)
    print(len(names))
      
 
def create_14_groups():
    for name in names:
        sqlstr='select cl_from From categorylinks,page  WHERE categorylinks.cl_from=page.page_id AND  categorylinks.cl_to LIKE %s'        
        cursor.execute(sqlstr, (name,))
        sons = [ x[0] for x in cursor.fetchall()]    
        with open(os.path.join("14_groups",name+'_sons.pkl'), 'wb' ) as handle:
            pickle.dump(sons, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(len(sons))   
    


def create_oneway_graph():
    missing = 0
    all_neighbors=[]; 
    All_DiGraph = nx.DiGraph()
    for i in range(len(names)):
        print("---------------------------------------------------starting loop number ",i)
        with open(os.path.join("14_groups",names[i]+'_sons.pkl'), 'rb' ) as f:
            sons = pickle.load(f)
        #add nodes of all the sons to the graph
        All_DiGraph.add_nodes_from(sons)       
        #for each of the leafs (sons) find its neighbors and their neiighbors and add to graph
        for idx_sons,son in enumerate(sons):   
            sqlstr='select pl_title,pl_namespace from pagelinks where pl_from=%s'
            cursor.execute(sqlstr, (son ,))
            first_neighbors = cursor.fetchall()
            #for each of the first neighbors,  find its id in order to add it to the graph
            for idx_first, f_neighbor in enumerate(first_neighbors):
                sqlstr='select page_id from page where page_title like %s and page_namespace = %s'
                cursor.execute(sqlstr, (f_neighbor[0], f_neighbor[1] ,))
                f_neighbor_id = cursor.fetchone()
                if f_neighbor_id is None:
                    missing +=1
                    continue
                else:
                    f_neighbor_id = f_neighbor_id[0]
                    All_DiGraph.add_edge(son,f_neighbor_id)
                #for each of the first neighbors we also find neighbors - second degree neighbors
                sqlstr='select pl_title,pl_namespace from pagelinks where pl_from=%s'
                cursor.execute(sqlstr, (f_neighbor_id ,))
                second_neighbors = cursor.fetchall()
                print("group name and num",names[i], i, "of 14","son number", idx_sons, "of", len(sons),"first number", idx_first, "of",len(first_neighbors),"has sec_neigh sum of",len(second_neighbors))
                for sec_idx, sec_neighbor in enumerate(second_neighbors):  
                    #find the second neighbor id
                    sqlstr='select page_id from page where page_title like %s and page_namespace = %s'
                    cursor.execute(sqlstr, (sec_neighbor[0], sec_neighbor[1] ,))
                    sec_neighbor_id = cursor.fetchone()
                    if sec_neighbor_id is None:
                        missing +=1
                        continue
                    else:
                        sec_neighbor_id = sec_neighbor_id[0]      
                        All_DiGraph.add_edge(f_neighbor_id,sec_neighbor_id) 
    print("missing", missing)        

    with open('14_tobe_group_graph.pkl', 'wb' ) as handle:
            pickle.dump(All_DiGraph, handle, protocol=pickle.HIGHEST_PROTOCOL)             
                    
        
        
           
    
    
    
    
    
    

if __name__ == "__main__":
    conn = MySQLdb.connect('localhost','root','webweb', 'WIKI')
    cursor = conn.cursor()
    names = []
    #create_4_groups()
    #create_oneway_graph(names)
    #list_to_networkx()
    #check_no_sons()
    
#---------------------------------------------              
    create_names()
    create_14_groups()
    create_oneway_graph()
    
    
    
    conn.close()


