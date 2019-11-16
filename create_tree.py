import MySQLdb
import pickle
import networkx as nx



#create the caegory tree of Wikipedia
def create_tree(): 
    j=0
    DG = nx.DiGraph()
    sqlstr='select page_id from page where page_namespace=14'
    cursor.execute(sqlstr)
    cats_b =  cursor.fetchall()
    cats = [ cats_b[i][0] for i in range (len(cats_b)) ]
    for cat in cats:
        j+=1
        if j % 50000 == 0:
            print("j", j)
            print("cat" ,cat)
        DG.add_node(cat)
        sqlstr='select cl_to from categorylinks where cl_from = %s'
        cursor.execute(sqlstr, ( cat ,))
        fathers_names =  cursor.fetchall()
        for father in fathers_names:
            sqlstr='select page_id from page where page_title like %s and page_namespace = 14'
            cursor.execute(sqlstr, ( father[0] ,))
            id_father = cursor.fetchone()
            if id_father is not None:
                if not isinstance(id_father[0], long):
                    print("error")
                    print(id_father[0])
                    print(id_father)
                    print(j)
                    exit()
                if id_father[0] not in DG:
                    DG.add_node(id_father[0])
                DG.add_edge(id_father[0], cat)  
    with open('tree.pkl', 'wb' ) as f:
            pickle.dump(DG, f, protocol=pickle.HIGHEST_PROTOCOL)


#ceck for the leafs - categories with no category sons in the ctagorylinks table 
def check_no_sons():
    min_sons = [1000, "",""]
    sqlstr = "select cl_from,cl_to,page_title from categorylinks,page where cl_to like %s and page_id = cl_from and page_namespace=14"
    cursor.execute(sqlstr, ("Musical_analysis",))
    sons = cursor.fetchall()
    for son in sons:
        sqlstr = "select cl_from,cl_to,page_title from categorylinks,page where cl_to like %s and page_id = cl_from and page_namespace=14"
        cursor.execute(sqlstr, (son[2],))
        sons2 = cursor.fetchall()
        print(len(sons2), son)

   
def func2():
    with open('tree.pkl', 'rb' ) as f:
        tree = pickle.load(f)
    print("loaded")
    print( tree.neighbors(12606057 ))
    #for i in list1:
     #   sqlstr= "select * from page where page_id = %s"
      #  cursor.execute(sqlstr, (i,))
       # l = cursor.fetchall()
        #print(l)
        

conn = MySQLdb.connect('localhost','root','webweb', 'WIKI')
cursor = conn.cursor()
create_tree()
#check_no_sons()


conn.close()
