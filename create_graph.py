import MySQLdb
import pickle
import networkx as nx




def creategraph():
    errors=0
    signs=0
    mysqlerrors=0
    for j in range(3,250):
        print("starting loop number: ",j)
        sqlstr='select * from pagelinks limit %s,%s'
        cursor.execute(sqlstr,(j*1000000, 1000000))
        rows = cursor.fetchall()
        print("fetch suceedded")
        for i in range(len(rows)):
            if "%" in rows[i][2]:
                signs+=1
                continue
            sqlstr='select page_id from page where page_title like %s and page_namespace = %s'
            cursor.execute(sqlstr, (rows[i][2],rows[i][1],))
            page_id=cursor.fetchone()
            sqlstr='insert into graph values(%s, %s)'
            try : 
                cursor.execute(sqlstr, (rows[i][0],page_id[0],))
            #in some cases the page_id turns null - not the program fault, checked with the tables- some are missing, especialy weird cuts
            except TypeError:
                errors+=1
            #duplicated rows  -  exists!
            except MySQLdb.Error:
                mysqlerrors+=1
        print("Erros: ", errors)
        print("signs: " , signs)
        print("mysql errors", mysqlerrors)
        conn.commit()
        print("comitted the" , j)




conn = MySQLdb.connect('localhost','root','webweb', 'WIKI')
cursor = conn.cursor()

creategraph()          

conn.close()