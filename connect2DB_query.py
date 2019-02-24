import mysql.connector
from mysql.connector import MySQLConnection, Error
import numpy as np
from sys import argv

with open(argv[1], 'r') as myfile:
    query=myfile.read().replace('\n', ' ')

outputFile= argv[2]


def iter_row(cursor, size=10):
    while True:
        rows = cursor.fetchmany(size)
        if not rows:
            break
        for row in rows:
            yield row

def connect(query):
    outputtable = []
    """ Connect to MySQL database """
    try:
        conn = mysql.connector.connect(host='localhost',
                                       database='chembl_23',
                                       user='root',
                                       password='')
        if conn.is_connected():
            print('Connected to MySQL database')
        outputtable = query_with_fetchmany(conn,query)
        return outputtable
    except Error as e:
        print(e)

    finally:
        conn.close()

def query_with_fetchmany(conn,query):
    outputtable=[]
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        for row in iter_row(cursor,10):
            #print(row)
            outputtable.append(row)
        return outputtable
    except Error as e:
        print(e)
    finally:
        cursor.close()


if __name__ == '__main__':
    outputtable=connect(query)
    np.savetxt(outputFile,outputtable,fmt='%s')
