import mysql.connector
from mysql.connector import MySQLConnection, Error
import numpy as np

query="SELECT chembl_id,canonical_smiles \
       FROM compound_properties \
       INNER JOIN compound_structures \
       ON compound_properties.molregno=compound_structures.molregno \
       AND full_mwt <= 500 \
       INNER JOIN molecule_dictionary \
       ON compound_properties.molregno=molecule_dictionary.molregno \
       "

outputFile='chembl_output_500.txt'


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
