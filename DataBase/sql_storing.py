import mysql.connector
import pandas as pd
import os
import numpy as np

# Expand the tilde to your actual home directory path
file_path = os.path.expanduser("~/password/sql.txt")


def get_header(file_path):
    data = pd.read_csv(file_path)
    sample = np.array(data)[0]
    column_headers = list(data.columns.values)
    hashmap = {}
    for head in range(len(column_headers)):
        data_type = str(type(sample[head]))
        ls = data_type.split("'")
        if ls[1] == "str":
            hashmap[column_headers[head]] = "char"
        else:
            hashmap[column_headers[head]] = ls[1]

    print(hashmap)
    return hashmap


file = open(file_path, "r")
password = file.readline()
mydb = mysql.connector.connect(
    host="localhost", user="root", password=password, database="neuralnetwork"
)
bc_file = os.path.expanduser("../data/breast-cancer.csv")
headers = get_header("../data/breast-cancer.csv")
mycursor = mydb.cursor()

mycursor.execute("drop table BC_DATA")
mycursor.execute(f"create table BC_DATA (id {headers['id']} )")
for header in headers:
    if header != "id":
        mycursor.execute(f"alter table BC_DATA add {
                         header} {headers[header]} ")
