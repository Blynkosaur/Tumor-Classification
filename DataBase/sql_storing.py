import mysql.connector
import pandas as pd
import os
import numpy as np

# Expand the tilde to your actual home directory path
password_file = os.path.expanduser("~/password/sql.txt")

# Python type to MySQL type mapping
TYPE_MAP = {
    'int64': 'INT',
    'float64': 'FLOAT',
    'str': 'VARCHAR(255)',
    'object': 'VARCHAR(255)',
    'bool': 'BOOLEAN'
}


def get_columns_with_types(file_path):
    """Read CSV and return column names with their SQL types."""
    data = pd.read_csv(file_path)
    columns = {}
    for col in data.columns:
        dtype = str(data[col].dtype)
        sql_type = TYPE_MAP.get(dtype, 'VARCHAR(255)')
        columns[col] = sql_type
    return columns, data


def create_table_if_not_exists(cursor, table_name, columns):
    """Create table with all columns in a single statement."""
    col_defs = ", ".join([f"`{col}` {dtype}" for col, dtype in columns.items()])
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")


def insert_data_bulk(cursor, table_name, data):
    """Insert all data using bulk insert."""
    if data.empty:
        return
    
    columns = ", ".join([f"`{col}`" for col in data.columns])
    placeholders = ", ".join(["%s"] * len(data.columns))
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    # Convert DataFrame to list of tuples
    values = [tuple(row) for row in data.values]
    cursor.executemany(query, values)


# Connect to database
with open(password_file, "r") as f:
    password = f.readline().strip()

mydb = mysql.connector.connect(
    host="localhost", user="root", password=password, database="neuralnetwork"
)
mycursor = mydb.cursor()

# Load and process breast cancer data
bc_file = "../data/breast-cancer.csv"
columns, data = get_columns_with_types(bc_file)

# Create table if it doesn't exist, then clear existing data
create_table_if_not_exists(mycursor, "BC_DATA", columns)
mycursor.execute("TRUNCATE TABLE BC_DATA")

# Bulk insert all data
insert_data_bulk(mycursor, "BC_DATA", data)
mydb.commit()

print(f"Inserted {len(data)} rows into BC_DATA")
mycursor.close()
mydb.close()
