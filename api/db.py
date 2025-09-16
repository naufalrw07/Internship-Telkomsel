import mysql.connector
from mysql.connector import Error

def get_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",         # default XAMPP
            password="",         # isi kalau ada password
            database="internshiptelkomsel"  # nama database kamu
        )
        return conn
    except Error as e:
        print("DB connection error:", e)
        return None
