import mariadb

try:
    conn = mariadb.connect(
        user="root",
        password="tuan123",
        host="localhost",
        port=3306
    )
    print("Connected to MariaDB!")

    cur = conn.cursor()
    cur.execute("SELECT VERSION()")
    print("MariaDB version:", cur.fetchone())

except mariadb.Error as e:
    print("Error connecting to MariaDB:", e)