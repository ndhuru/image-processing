import tkinter as tk
import sqlite3
import subprocess

def initialize_database():
    # connect to SQLite database
    conn = sqlite3.connect('allmyusers.db')
    cursor = conn.cursor()

    # create a table for users if it doesn't exist
    newtable_query = '''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        first_name TEXT,
        last_name TEXT
    );
    '''
    cursor.execute(newtable_query)
    conn.commit()

    return conn, cursor

def newUser(conn, cursor, username, password, first_name, last_name):
    insert_query = '''
    INSERT INTO users (username, password, first_name, last_name)
    VALUES (?, ?, ?, ?);
    '''
    cursor.execute(insert_query, (username, password, first_name, last_name))
    conn.commit()
    print("New user created successfully!")

def loginUser(conn, cursor, username, password):
    select_query = '''
    SELECT * FROM users WHERE username = ? AND password = ?;
    '''
    cursor.execute(select_query, (username, password))
    user = cursor.fetchone()

    if user:
        # Write the username to a temporary file
        with open("temp_username.txt", "w") as temp_file:
            temp_file.write(username)

        # Launch a subprocess to execute another Python script (client.py)
        subprocess.Popen(["python", "client.py"])
    else:
        print("Not a user in the system, sign up first please.")

def displayAllUsers(conn, cursor, admin_password):
    if admin_password == "LosPollosHermanos":
        select_all_query = '''
        SELECT * FROM users;
        '''
        cursor.execute(select_all_query)
        all_users = cursor.fetchall()
        viewer_window = tk.Toplevel()

        if all_users:
            # display user information in the new window
            for user in all_users:
                id_label = tk.Label(viewer_window, text=f"ID: {user[0]}")
                username_label = tk.Label(viewer_window, text=f"Username: {user[1]}")
                password_label = tk.Label(viewer_window, text=f"Password: {user[2]}")
                first_name_label = tk.Label(viewer_window, text=f"First Name: {user[3]}")
                last_name_label = tk.Label(viewer_window, text=f"Last Name: {user[4]}")

                id_label.pack()
                username_label.pack()
                password_label.pack()
                first_name_label.pack()
                last_name_label.pack()
        else:
            # display a message if no users are found
            no_users_label = tk.Label(viewer_window, text="No users found")
            no_users_label.pack()
    else:
        # display a message for an invalid password
        invalid_password_label = tk.Label(viewer_window, text="Invalid password")
        invalid_password_label.pack()

def close_database(conn):
    # close the database connection
    conn.close()
