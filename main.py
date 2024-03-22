import tkinter as tk
from login import initialize_database, newUser, loginUser, displayAllUsers, close_database

# create the main application window
root = tk.Tk()
root.geometry("900x600")
root.title("Login thing")

# initialize the database connection and cursor
conn, cursor = initialize_database()

def signupMaker():
    def signup():
        username = username_entry.get()
        password = password_entry.get()
        first_name = first_name_entry.get()
        last_name = last_name_entry.get()
        newUser(conn, cursor, username, password, first_name, last_name)
        signup_window.destroy()

    signup_window = tk.Toplevel(root)

    username_label = tk.Label(signup_window, text="Username:")
    username_label.pack()
    username_entry = tk.Entry(signup_window)
    username_entry.pack()

    password_label = tk.Label(signup_window, text="Password:")
    password_label.pack()
    password_entry = tk.Entry(signup_window)
    password_entry.pack()

    first_name_label = tk.Label(signup_window, text="First Name:")
    first_name_label.pack()
    first_name_entry = tk.Entry(signup_window)
    first_name_entry.pack()

    last_name_label = tk.Label(signup_window, text="Last Name:")
    last_name_label.pack()
    last_name_entry = tk.Entry(signup_window)
    last_name_entry.pack()

    signup_button = tk.Button(signup_window, text="Sign Up", command=signup)
    signup_button.pack()

def loginMode():
    def login():
        username = username_entry.get()
        password = password_entry.get()
        loginUser(conn, cursor, username, password)
        login_window.destroy()

    login_window = tk.Toplevel(root)

    username_label = tk.Label(login_window, text="Username:")
    username_label.pack()
    username_entry = tk.Entry(login_window)
    username_entry.pack()

    password_label = tk.Label(login_window, text="Password:")
    password_label.pack()
    password_entry = tk.Entry(login_window)
    password_entry.pack()

    login_button = tk.Button(login_window, text="Login", command=login)
    login_button.pack()

def adminMode():
    def viewUsers():
        admin_password = admin_password_entry.get()
        displayAllUsers(conn, cursor, admin_password)
        viewUsers_window.destroy()

    viewUsers_window = tk.Toplevel(root)

    admin_password_label = tk.Label(viewUsers_window, text="Admin Password:")
    admin_password_label.pack()
    admin_password_entry = tk.Entry(viewUsers_window, show="*")
    admin_password_entry.pack()

    viewUsers_button = tk.Button(viewUsers_window, text="View Users", command=viewUsers)
    viewUsers_button.pack()

# buttons for user interaction
signup_button = tk.Button(root, text="Sign Up", command=signupMaker)
signup_button.pack()

login_button = tk.Button(root, text="Log In", command=loginMode)
login_button.pack()

admin_view_button = tk.Button(root, text="Admin View (display users here)", command=adminMode)
admin_view_button.pack()

# start the main event loop
root.mainloop()

# close the database connection when the application is closed
close_database(conn)
