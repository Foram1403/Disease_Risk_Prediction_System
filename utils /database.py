import sqlite3

conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

# USERS
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT, password TEXT)''')

# HISTORY
c.execute('''CREATE TABLE IF NOT EXISTS history
             (username TEXT, disease TEXT, risk REAL)''')

conn.commit()

def add_user(u, p):
    c.execute("INSERT INTO users VALUES (?, ?)", (u, p))
    conn.commit()

def login_user(u, p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
    return c.fetchone()

def save_history(u, disease, risk):
    c.execute("INSERT INTO history VALUES (?, ?, ?)", (u, disease, risk))
    conn.commit()

def get_history(u):
    c.execute("SELECT disease, risk FROM history WHERE username=?", (u,))
    return c.fetchall()
