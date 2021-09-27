import mysql.connector

mydb = mysql.connector.connect(user='root', password='19911017',
                               host='127.0.0.1', database='wikipedia',
                               auth_plugin='mysql_native_password',
                               buffered=True)
# mydb = mysql.connector.connect(user='pengshan', password='bionlpdb',
#                                host='127.0.0.1', database='wikipedia',
#                                auth_plugin='mysql_native_password',
#                                buffered=True)
mycursor = mydb.cursor()


def get_passage(title):
    title = title.replace('\"', '\'').lower().replace("'", "\\'")
    mycursor.execute("SELECT passage FROM wikipedia.passages WHERE title='%s'" % title)
    result = mycursor.fetchone()
    try:
        return result[0]
    except:
        return None

