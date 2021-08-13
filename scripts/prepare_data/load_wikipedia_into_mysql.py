import csv
import mysql.connector

'''
This script load psgs_w100.tsv into mysql
psgs_w100.tsv contains the entire Wikipedia
To download psgs_w100.tsv you need the following steps:
    git clone https://github.com/facebookresearch/DPR
    cd DPR
    python dpr/data/download_data.py --resource data.wikipedia_split.psgs_w100

You would also need to create a database called Wikipedia
In Wikipedia create a table called passages
CREATE TABLE `wikipedia`.`passages` (
  `idx` INT NOT NULL AUTO_INCREMENT,
  `title` VARCHAR(256) NULL,
  `passage` MEDIUMTEXT NULL,
  PRIMARY KEY (`idx`));
'''


path_in = '/path/to/psgs_w100.tsv'


mydb = mysql.connector.connect(user='root', password='root1111',
                              host='127.0.0.1', database='wikipedia',
                              auth_plugin='mysql_native_password')
mycursor = mydb.cursor()


def add_passage(title, passage):
    title = title.replace('\"', '\'').lower()
    passage = passage.replace('\"', '\'')
    sql = "INSERT INTO passages (title, passage) VALUES (%s, %s)"
    val = (title, passage)
    mycursor.execute(sql, val)
    mydb.commit()


con = []
cur_passage = ''
cur_title = ''
max_len = 21015323
with open(path_in, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    head = next(reader)
    for i, row in enumerate(reader):
        idx, para, title = row
        if i % 100000 == 0:
            print('Progress: %s/%s - %s per cent' % (i, max_len, 100 * round(i/max_len, 3)))
        if title != cur_title:
            # new passage appears, so we add the existing passage to the DB
            if cur_passage != '':
                add_passage(cur_title, cur_passage)
            cur_passage = para
            cur_title = title
        else:
            cur_passage += para
add_passage(cur_title, cur_passage)