-- This command should run when you have finished loading all data into database by running scripts/prepare_data/load_wikipedia_into_mysql.py
CREATE INDEX index_title ON wikipedia.passages (title)
