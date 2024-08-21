import sqlite3

## Connection to the SQLite Database
connection=sqlite3.connect("LLM_Models.db")

## The Cursor responsible for running the queries
cursor=connection.cursor()

## The table creation
table_info="""
    CREATE TABLE MODEL(NAME VARCHAR(25), PROVIDER VARCHAR(25),
    CONTEXT_WINDOW_IN_K INT, PRICE_IN_MILLION INT);
"""

cursor.execute(table_info)

## Inserting data
cursor.execute('''INSERT INTO MODEL VALUES('GPT-4o','OpenAI','128','8') ''')
cursor.execute('''INSERT INTO MODEL VALUES('GPT-4 Turbo','OpenAI','128','15') ''')
cursor.execute('''INSERT INTO MODEL VALUES('Claude 3','Anthropic','200','30') ''')
cursor.execute('''INSERT INTO MODEL VALUES('Command R','Cohere','128','6') ''')
cursor.execute('''INSERT INTO MODEL VALUES('Llama 3 70B','Meta AI','8','1') ''')
cursor.execute('''INSERT INTO MODEL VALUES('Mistral Large','Mistral AI','32','12') ''')
cursor.execute('''INSERT INTO MODEL VALUES('DBRX','Databricks','32','2') ''')

print("The LLM models are ")
models=cursor.execute('''SELECT * FROM MODEL''')

for row in models:
    print(row)

connection.commit()
connection.close