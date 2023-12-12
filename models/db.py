import oracledb as od
import pandas as pd
import config

#print(load_data(data ='recipe', n = 100))
#print(load_data(data ='nutri', n = 100))

def load_data(data = 'recipe', n=1000):
    if data =='recipe' :
        od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12") # DB 연결
        conn = od.connect(user=config.DB_CONFIG['user'], password=config.DB_CONFIG['password'], dsn=config.DB_CONFIG['dsn'])
        exe = conn.cursor()
        exe.execute(f'SELECT * FROM (SELECT * FROM recipe_table ORDER BY row_cnt ASC) WHERE row_cnt <= {n}')
        result = pd.DataFrame(exe.fetchall(), columns=[col[0].lower() for col in exe.description])  # row와 column 이름을 가져와 DataFrame 생성
        conn.close() #실험 # 수정
        return result
    
    elif data == 'nutri' :
        od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12") # DB 연결
        conn = od.connect(user=config.DB_CONFIG['user'], password=config.DB_CONFIG['password'], dsn=config.DB_CONFIG['dsn'])
        exe = conn.cursor()
        query = f'SELECT * FROM (SELECT d.*, ROW_NUMBER() OVER (ORDER BY NUTRIENT) AS rnum FROM NUTRIENT_DATA_TABLE d) WHERE rnum <= {n}'
        exe.execute(query)
        result = pd.DataFrame(exe.fetchall(), columns=[col[0].lower() for col in exe.description])  # row와 column 이름을 가져와 DataFrame 생성
        conn.close() #실험 # 수정
        return result
    
    else: pass




def df2oracle(dataframe, table_name):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
    conn = od.connect(user = config.DB_CONFIG['user'], password = config.DB_CONFIG['password'], dsn = config.DB_CONFIG['dsn'])

    columns_info = ', '.join([f'"{col}" VARCHAR2(300)' for col in dataframe.columns.astype(str)])
    # 테이블 생성
    create_table_sql = f"CREATE TABLE {table_name} ({columns_info})"
    exe = conn.cursor()
    exe.execute(create_table_sql)
    exe.close()
    # oracle db에 insert 
    def insert_into_oracle(dataframe, table_name, conn):    
        exe = conn.cursor()    
        
        insert = [tuple(x) for x in dataframe.values]
        exe.executemany(
            f"INSERT INTO {table_name} VALUES ({','.join([':' + str(i+1) for i in range(len(dataframe.columns))])})",
            insert)
        
        conn.commit()
        exe.close()
    
    insert_into_oracle(dataframe, table_name, conn)
    conn.close()

data = pd.read_csv(r'C:\Users\HwijunKwon\github\recipe\models\data\info_list.csv')
data.columns.astype(str)
df2oracle(data, 'test2')