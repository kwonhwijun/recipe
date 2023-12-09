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