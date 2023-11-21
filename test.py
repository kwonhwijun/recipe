def oracleTopd(query):    
    import oracledb as od
    import pandas as pd

    # db connection
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
    conn = od.connect(user='admin', password='INISW2inisw2', dsn='inisw2_high')
    exe = conn.cursor()
    exe.execute(query)
    
    row = exe.fetchall() # row 불러오기

    column_name = exe.description # column 불러오기
    columns=[]
    
    for i in column_name:
        columns.append(i[0])
    
    # row, column을 pandas DataFrame으로 나타내기
    result = pd.DataFrame(row, columns=columns)
    
    # dtype clob을 string으로 변환
    for clob in result.columns:
        if result[clob].dtype == object:
            result[clob] = result[clob].astype("string")
    
    conn.close()
    
    return result

query = 'select * from recipe_data_table where rownum <= 100'
df = oracleTopd(query)
df

query = 'select * from nutrient_filter_table where rownum <= 100'
df = oracleTopd(query)
df

