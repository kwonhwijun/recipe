def oracleTopd(query):    
    import oracledb as od
    import pandas as pd
    import config

    # db connection
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
    conn = od.connect(user = config.DB_CONFIG['user'], password = config.DB_CONFIG['password'],
                      dsn = config.DB_CONFIG['dsn'])
    exe = conn.cursor()
    exe.execute(query)
    
    row = exe.fetchall() # row 불러오기

    column_name = exe.description # column 불러오기
    columns=[]
    
    for i in column_name:
        columns.append(i[0])
    
    # row, column을 pandas DataFrame으로 나타내기
    result = pd.DataFrame(row, columns=columns)
    result.rename(mapper=str.lower, axis='columns', inplace=True)
    
    # dtype clob을 string으로 변환
    
    conn.close()
    
    return result

#query = 'select * from test_recipe_table'
#df = oracleTopd(query)

#query = 'select * from nutrient_filter_table where rownum <= 100'
# df = oracleTopd(query)
