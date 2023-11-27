
#1. 식재료 단위 별로 쪼개기
#2. 단위에 따른 g수 계산
#3. 각 레시피의 영양소 할당
#4. 레시피를 MATRIX로 바꾸는데(1. 레시피*식재료  레시피*영양소)
#5. Matrix 3개를 svd

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



def load_recipe():
    data = oracleTopd('SELECT * FROM RECIPE_TABLE WHERE ROWNUM <100')

    return data

recipe = load_recipe()



print(recipe.head(3))


def a2(data):
    return data


