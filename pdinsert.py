# oracledb에 insert 시키기
# df는 pandas data frame
ln = range(1,len(df.index)) # row 개수 range로 저장
row = list(ln) # list형 [1~index] 

import oracledb as od
import config

# db connection
od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12")
conn = od.connect(user = config.DB_CONFIG['user'], password = config.DB_CONFIG['password'],
                            dsn = config.DB_CONFIG['dsn'])
exe = conn.cursor()

for i in row:
    data1 = df.iloc[i]['recipe_title']
    data2 = df.iloc[i]['recipe_step']
    data3 = df.iloc[i]['recipe_ingredients'] # insert 하고싶은 컬럼명들 작성. 컬럼개수 추가 삭제도 가능
    query = "insert into python_to_oracle values(:1, :2, :3)" # 여기서 원하는대로 insert문 수정
    data = [data1, data2, data3] # 컬럼 개수만큼
    exe.execute(query, data) # 형식 맞춰야함
    # "insert into 테이블명 values(:1, :2, :3)", [data1, data2, data3]
    # df 전체 컬럼이 아닌 일부만 insert 하고 싶은 경우
    # "insert into 테이블명(컬럼명1, 컬럼명2, ....) values(:1, :2, ...)", [data1, data2,...]
    
exe.execute("commit") # commit을 해야 db에 저장됨

exe.close()
conn.close()