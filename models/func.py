#0. 데이터 불러오기
#0.5 데이터 전처리
#1. 식재료 단위 별로 쪼개기
#2. 단위에 따른 g수 +계산
#3. 각 레시피의 영양소 할당
#4. 레시피를 MATRIX로 바꾸는데(1. 레시피*식재료  레시피*영양소)
#5. Matrix 3개를 svd

import oracledb as od
import pandas as pd
import numpy as np
import config
from tqdm import tqdm
import ast
import re
from sklearn.decomposition import TruncatedSVD


#0. 데이터 불러오기
def load_recipe(n =1000):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12") # db connection
    conn = od.connect(user = config.DB_CONFIG['user'], password = config.DB_CONFIG['password'],  dsn = config.DB_CONFIG['dsn'])
    exe = conn.cursor()
    exe.execute(f'select * from recipe_table where rownum <= {n}')
    row = exe.fetchall() # row 불러오기
    column_name = exe.description # column 불러오기
    columns=[]
    for i in column_name:
        columns.append(i[0])
    result = pd.DataFrame(row, columns=columns) # row, column을 pandas DataFrame으로 나타내기
    result.rename(mapper=str.lower, axis='columns', inplace=True)
    conn.close()
    return result

def recipe_preprocessing(raw):
    data = raw.loc[raw['recipe_ingredients'].notnull()].copy()  # None 값 제거

    def clean_ingredients(ingredients):
        if ingredients is not None:
            ingredients = ingredients.replace('\\ufeff', '').replace('\\u200b', '')
        return ingredients
    data["recipe_ingredients"] = data["recipe_ingredients"].apply(clean_ingredients)
    result = data[['recipe_title', 'recipe_ingredients']]

    return result

#1. 식재료 단위 별로 쪼개기
def split_ingredient(data):
    for i in range(1, 21):
        data.loc[:, f'ingredient{i}'] = None
        data.loc[:, f'quantity{i}'] = None
        data.loc[:, f'unit{i}'] = None
   


    # 패턴과 일치하지 않는 데이터를 저장할 딕셔너리
    non_matching_items = {}
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]): #tqdm으로 진행상황 확인
        if row['recipe_ingredients']:
            ingredients_dict = ast.literal_eval(row["recipe_ingredients"]) #딕셔너리 형태로 저장된 recipe_ingredients 불러오기
            ingredient_count = 1
            for category, items in ingredients_dict.items(): #category : 재료, 양념재료, items: 사과1개, 돼지고기600g
                if items:  # 아이템이 존재하는 경우
                    for item in items:
                        match = re.match(r'([가-힣a-zA-Z]+(\([가-힣]+\))?)([\d.+/~-]*)([가-힣a-zA-Z]+|약간|조금)?', item) # 정규식
                        if match:
                            ingredient, _, quantity, unit = match.groups()
                            
                            data.loc[idx, f'ingredient{ingredient_count}'] = ingredient
                            data.loc[idx, f'quantity{ingredient_count}'] = quantity
                            data.loc[idx, f'unit{ingredient_count}'] = unit

                            ingredient_count += 1
                        else:
                            # 패턴과 일치하지 않는 경우 딕셔너리에 추가
                            non_matching_items[idx] = item
        else:
            pass

    # 패턴과 일치하지 않는 데이터 출력
    for idx, item in non_matching_items.items():
        print(f'Row {idx}: {item}')
    return data

# 4. Matrix 변환

def recipe_food_matrix(data):
    def convert_fraction_to_float(quantity):
        from fractions import Fraction

        try:
            return float(Fraction(quantity))
        except ValueError:
            return None 
    def convert_unit_to_number(unit):
        '''
        단위에 따른 g 수 변환
        '''
        unit_conversion = {
            'g': 1,
            '개': 100,
            '조금' :10
        }
        return unit_conversion.get(unit, 1)
    ingredient_columns = data.filter(like='ingredient').drop(columns=['recipe_ingredients'])
    all_ingredients = [item for sublist in ingredient_columns.values for item in sublist if pd.notna(item)] 
    all_ingredients = set()
    for i in range(1, 21):  
        all_ingredients.update(data[f'ingredient{i}'].dropna().unique())

    recipe_ingredients_df = pd.DataFrame(columns=list(all_ingredients))

    recipe_rows = []
    for idx, row in data.iterrows():
        recipe_data = {ingredient: 0.0 for ingredient in all_ingredients}  # 모든 식재료를 None으로 초기화
        for i in range(1, 21):  
            ingredient = row[f'ingredient{i}']
            quantity = row[f'quantity{i}']
            unit = row[f'unit{i}']
            if pd.notna(ingredient) and pd.notna(quantity):
                quantity_float = convert_fraction_to_float(quantity)
                if quantity_float is not None:
                    unit_number = convert_unit_to_number(unit) if pd.notna(unit) else 1
                    recipe_data[ingredient] = quantity_float * unit_number
        recipe_rows.append(recipe_data)

    # 새로운 데이터프레임 생성 (모든 식재료를 열로 가짐)
    recipe_ingredients_df = pd.concat([pd.DataFrame([row]) for row in recipe_rows], ignore_index=True)

    recipe_ingredients_df = recipe_ingredients_df.astype('float64')

    return recipe_ingredients_df



# 영양소 기반 SVD
def nutri_svd(df, n): # df = 입력할 테이블, n = 차원수


    nutrients_df = df.drop(columns=['recipe_title'])
    matrix = nutrients_df.to_numpy()

    svd = TruncatedSVD(n_components=n)
    result = svd.fit_transform(matrix)
    return result

# 예시
# nutri_embedded_recipe = nutri_svd(df, 20)

# 코사인 유사도 기반 레시피 나열
def recipe_cos(df, result, index): # df = 테이블, result = 특정 차원으로 표현된 레시피 array, index = 기준 인덱스
    target_vector = result[index]
    # 타겟 벡터를 2D 배열로 변환
    target_vector = target_vector.reshape(1, -1)
    # 코사인 유사도 계산
    similarities = cosine_similarity(result, target_vector)

    # 데이터프레임 생성
    similarity = pd.DataFrame(similarities, columns=['Similarity'])
    # 'Similarity' 열을 기준으로 내림차순 정렬
    sorted_df = similarity.sort_values(by='Similarity', ascending=False)
    # df는 데이터프레임 객체, 'Similarity'는 컬럼명으로 가정합니다.
    indexes = sorted_df.index.tolist()
    
    selected_titles = df.loc[indexes, 'recipe_title']
    return selected_titles

# 예시
# sorted_recipe = recipe_cos(df, nutri_embedded_recipe, 1)

    
# 실습
# raw_data = load_recipe(n=10000)
# recipe = recipe_preprocessing(raw_data)
# split_ingredient(recipe)


## 기타 함수
# -단위의 개수 세는 함수
# -식재료 종류 세느 ㄴ함수 
