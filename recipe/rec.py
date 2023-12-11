# 레시피 데이터에 관한 파일
# recipe 전처리 1) 문자열 쪼개기 2) 개수 할당해주기 3) 영양소 할당해주기 4) 날리기

import db
import oracledb as od
import pandas as pd
import numpy as np
import config
from tqdm import tqdm
import ast
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import datetime

def load_recipe(num = 1000) :
    return db.load_data(data = 'recipe', n = num)

def load_nutri(num = 500):
    return db.load_data(data = 'nutri', n = num)


def recipe_preprocessing(raw):
    data = raw.loc[raw['recipe_ingredients'].notnull()].copy()  # None 값 제거
    def clean_ingredients(ingredients):
        if ingredients is not None:
            ingredients = ingredients.replace('\\ufeff', '').replace('\\u200b', '')
        return ingredients
    
    # recipe_ingredinents가 비어있지 않은 행만 남기기
    def not_empty_ingredients(row):
        return row['recipe_ingredients'].strip() != '{}' 

    data["recipe_ingredients"] = data["recipe_ingredients"].apply(clean_ingredients)
    data = data[data.apply(not_empty_ingredients, axis=1)]
    result = data[['recipe_title', 'recipe_ingredients', 'recipe_step']].copy()

    title_idx = result[result['recipe_title'].isnull()].index # title이 null값인 행 인덱스 찾기
    del_idx = result[result['recipe_ingredients'].str.startswith('소시지')].index #소시지~ 로 시작해서 오류 일으키는 행 인덱스 찾기
    result.drop(del_idx, inplace=True) # 오류 일으키는 행 제거
    result.drop(title_idx, inplace=True) # title null값인 행 제거
    result = result.drop_duplicates() # 중복 제거

    return result

def split_ingredient(data):
    num_ingredients = 74

    column_names = [f'ingredient{i}' for i in range(1, num_ingredients + 1)] + \
               [f'quantity{i}' for i in range(1, num_ingredients + 1)] + \
               [f'unit{i}' for i in range(1, num_ingredients + 1)]
    empty_columns = pd.DataFrame(columns=column_names)
    data = pd.concat([data, empty_columns], axis=1)


    non_matching_items = {} # 패턴과 일치하지 않는 데이터를 저장할 딕셔너리

    for idx, row in tqdm(data.iterrows(), total=data.shape[0]): #tqdm으로 진행상황 확인
        if row['recipe_ingredients']:
            ingredients_dict = ast.literal_eval(row["recipe_ingredients"]) #딕셔너리 형태로 저장된 recipe_ingredients 불러오기
            ingredient_count = 1

            for items in ingredients_dict.values():
                for item in items:
                    match = re.match(r'([가-힣a-zA-Z]+(\([가-힣a-zA-Z]+\))?|\d+[가-힣a-zA-Z]*|\([가-힣a-zA-Z]+\)[가-힣a-zA-Z]+)([\d.+/~-]*)([가-힣a-zA-Z]+|약간|조금)?', item)

                    if match:
                        ingredient, _, quantity, unit = match.groups()

                        data.at[idx, f'ingredient{ingredient_count}'] = ingredient
                        data.at[idx, f'quantity{ingredient_count}'] = quantity
                        data.at[idx, f'unit{ingredient_count}'] = unit

                        ingredient_count += 1
                    else:
                        non_matching_items[idx] = item

    data = data.drop([k for k, v in non_matching_items.items() if v != ''])

    #i가 75 이상인 경우 제거하는 조건문
    data = data.copy()

    columns_to_drop = []
    for i in range(data.shape[1]):
        if i >= 75:
            column_prefixes = [f'ingredient{i}', f'quantity{i}', f'unit{i}']
            columns_to_drop.extend(column_prefixes)

    # 실제로 데이터프레임에 존재하는 열만 삭제
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(existing_columns_to_drop, axis=1, inplace=True)

    return data


def process_ingredient(dataframe):
    dataframe = dataframe.copy()
    def process_pattern(dataframe, pattern, replacement):
        for i in range(1, 75):
            col_name = f'ingredient{i}'
            unit_col_name = f'unit{i}'
            dataframe[unit_col_name] = np.where(dataframe[col_name].notna() & dataframe[col_name].str.contains(pattern, regex=True), replacement, dataframe[unit_col_name])
            dataframe[col_name] = dataframe[col_name].str.replace(pattern, '', regex=True)

        dataframe = dataframe.drop_duplicates()

        return dataframe

    # '약간', '적당량', '조금', '톡톡', '적당히' 패턴 처리
    dataframe = process_pattern(dataframe, r'약간', '약간')
    dataframe = process_pattern(dataframe, r'적당량', '적당량')
    dataframe = process_pattern(dataframe, r'적당히', '적당량')
    dataframe = process_pattern(dataframe, r'적당양', '적당량')
    dataframe = process_pattern(dataframe, r'조금.*', '조금')
    dataframe = process_pattern(dataframe, r'톡톡(톡)?', '톡톡')

    # 괄호 제거
    for i in range(1, 75):
        col_name = f'ingredient{i}'
        dataframe[col_name] = dataframe[col_name].str.replace(r'\([^)]*\)', '', regex=True)
        dataframe = dataframe.drop_duplicates() # 중복 제거

    return dataframe


