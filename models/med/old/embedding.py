from fractions import Fraction
from oracle import oracleTopd
from tqdm import tqdm
import ast
import re
import pandas as pd

def convert_fraction_to_float(quantity):
    try:
        return float(Fraction(quantity))
    except ValueError:
        return None 
    
def convert_unit_to_number(unit):
    unit_conversion = {
        'g': 1,
        '개': 100,
        '조금' :10
    }
    return unit_conversion.get(unit, 1)

def slicefood(data):
    # recipe_ingredients가 NA인 행 제거
    toy = data.loc[data["recipe_ingredients"].notna(), :]

    # 문자열 전처리
    toy["recipe_ingredients"] = toy["recipe_ingredients"].apply(lambda x: x.replace('\\ufeff', '').replace('\\u200b', ''))

    # 새로운 칼럼 생성
    for i in range(1, 21):
        toy.loc[:, f'ingredient{i}'] = None
        toy.loc[:, f'quantity{i}'] = None
        toy.loc[:, f'unit{i}'] = None

    # 패턴과 일치하지 않는 데이터를 저장할 딕셔너리
    non_matching_items = {}

    # 식재료 칼럼 쪼개기
    for idx, row in tqdm(toy.iterrows(), total=toy.shape[0]):
        ingredients_dict = ast.literal_eval(row["recipe_ingredients"])
        ingredient_count = 1
        for category, items in ingredients_dict.items():
            if items:
                for item in items:
                    match = re.match(r'([가-힣]+(\([가-힣]+\))?)([\d.+/~-]*)([가-힣a-zA-Z]+|약간|조금)?', item)
                    if match:
                        ingredient, _, quantity, unit = match.groups()

                        toy.at[idx, f'ingredient{ingredient_count}'] = ingredient
                        toy.at[idx, f'quantity{ingredient_count}'] = quantity
                        toy.at[idx, f'unit{ingredient_count}'] = unit

                        ingredient_count += 1
                    else:
                        non_matching_items[idx] = item

    # 패턴과 일치하지 않는 데이터 출력
    if non_matching_items:
        for idx, item in non_matching_items.items():
            print(f'Row {idx}: {item}')

    return toy
