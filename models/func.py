#0. 데이터 불러오기
#0.5 데이터 전처리
#1. 식재료 단위 별로 쪼개기
#2. 단위에 따른 g수 계산
#3. 각 레시피의 영양소 할당
#4. 레시피를 MATRIX로 바꾸는데(1. 레시피*식재료  레시피*영양소)
#5. Matrix 3개를 svd

#0. 데이터 불러오기
def load_recipe(n = 1000): 
    '''
    레시피 데이터 불러오는 함수 (나중에 오라클에서 직접 가져오도록 바꿔야 함)

    n : int 
        불러오고 싶은 레시피의 수 (Defalut 1000)
    '''
    import pandas as pd

    raw = pd.read_csv(r'models/data/RecipeData.csv')
    data = raw.head(n = n).copy()
    return data

#0.5 전처리
def recipe_preprocessing(raw) :
    # 이상한 문자열 제거
    raw["recipe_ingredients"] = raw["recipe_ingredients"].apply(lambda x: x.replace('\\ufeff', '').replace('\\u200b', ''))
    raw = raw[['recipe_title', 'recipe_ingredients']]

    return raw

#1. 식재료 단위 별로 쪼개기
def split_ingredient(data):
    from tqdm import tqdm
    import ast
    import re

    for i in range(1, 21):
        data.loc[:, f'ingredient{i}'] = None
        data.loc[:, f'quantity{i}'] = None
        data.loc[:, f'unit{i}'] = None
    
    # 패턴과 일치하지 않는 데이터를 저장할 딕셔너리
    non_matching_items = {}
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]): #tqdm으로 진행상황 확인
        ingredients_dict = ast.literal_eval(row["recipe_ingredients"])
        ingredient_count = 1
        for category, items in ingredients_dict.items():
            if items:  # 아이템이 존재하는 경우
                for item in items:
                    match = re.match(r'([가-힣]+(\([가-힣]+\))?)([\d.+/~-]*)([가-힣a-zA-Z]+|약간|조금)?', item)
                    if match:
                        ingredient, _, quantity, unit = match.groups()
                        
                        data.at[idx, f'ingredient{ingredient_count}'] = ingredient
                        data.at[idx, f'quantity{ingredient_count}'] = quantity
                        data.at[idx, f'unit{ingredient_count}'] = unit

                        ingredient_count += 1
                    else:
                        # 패턴과 일치하지 않는 경우 딕셔너리에 추가
                        non_matching_items[idx] = item

    # 패턴과 일치하지 않는 데이터 출력
    for idx, item in non_matching_items.items():
        print(f'Row {idx}: {item}')
    return data

# 실습
# raw_data = load_recipe(n=10000)
# recipe = recipe_preprocessing(raw_data)
# split_ingredient(recipe)
