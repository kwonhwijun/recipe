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
        ingredients_dict = ast.literal_eval(row["recipe_ingredients"]) #딕셔너리 형태로 저장된 recipe_ingredients 불러오기
        ingredient_count = 1
        for category, items in ingredients_dict.items(): #category : 재료, 양념재료, items: 사과1개, 돼지고기600g
            if items:  # 아이템이 존재하는 경우
                for item in items:
                    match = re.match(r'([가-힣a-zA-Z]+(\([가-힣]+\))?)([\d.+/~-]*)([가-힣a-zA-Z]+|약간|조금)?', item) # 정규식
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


def slicefood(data):
    from oracle import oracleTopd
    from tqdm import tqdm
    import ast
    import re
    import pandas as pd

    def convert_fraction_to_float(quantity):
        from fractions import Fraction

        try:
            return float(Fraction(quantity))
        except ValueError:
            return None 
    
    # 단위에 따른 g 수
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
    
    # recipe_ingredients가 NA인 행 제거
    toy = data.copy()

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


# 영양소 기반 SVD
def nutri_svd(df, n): # df = 입력할 테이블, n = 차원수
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    nutrients_df = df.drop(columns=['recipe_title'])
    matrix = nutrients_df.to_numpy()

    svd = TruncatedSVD(n_components=n)
    result = svd.fit_transform(matrix)
    return result

# 예시
nutri_embedded_recipe = nutri_svd(df, 20)

# 코사인 유사도 기반 레시피 나열
def recipe_cos(df, result, index): # df = 테이블, result = 특정 차원으로 표현된 레시피 array, index = 기준 인덱스
        
    import pandas as pd
    import numpy as np
    
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
sorted_recipe = recipe_cos(df, nutri_embedded_recipe, 1)

    
# 실습
# raw_data = load_recipe(n=10000)
# recipe = recipe_preprocessing(raw_data)
# split_ingredient(recipe)







## 기타 함수
# -단위의 개수 세는 함수
# -식재료 종류 세느 ㄴ함수 
