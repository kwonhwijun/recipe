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

#0. 데이터 불러오기
def load_recipe(n =1000):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12") # db connection
    conn = od.connect(user = config.DB_CONFIG['user'], password = config.DB_CONFIG['password'],  dsn = config.DB_CONFIG['dsn'])
    exe = conn.cursor()
    exe.execute(f'select * from (select * from recipe_table order by row_cnt asc) where row_cnt <= {n}')
    row = exe.fetchall() # row 불러오기
    column_name = exe.description # column 불러오기
    columns=[]
    for i in column_name:
        columns.append(i[0])
    result = pd.DataFrame(row, columns=columns) # row, column을 pandas DataFrame으로 나타내기
    result.rename(mapper=str.lower, axis='columns', inplace=True)
    conn.close()
    return result

# query문 직접 작성해서 select 할때 사용
def select_table(query):
    od.init_oracle_client(lib_dir=r"C:\Program Files\Oracle\instantclient_21_12") # db connection
    conn = od.connect(user = config.DB_CONFIG['user'], password = config.DB_CONFIG['password'],  dsn = config.DB_CONFIG['dsn'])
    exe = conn.cursor()
    exe.execute(query)
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

    def not_empty_ingredients(row):
        return row['recipe_ingredients'].strip() != '{}' # 결측치 제거

    data["recipe_ingredients"] = data["recipe_ingredients"].apply(clean_ingredients)
    data = data[data.apply(not_empty_ingredients, axis=1)]
    result = data[['recipe_title', 'recipe_ingredients']]

    title_idx = result[result['recipe_title'].isnull()].index
    del_idx = result[result['recipe_ingredients'].str.startswith('소시지')].index
    result.drop(del_idx, inplace=True)
    result.drop(title_idx, inplace=True)

    return result

#1. 식재료 단위 별로 쪼개기
def split_ingredient(data):
    for i in range(1, 21):
        data.loc[:, f'ingredient{i}'] = None
        data.loc[:, f'quantity{i}'] = None
        data.loc[:, f'unit{i}'] = None
   
    non_matching_items = {} # 패턴과 일치하지 않는 데이터를 저장할 딕셔너리

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

    # 패턴과 일치하지 않는 데이터 출력 X => 날려버리기!
    for idx, item in non_matching_items.items():
        print(f'Row {idx}: {item}')
    return data

    #재료가 ingredient1부터 안 들어가서 null값인 거 날려버리기!

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
    recipe_ingredients_df['RECIPE_TITLE'] = data['recipe_title']
    
    # RECIPE_TITLE 컬럼을 젤 앞으로
    recipe_ingredients_df = recipe_ingredients_df[['RECIPE_TITLE'] + [col for col in recipe_ingredients_df.columns if col != 'RECIPE_TITLE']]

    return recipe_ingredients_df

#---------------------------------------------------------------------------------------------------#
# 재료 쪼갠 후 레시피별 영양소 나오는 테이블
def recipe_nutri(new_recipe1, nutri_df):
    # txt 파일 경로 (딕셔너리 수정시 수정 필요함)
    file_path = r"C:\Users\admin\OneDrive\바탕 화면\change.txt"

    unit_conversion = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split()
            unit = line[0]
            value = line[1] if line[1].isdigit() else None
            unit_conversion[unit] = value

    # DataFrame 생성
    df11 = pd.DataFrame(list(unit_conversion.items()), columns=['unit', 'value'])

    # DataFrame을 딕셔너리로 변환
    df11_dict = df11.set_index('unit')['value'].to_dict()

    # 딕셔너리의 값을 숫자로 변환하여 새로운 딕셔너리 생성
    df_dict = {key: int(value) if value is not None else None for key, value in df11_dict.items()}
    df_dict

    # unit{i} 컬럼에 딕셔너리로 지정한 key : value값으로 치환
    for i in range(1, 15):
        column_name = f'unit{i}'
        if column_name in new_recipe1.columns:
            new_recipe1[column_name] = new_recipe1[column_name].apply(lambda x: df_dict.get(x) if pd.notna(x) and not re.match(r'\d+[^\d]*$', str(x)) else x)
    
    # new_recipe1에 recipe_title, ingredient{i}, quantity{i}, unit{i}만 저장
    new_recipe1 = new_recipe1[['recipe_title'] + [f'{name}{i}' for i in range(1, 14) for name in ['ingredient', 'quantity', 'unit']]]
 
    # 계산을 위해 quantity의 타입변경 str => float
    for i in range(1, 14):
        try:
            new_recipe1[f'quantity{i}'] = pd.to_numeric(new_recipe1[f'quantity{i}'], errors='coerce').astype('float16')
        except ValueError:
            new_recipe1[f'quantity{i}'] = 0
    
    # mulit{i} 컬럼 생성 후 quantity * unit 값 대입
    for i in range(1,14):
        new_recipe1[f'multi{i}'] = None
        
    for i in range(1, 14):
        new_recipe1[f'multi{i}'] = new_recipe1[f'quantity{i}'] * new_recipe1[f'unit{i}']
        
    # quantity, unit 컬럼 전부 삭제
    for i in range(1,14):
        new_recipe1 = new_recipe1.drop(f'quantity{i}',axis = 1)
        new_recipe1 = new_recipe1.drop(f'unit{i}',axis = 1)
    
    # new_recipe1의 컬럼 재배열 (recipe_title ingredient1 multi1 ... 식으로)
    new_columns = [
        'recipe_title', 
        'ingredient1', 'multi1', 
        'ingredient2', 'multi2', 
        'ingredient3', 'multi3', 
        'ingredient4', 'multi4', 
        'ingredient5', 'multi5', 
        'ingredient6', 'multi6', 
        'ingredient7', 'multi7', 
        'ingredient8', 'multi8', 
        'ingredient9', 'multi9', 
        'ingredient10', 'multi10', 
        'ingredient11', 'multi11', 
        'ingredient12', 'multi12', 
        'ingredient13', 'multi13'
    ]
    new_recipe1 = new_recipe1[new_columns]
    
    # 영양소 테이블의 컬럼명 변경
    nutri_df.rename(columns={'대표식품명':'ingredient'}, inplace=True)
    
    nutrient_list = ['에너지(kcal)', '수분(g)', '단백질(g)',
                '지방(g)', '회분(g)', '탄수화물(g)', '당류(g)', '식이섬유(g)', '칼슘(mg)', '철(mg)',
                '인(mg)', '칼륨(mg)', '나트륨(mg)', '비타민a(μg rae)', '레티놀(μg)', '베타카로틴(μg)',
                '티아민(mg)', '리보플라빈(mg)', '니아신(mg)', '비타민c(mg)', '비타민d(μg)', '콜레스테롤(mg)',
                '포화지방산(g)', '트랜스지방산(g)', '폐기율(%)']
    
    # 기존의 new_recipe1테이블(레시피명, 식재료명, multi)를 왼쪽, 영양소 테이블 nutri_df의 ingredient를 오른쪽에 두고 merge
    # for문 안에 merge가 있음. 즉 recipe_title ingredient1 multi1 ingredient 에너지(kcal) 수분(g)...
    # recipe_title ingredient2 multi2 ingredient 에너지(kcal)... 형식을 반복
    # 여기서 ingredient 열은 ingredient{i}와 nutri_df의 ingredient을 비교해서 데이터가 있으면 해당 열에 집어 넣는 형식
    # 예를들면 ingredient13에 참기름이 있는데 nutri_df에 참기름이 있으면 ingredient열에 참기름이 들어가고 
    # 없다면 NaN이 들어감
    
    # for index, row in merged_df.iterrows(): = merged_df를 행 단위로 반복
    # if pd.notna(row['ingredient']) = ingredient열 데이터가 NaN이 아니면 실행
    # multiplier = multi{i}에서 100을 나눈값 (영양소 테이블이 100g 기준이기 때문)
    # 위에 만든 nutrient_list로 for문 실행
    # new_recipe1.at[index, f'{nutrient}{i}'] = row[nutrient] * multiplier
    # = merge된 테이블에서 영양소 부분을 가져와 multiplier값을 곱하고 nutrient{i}에 저장
    # 예를들면 ingredient13이 고추장이고 영양소에서 고추장 에너지(kcal)가 100이면 100*multiplier한 값을
    # 에너지(kcal)13 에 저장하는 방식    
    
    for i in range(1, 14):  # ingredient1부터 ingredient13까지 처리
        ingredient_col = f'ingredient{i}'
        multi_col = f'multi{i}'
        
        # 필요한 컬럼만 추출하여 병합
        merged_df = pd.merge(new_recipe1[['recipe_title', ingredient_col, multi_col]],
                            nutri_df,
                            left_on=ingredient_col,
                            right_on='ingredient',
                            how='left')
        
        # 각 값에 대해 계산
        for index, row in merged_df.iterrows():
            if pd.notna(row['ingredient']): 
                multiplier = row[multi_col] / 100 # row[index]로 변경가능
                for nutrient in nutrient_list:
                    new_recipe1.at[index, f'{nutrient}{i}'] = row[nutrient] * multiplier
            else:
                for nutrient in nutrient_list:
                    new_recipe1.at[index, f'{nutrient}{i}'] = None  # 또는 0 또는 다른 값으로 설정할 수 있음

    # NaN값 제거
    new_recipe1 = new_recipe1.dropna(subset=['recipe_title'])
    
    # ingredient, multi 컬럼 전부 삭제
    for i in range(1,14):
        new_recipe1 = new_recipe1.drop(f'ingredient{i}',axis = 1)
        new_recipe1 = new_recipe1.drop(f'multi{i}',axis = 1)
        
    # 총합 영양소 컬럼 생성
    total_columns = ['총합_에너지(kcal)', '총합_수분(g)', '총합_단백질(g)', '총합_지방(g)', '총합_회분(g)', '총합_탄수화물(g)',
                    '총합_당류(g)', '총합_식이섬유(g)', '총합_칼슘(mg)', '총합_철(mg)', '총합_인(mg)', '총합_칼륨(mg)',
                    '총합_나트륨(mg)', '총합_비타민a(μg rae)', '총합_레티놀(μg)', '총합_베타카로틴(μg)', '총합_티아민(mg)',
                    '총합_리보플라빈(mg)', '총합_니아신(mg)', '총합_비타민c(mg)', '총합_비타민d(μg)', '총합_콜레스테롤(mg)',
                    '총합_포화지방산(g)', '총합_트랜스지방산(g)', '총합_폐기율(%)']

    new_recipe1[total_columns] = 0
    
    # 각 컬럼당 sum값을 방금 만든 총합 ~ 컬럼에 각각 적용
    nutrient_columns = ['에너지(kcal)', '수분(g)', '단백질(g)', '지방(g)', '회분(g)', '탄수화물(g)', '당류(g)',
                        '식이섬유(g)', '칼슘(mg)', '철(mg)', '인(mg)', '칼륨(mg)', '나트륨(mg)', '비타민a(μg rae)',
                        '레티놀(μg)', '베타카로틴(μg)', '티아민(mg)', '리보플라빈(mg)', '니아신(mg)', '비타민c(mg)',
                        '비타민d(μg)', '콜레스테롤(mg)', '포화지방산(g)', '트랜스지방산(g)', '폐기율(%)']

    for nutrient in nutrient_columns:
        new_recipe1[f'총합_{nutrient}'] = new_recipe1[[f'{nutrient}{i}' for i in range(1, 14)]].sum(axis=1)
    
    # 남길 컬럼만 선택
    columns_to_keep = ['recipe_title'] + [f'총합_{nutrient}' for nutrient in nutrient_columns]    
    new_recipe1 = new_recipe1.loc[:, columns_to_keep]
    
    # 소수점 3자리까지만 표시
    new_recipe1 = round(new_recipe1, 3)
    
    return new_recipe1

# def split_ingredient 까지 진행한 df로 사용해야함. recipe_food_matrix 진행 x
# 예시 recipe_nutri(저장한 df명, 영영소 테이블 df명)
#---------------------------------------------------------------------------------------------------#

# 영양소 기반 SVD
def nutri_svd(method, df, n): # method = svd라이브러리 선택df = 입력할 테이블, n = 차원수
    if method == 'sklearn':
        if 'recipe_title' in df.columns:
            nutrients_df = df.drop(columns=['recipe_title'])
        else : 
            nutrients_df = df
        nutrients_df = df.drop(columns=['recipe_title'])
        matrix = nutrients_df.to_numpy()
        svd = TruncatedSVD(n_components=n)
        result = svd.fit_transform(matrix)
        return result

    # scipy
    elif method == 'scipy':
        if 'recipe_title' in df.columns:
            nutrients_df = df.drop(columns=['recipe_title'])
        else :
            nutrients_df = df
        matrix = nutrients_df.to_numpy()
        matrix = matrix.astype(float) 

        num_components = n
        U, Sigma, Vt = svds(matrix, k=num_components)
        matrix_tr = np.dot(np.dot(U,np.diag(Sigma)), Vt)# output of TruncatedSVD
        return U, Sigma, Vt

# 예시
# a,b,c = nutri_svd('scipy', df, 10)

# 식재료 기반 SVD
def food_svd(df, n): # df = 입력할 테이블, n = 차원수
    from sklearn.decomposition import TruncatedSVD
    if 'recipe_title' in df.columns :
        nutrients_df = df.drop(columns=['recipe_title'])
    else :
        nutrients_df = df
    matrix = nutrients_df.to_numpy()

    svd = TruncatedSVD(n_components=n)
    result = svd.fit_transform(matrix)
    return result
    
# 예시
# food_svd_recipe = nutri_svd(df, 20)

# 임베딩 합치기
def add_embedding(method, food_embedded_recipe, nutri_embedded_recipe, dim1, dim2):  # ['add', 'average', 'concat'] 중 하나 입력하면 입력한 방법으로 임베딩 합쳐줌
    if method == 'add':
        result = food_embedded_recipe + nutri_embedded_recipe
        return result
    
    elif method == 'average':
        result = (food_embedded_recipe + nutri_embedded_recipe)/2
        return result

    elif method == 'concat':
        result = np.concatenate((food_embedded_recipe, nutri_embedded_recipe))
        return result

    elif method == 'gate': # 컨캣한 차원, 입력 차원
        import torch
        import numpy as np

        # NumPy 배열을 Tensor로 변환
        food_embedded_recipe_tensor = torch.tensor(food_embedded_recipe).float()
        nutri_embedded_recipe_tensor = torch.tensor(nutri_embedded_recipe).float()

        gate_layer = torch.nn.Linear(dim1, dim2) 
        gate_sigmoid = torch.nn.Sigmoid()

        def gate(A_embedding, B_embedding):
            AB_concat = torch.cat((A_embedding, B_embedding), -1)
            context_gate = gate_sigmoid(gate_layer(AB_concat))
            return torch.add(context_gate * A_embedding, (1. - context_gate) * B_embedding)

        # Tensor로 변환한 데이터를 함수에 입력하고 결과 얻기
        result_tensor = gate(food_embedded_recipe_tensor, nutri_embedded_recipe_tensor)

        # 결과 텐서를 numpy 배열로 변환
        result_numpy = result_tensor.detach().numpy()

        return result_numpy  # 결과의 형태 출력
# 예시
# result = add_embedding('concat', food_embedded_recipe, nutri_embedded_recipe, 40, 20) 

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
