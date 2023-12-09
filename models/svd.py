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