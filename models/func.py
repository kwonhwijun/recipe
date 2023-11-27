
#1. 식재료 단위 별로 쪼개기
#2. 단위에 따른 g수 계산
#3. 각 레시피의 영양소 할당
#4. 레시피를 MATRIX로 바꾸는데(1. 레시피*식재료  레시피*영양소)
#5. Matrix 3개를 svd

def load_recipe(n = 1000): 
    import pandas as pd

    raw = pd.read_csv(r'models/data/RecipeData.csv')
    data = raw.head(n = n).copy()
    return data

def recipe_preprocessing(raw) :
    # 이상한 문자열 제거
    raw["recipe_ingredients"] = raw["recipe_ingredients"].apply(lambda x: x.replace('\\ufeff', '').replace('\\u200b', ''))
    raw = raw[['recipe_title', 'recipe_ingredients']]

    return raw


recipe_raw = load_recipe()
recipe = recipe_preprocessing(recipe_raw)
print(recipe.head(5))

def a2(data):
    return data


