

def recipe_food_matrix(data):
    data.index = range(len(data)) # index 초기화

    
    
    # all_ingredients: 모든 식재료 리스트
    ingredient_columns = data.filter(like='ingredient').drop(columns=['recipe_ingredients'])
    all_ingredients = set()
    for i in range(1, 75):  
        all_ingredients.update(data[f'ingredient{i}'].dropna().unique())

    # recipe_ingredients_df: 비어있는 레시피 X 식재료 df
    col_name = ['recipe_title'].append(list(all_ingredients))
    recipe_ingredients_df = pd.DataFrame(columns=col_name)

    # 레시피 하나씩 붙이기 
    recipe_rows = []
    for idx, row in tqdm(data.iterrows(), total = data.shape[0]) : # tqdm으로 진행상황 확인
        recipe_data = {ingredient: 0.0 for ingredient in all_ingredients}  # 모든 식재료를 None으로 초기화
        for i in range(1, 50):  
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
    recipe_ingredients_df['recipe_title'] = data['recipe_title']

    # RECIPE_TITLE 컬럼을 젤 앞으로
    recipe_ingredients_df = recipe_ingredients_df[['recipe_title'] + [col for col in recipe_ingredients_df.columns if col != 'recipe_title']]

    return recipe_ingredients_df