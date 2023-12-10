# 레시피 관련 함수
import func_old
import pandas as pd
import oracledb as od
import config
from fractions import Fraction

import sys
sys.path.append('.')


def load_recipe(path = 'csv', rows = 1000):
    if path == "csv":   
        data = pd.read_csv(r'models\core\process_ingre.csv')
        result =  data.iloc[:rows]

    if path == "oracle":
        data = func.load_recipe(n = rows)
        data2 = func.recipe_preprocessing(data)
        result = func.split_ingredient(data2)

    return result

test2 = load_recipe(path = 'oracle')




# 분수를 숫자로 convert_fracion_to_float(1/2) = 0.5)
def convert_fraction_to_float(quantity):
    try:
        return float(Fraction(quantity))
    except ValueError:
        return None 
        
# 단위를 g으로 : convert_unit_to_number('조금') = 10
def convert_unit_to_number(unit):
        unit_conversion = {
            'g': 1,
            '개': 100,
            '조금' :10
        }
        return unit_conversion.get(unit, 1)