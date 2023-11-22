from fractions import Fraction
from oracle import oracleTopd

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


