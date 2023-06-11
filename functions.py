# Define the cost functions for each generator
def marginal_price(tech_data, generator,co2_price):
    """
    Returns the marginal price for the given generator in €/MWh, based on the fuel price, effficiency, further variable costs, and carbon emissions of that generator

    Parameters:
        generator: generator in question
        co2_price: CO2 price in €/tCO2

    Returns:
        marginal_price: marginal price in €/MWh
    """
    df = tech_data[tech_data['tech'] == generator]
    return round((df['fuel_p']/df['eta'] + df['c_var_other'] + co2_price*df['emf']/df['eta']).values[0],2)