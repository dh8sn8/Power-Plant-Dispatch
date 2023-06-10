import pyomo.environ as pyo
import pandas as pd
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory

#Load input data
demand = pd.read_csv('load.csv',
                     names=['timestep','load_MW'])

cf_names=['tech',"t1", "t2", "t3", "t4", "t5","t6", "t7", "t8", "t9", "t10"]
cf = pd.read_csv('capacity_factors.csv',
                 header=0,
                 names=cf_names)

duration = pd.read_csv('duration.csv',
                       names=['timestep','length'])

tech_names=['tech','cap_MW', 'eta', 'fuel_p', 'c_var_other', 'emf']
tech_data = pd.read_csv('tech_data.csv',
                        header=0,
                        names=tech_names).drop([0])

# reset index and change dataframes object type to numeric value)
tech_data.reset_index(drop=True, inplace=True)
tech_data[["cap_MW", "eta", "fuel_p", "c_var_other", "emf"]] = tech_data[["cap_MW", "eta", "fuel_p", "c_var_other", "emf"]].apply(pd.to_numeric)

# Create a concrete Pyomo model
m = pyo.ConcreteModel()
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) #specify dual variables (shadow prices)

#Create a set S for the technologies
m.S = pyo.Set(initialize=tech_data['tech'])
m.S.pprint()

# Define the decision variable for each generator
m.generators = Var(m.S, domain=pyo.NonNegativeReals)
m.generators.pprint()

# Define the cost functions for each generator

def marginal_price(generator,co2_price):
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