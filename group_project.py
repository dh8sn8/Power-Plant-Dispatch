import pyomo.environ as pyo
import pandas as pd
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory

#Import function defined in functions.py
# ...import * if alle function in functions.py shpuld be imported
from functions import marginal_price


# Load input data
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

# Reset index and change dataframes object type to numeric value)
tech_data.reset_index(drop=True, inplace=True)
tech_data[["cap_MW", "eta", "fuel_p", "c_var_other", "emf"]] = tech_data[["cap_MW", "eta", "fuel_p", "c_var_other", "emf"]].apply(pd.to_numeric)

# Create a concrete Pyomo model
m = pyo.ConcreteModel()
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) #specify dual variables (shadow prices)

# Create a set S for the technologies
m.S = pyo.Set(initialize=tech_data['tech'])
print(m.S)

# Define the decision variable for each generator
m.generators = Var(m.S, domain=pyo.NonNegativeReals)
#print(m.generators)

# Calculate the marginal costs wit no CO2 price
tech_data_noCO2price = tech_data.copy()
tech_data_noCO2price['marginal_cost'] = round(tech_data_noCO2price['fuel_p']/tech_data_noCO2price['eta']+tech_data_noCO2price['c_var_other'],2)
#print(tech_data_noCO2price)

# Clone the concrete model `m` for no CO2 price model m_0
m_0 = m.clone()

# Define the objective function
m_0.cost = Objective(expr=sum(marginal_price(tech_data,s,0)*m_0.generators[s] for s in m_0.S),
                      sense=pyo.minimize)

# Create empty dictionary
models = {}

for i in range(len(duration)):
    models[duration['timestep'][i]] = m_0.clone() #clone original model for each of the timesteps

# Create a list of timesteps
timesteps = list(models.keys())

# Each timestep now has model
print(models)

for i in timesteps:
    model = models[i]
    # add generator limit: capacity*cf at a given timestep

    @model.Constraint(model.S)
    def generator_limit(model, s):
        return model.generators[s] <= cf[cf['tech'] == s][i].values[0] * tech_data[tech_data['tech'] == s].cap_MW.values[0]

        models[i] = model

    # add demand constraint: sum of generation must equal demand
    models[i].demand_constraint = Constraint(
        expr=sum(models[i].generators[s] for s in models[i].S) == demand[demand.timestep == i].load_MW.values[0])

for i in timesteps:
    SolverFactory('cbc').solve(models[i]).write()

# Extract optimum dispatch in MW for each timestep from the results
dispatch = pd.Series(models[timesteps[0]].generators.get_values(),name=timesteps[0]).to_frame()
for i in timesteps[1:]:
    d = pd.Series(models[i].generators.get_values(),name=i).to_frame()
    dispatch = dispatch.join(d)
print(dispatch)

