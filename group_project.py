import pyomo.environ as pyo
import pandas as pd
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
import matplotlib.pyplot as plt
import numpy as np

# Import function defined in functions.py
# ...import * if alle function in functions.py shpuld be imported
from functions import marginal_price

# Load input data
demand = pd.read_csv('load.csv',
                     names=['timestep', 'load_MW'])

cf_names = ['tech', "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]
cf = pd.read_csv('capacity_factors.csv',
                 header=0,
                 names=cf_names)

duration = pd.read_csv('duration.csv',
                       names=['timestep', 'length'])

tech_names = ['tech', 'cap_MW', 'eta', 'fuel_p', 'c_var_other', 'emf']
tech_data = pd.read_csv('tech_data.csv',
                        header=0,
                        names=tech_names).drop([0])

# Reset index and change dataframes object type to numeric value)
tech_data.reset_index(drop=True, inplace=True)
tech_data[["cap_MW", "eta", "fuel_p", "c_var_other", "emf"]] = tech_data[
    ["cap_MW", "eta", "fuel_p", "c_var_other", "emf"]].apply(pd.to_numeric)

# Create a concrete Pyomo model
m = pyo.ConcreteModel()
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)  # specify dual variables (shadow prices)

# Create a set S for the technologies
m.S = pyo.Set(initialize=tech_data['tech'])
# print(m.S)

# Define the decision variable for each generator
m.generators = Var(m.S, domain=pyo.NonNegativeReals)

# Calculate the marginal costs wit no CO2 price
tech_data_noCO2price = tech_data.copy()
tech_data_noCO2price['marginal_cost'] = round(
    tech_data_noCO2price['fuel_p'] / tech_data_noCO2price['eta'] + tech_data_noCO2price['c_var_other'], 2)

# Clone the concrete model `m` for no CO2 price model m_0
m_0 = m.clone()

# Define the objective function
m_0.cost = Objective(expr=sum(marginal_price(tech_data, s, 0) * m_0.generators[s] for s in m_0.S),
                     sense=pyo.minimize)

# Create empty dictionary
models = {}

for i in range(len(duration)):
    models[duration['timestep'][i]] = m_0.clone()  # clone original model for each of the timesteps

# Create a list of timesteps
timesteps = list(models.keys())

# Each timestep now has model
# print(models)

for i in timesteps:
    model = models[i]


    # add generator limit: capacity*cf at a given timestep

    @model.Constraint(model.S)
    def generator_limit(model, s):
        return model.generators[s] <= cf[cf['tech'] == s][i].values[0] * \
            tech_data[tech_data['tech'] == s].cap_MW.values[0]

        models[i] = model


    # add demand constraint: sum of generation must equal demand
    models[i].demand_constraint = Constraint(
        expr=sum(models[i].generators[s] for s in models[i].S) == demand[demand.timestep == i].load_MW.values[0])

for i in timesteps:
    SolverFactory('cbc').solve(models[i]).write()

# Extract optimum dispatch in MW for each timestep from the results
dispatch = pd.Series(models[timesteps[0]].generators.get_values(), name=timesteps[0]).to_frame()
for i in timesteps[1:]:
    d = pd.Series(models[i].generators.get_values(), name=i).to_frame()
    dispatch = dispatch.join(d)
print(dispatch)


def merit_order(dispatch,cap_fac, tech_data, demand):
    cf=cap_fac.get(['t1']).set_index(dispatch.index)
    m_o = tech_data.get(['cap_MW', 'marginal_cost']).set_index(dispatch.index)
    m_o['available_cap'] = cf['t1'] * m_o['cap_MW']

    color_list = ['#b20101', '#d35050', '#08ad97', '#707070', '#9e5a01', '#ff9000', '#235ebc', '#f9d002']
    color_df = pd.DataFrame({'tech': m_o.index,
                             'colors': color_list}).set_index('tech')
    #m_o = m_o.join(color_df).sort_values('marginal_cost')
    m_o = m_o.join(color_df).sort_values('marginal_cost')
    m_o['cumulative_cap'] = m_o['available_cap'].cumsum()

    # For-loop to determine the position of ticks in x-axis for each bar
    m_o["xpos"] = ""

    for index in m_o.index:
        i = m_o.index.get_loc(index)  # get index number based on index name

        if index == "Solar":  # First index
            m_o.loc[index, "xpos"] = m_o.loc[index, 'available_cap']

        else:
            # Sum of cumulative capacity in the row above and the half of available capacity in
            m_o.loc[index, "xpos"] = m_o.loc[index, 'available_cap'] / 2 + m_o.iloc[
                i - 1, m_o.columns.get_loc('cumulative_cap')]
            #m_o.loc[index, "xpos"] = m_o.loc[index, 'available_cap'] / 2 + m_o.iloc[
            #    i - 1,4]  # aufpassen welche Spalte/Zeile hier genommen wird
    print(m_o)

    # Function to determine the cut_off_power_plant that sets the market clearing price
    def cut_off(demand):

        for index in m_o.index:
            if m_o.loc[index, 'cumulative_cap'] < demand:
                pass

            else:
                cut_off_power_plant = index
                print("Power plant that sets the electricity price is: ", cut_off_power_plant)
                break

        return cut_off_power_plant

    def merit_order_curve(demand):
        plt.figure(figsize=(15, 5))
        plt.rcParams["font.size"] = 16

        colors = m_o.colors.values
        xpos = m_o['xpos'].values.tolist()
        #xpos = [0] + m_o['xpos'].values.tolist()[:-1]
        y = m_o['marginal_cost'].values.tolist()
        # width of each bar
        w = m_o['available_cap'].values.tolist()
        cut_off_power_plant = cut_off(demand)

        # Calculate the gap size
        gap_size = xpos[1] - 0

        # Adjust the width of the first bar
        w[0] += gap_size

        # Create legend handles and labels
        handles = [plt.bar([0], [0], color=color) for color in colors]
        labels = m_o.index.tolist()

        plt.style.use('bmh')
        plt.bar(xpos,height=y, width=w, fill=True,color=colors)

        plt.xlim(0,m_o['available_cap'].sum())
        plt.ylim(0, m_o['marginal_cost'].max()+5)

        plt.hlines(y=m_o.loc[cut_off_power_plant, 'marginal_cost'],
                   xmin=0,
                   xmax=demand,
                   color="red",
                   linestyle="dashed")

        plt.vlines(x=demand,
                   ymin=0,
                   ymax=m_o.loc[cut_off_power_plant, 'marginal_cost'],
                   color="red",
                   linestyle="dashed",
                   label="Demand")

        plt.legend(handles, labels, loc='upper left', ncol=3)

        plt.text(x=demand - m_o.loc[cut_off_power_plant, 'available_cap'] / 2-3000,
                 y=m_o.loc[cut_off_power_plant, "marginal_cost"] + 10,
                 s=f"Electricity price: \n    {round(m_o.loc[cut_off_power_plant, 'marginal_cost'], 2)} €/MWh")

        plt.xlabel("Power plant capacity (GW)")
        plt.ylabel("Marginal Cost (€/MWh)")
        plt.show()
        plt.tight_layout()
        plt.savefig('results/merit_order_plot.png')

    merit_order_curve(demand=demand)


merit_order(dispatch,cf, tech_data_noCO2price, 83115)

