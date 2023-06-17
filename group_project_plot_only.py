#!/usr/bin/env python
# coding: utf-8

# # Energy Systems Group Assignment: RPower Plant Dispatch
# #### Tobias Benedikt Blenk, Darlene Sonal D'Mello, Adnan Moiz, Diep Hang Stefanie Ngoc Nguyen ####

# Import necessary packages

# In[1]:


import pyomo.environ as pyo
import pandas as pd
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
import pandas as pd
import matplotlib.pyplot  as plt


# Import electricity demand, capacity factors, time step duration and power plant data 

# In[2]:


demand = pd.read_csv('load.csv',header=None)
demand.rename(columns={0:"timestep", 1:"load_MW"}, inplace=True)
demand


# In[3]:


cf = pd.read_csv('capacity_factors.csv')
cf.rename(columns={cf.columns[0]:"tech"}, inplace=True)
cf[["t1", "t2", "t3", "t4", "t5","t6", "t7", "t8", "t9", "t10"]] = cf[["t1", "t2", "t3", "t4", "t5","t6", "t7", "t8", "t9", "t10"]].apply(pd.to_numeric)
cf


# In[4]:


duration = pd.read_csv('duration.csv',header=None)
duration.rename(columns={0:"timestep", 1:"length"}, inplace=True)
duration


# In[5]:


tech_data = pd.read_csv('tech_data.csv',header=None)
tech_data.rename(columns={0:"tech", 1:"cap_MW",2:"eta",3:"fuel_p",4:"c_var_other",5:"emf"}, inplace=True)
tech_data.drop([0,1],inplace=True)
tech_data[["cap_MW", "eta", "fuel_p", "c_var_other", "emf"]] = tech_data[["cap_MW", "eta", "fuel_p", "c_var_other", "emf"]].apply(pd.to_numeric)
tech_data.reset_index(drop=True, inplace=True)
tech_data


# Create a generic model `m` for all timesteps

# In[6]:


# Create a concrete Pyomo model
m = pyo.ConcreteModel()
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) #specify dual variables (shadow prices)


# Create a set $S$ for the technologies:

# In[7]:


m.S = pyo.Set(initialize=tech_data['tech'])


# In[8]:


m.S.pprint()


# Define the decision variable generators $g_{s,t}$

# In[9]:


# Define the decision variable for each generator
m.generators = Var(m.S, domain=pyo.NonNegativeReals)


# In[10]:


m.generators.pprint()


# Define the marginal cost for each generator

# In[11]:


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


# ### A. No CO2 price

# Calculate the marginal cost $o_s$ with no CO2 price
# 
# \begin{equation}
#     o_s = \frac{\text{fuel price}}{\eta} + \text{other variable costs} \\
# \end{equation}
# 
# 

# In[12]:


tech_data_noCO2price = tech_data.copy()
tech_data_noCO2price['marginal_cost'] = round(tech_data_noCO2price['fuel_p']/tech_data_noCO2price['eta']+tech_data_noCO2price['c_var_other'],2)
tech_data_noCO2price


# Clone `m` for no CO2 price models `m_0`

# In[13]:


m_0 = m.clone()


# Define the objective function: we want to minimise the operating costs of the electricity system while meeting the demand for each timestep
# 
# \begin{equation}
#     \min_{g_{s,t}} \sum_s o_s g_{s,t} \quad \forall t \\
# \end{equation}
# 
#   such that
#   
#   \begin{align}
#     g_{s,t} &\leq G_{s,t} \\
#     g_{s,t} &\geq 0 \\
#     \sum_s g_{s,t} &= d_t
#   \end{align}
# 
# where $G_{s,t}$ is the available capacity at timestep $t$, $g_{s,t}$ is the generation of generator $s$ at timestep $t$, $o_s$ is the marginal cost for generator $s$, and $d_t$ is the demand at timestep $t$

# In[14]:


# Define the objective function
m_0.cost = Objective(expr=sum(marginal_price(s,0)*m_0.generators[s] for s in m_0.S),
                      sense=pyo.minimize)


# In[15]:


m_0.cost.pprint()


# Create dictionary of models for all timesteps

# In[16]:


models = {}
for i in range(len(duration)):
    models[duration['timestep'][i]] = m_0.clone() #clone original model for each of the timesteps

#create a list of timesteps
timesteps = list(models.keys())


# Each timestep now has a model

# In[17]:


models


# Add demand and generator limit constraints to all timesteps
# 
#   \begin{align}
#     g_{s,t} &\leq G_{s,t} \\
#     g_{s,t} &\geq 0 \\
#     \sum_s g_{s,t} &= d_t
#   \end{align}

# In[18]:


for i in timesteps:
    model = models[i]
    #add generator limit: capacity*cf at a given timestep
    
    @model.Constraint(model.S)
    def generator_limit(model, s):
        return model.generators[s] <= cf[cf['tech'] == s][i].values[0]*tech_data[tech_data['tech'] == s].cap_MW.values[0] 
    models[i] = model
    
    #add demand constraint: sum of generation must equal demand    
    models[i].demand_constraint = Constraint(expr=sum(models[i].generators[s] for s in models[i].S) == demand[demand.timestep == i].load_MW.values[0]) 


# Solve the models at all timesteps

# In[19]:


for i in timesteps:
    SolverFactory('cbc').solve(models[i]).write()


# Extract optimum dispatch in MW for each timestep from the results

# In[20]:


dispatch = pd.Series(models[timesteps[0]].generators.get_values(),name=timesteps[0]).to_frame()
for i in timesteps[1:]:
    d = pd.Series(models[i].generators.get_values(),name=i).to_frame()
    dispatch = dispatch.join(d)
dispatch #MW


# Calculate dispatch expenditure [€] for each timestep: electricity price [€/MWh] * load [MW] * length of timestep [h]

# In[21]:


dispatch_costs = duration.copy()
dispatch_costs['load_MW'] = demand['load_MW']
dispatch_costs['electricity_price'] = [models[t].dual[models[t].demand_constraint] for t in timesteps]
dispatch_costs['expenditure'] = [models[t].cost.expr() for t in timesteps]*dispatch_costs['length']
dispatch_costs.to_csv('results/dispatch_costs.csv')
dispatch_costs


# Calculate total annual expenditure and average electricity price

# In[22]:


print("Total annual expenditure [billion €]:", round((dispatch_costs.expenditure).sum()/1e9,3))
print("Average electricity price [€/MWh]:", round(((dispatch_costs.electricity_price*dispatch_costs.length).sum()/dispatch_costs.length.sum()),2))


# Calculate dispatch emissions $E$ per timestep in tCO2: 
# \begin{equation}
#     E = EMF/\eta * g_s * t 
# \end{equation}

# In[23]:


dispatch_emissions = dispatch.copy()


# In[24]:


for i in timesteps:
    dispatch_emissions[i] = (tech_data.set_index(dispatch.index)['emf'].values*dispatch[i]/tech_data.set_index(dispatch.index)['eta'])*duration[duration['timestep'] == i].length.values[0]


dispatch_emissions.to_csv('results/dispatch_emissions.csv')
dispatch_emissions  #total emissions per timestep, tCO2


# In[25]:


print('Total emissions [million tCO2]:', round(dispatch_emissions.sum().sum()/1e6,2))


# Calculate dual prices/shadow prices for each generator for all the timesteps: (how much more expensive the objective function would be if you increased the capacity by one unit)

# In[26]:


dual_prices = pd.Series({s: models[timesteps[0]].dual[models[timesteps[0]].generator_limit[s]] for s in models[timesteps[0]].S},name=timesteps[0]).to_frame()
for i in timesteps[1:]:
    dp = pd.Series({s: models[i].dual[models[i].generator_limit[s]] for s in models[i].S},name=i).to_frame()
    dual_prices = dual_prices.join(dp)

dual_prices.to_csv('results/dual_prices.csv')
dual_prices


# Calculate annual contribution margin for the individual technologies, where of each technology:
# 
# $$\text{Contribution Margin} = \text{Revenue} - \text{Operational Cost}$$

# In[27]:


operational_costs = dispatch.multiply(tech_data_noCO2price.set_index('tech')['marginal_cost'],axis=0)
operational_costs


# In[28]:


revenue = dispatch.copy()
for i in timesteps:
    revenue[i] = revenue[i]*dispatch_costs[dispatch_costs["timestep"] == i]['electricity_price'].values[0]
revenue


# In[29]:


profit = revenue - operational_costs
profit


# In[30]:


profit.sum(axis=1).to_frame(name='Contribution Margin [€]').to_csv('results/profit.csv')
profit.sum(axis=1).to_frame(name='Contribution Margin [€]')


# In[31]:


print("Total Annual Profit [million €]", round(profit.sum(axis=1).sum()/1e6,3))


# ### B. CO2 price of 120 Euros/tCO2

# Calculate the marginal cost $o_s$ with CO2 price = 120 €/tCO2
# 
# \begin{equation}
#     o_s = \frac{\text{fuel price}}{\eta} + 120*\frac{\text{EMF}}{\eta} + \text{other variable costs}
# \end{equation}
# 

# In[32]:


tech_data_CO2price = tech_data.copy()
tech_data_CO2price['marginal_cost'] = round(tech_data_noCO2price['fuel_p']/tech_data_noCO2price['eta']+tech_data_noCO2price['c_var_other'] + 120*tech_data_noCO2price['emf']/tech_data_noCO2price['eta'],2)
tech_data_CO2price


# Clone generic `m` to create model `m_co2`

# In[33]:


m_co2 = m.clone()


# Define the objective function: we want to minimise the operating costs of the electricity system while meeting the demand for each timestep
# 
# \begin{equation}
#     \min_{g_{s,t}} \sum_s o_s g_{s,t} \quad \forall t
#   \end{equation}
# 
#   such that
#   
#   \begin{align}
#     g_{s,t} &\leq G_{s,t} \\
#     g_{s,t} &\geq 0 \\
#     \sum_s g_{s,t} &= d_t
#   \end{align}

# In[34]:


# Define the objective function, include CO2 price of 120 euros/tCO2
co2_price = 120
m_co2.cost = Objective(expr=sum(marginal_price(s,co2_price)*m_co2.generators[s] for s in m_co2.S),
                      sense=pyo.minimize)


# In[35]:


m_co2.cost.pprint()


# Create dictionary of models for all timesteps

# In[36]:


models_co2 = {}
for i in range(len(duration)):
    models_co2[duration['timestep'][i]] = m_co2.clone() #clone original model for each of the timesteps

#create a list of timesteps
timesteps = list(models_co2.keys())


# Each timestep now has a model

# In[37]:


models_co2


# Add demand and generator limit constraints to all timesteps
# 
#   \begin{align}
#     g_{s,t} &\leq G_{s,t} \\
#     g_{s,t} &\geq 0 \\
#     \sum_s g_{s,t} &= d_t
#   \end{align}

# In[38]:


for i in timesteps:
    model = models_co2[i]
    #add generator limit: capacity*cf at a given timestep
    
    @model.Constraint(model.S)
    def generator_limit(model, s):
        return model.generators[s] <= cf[cf['tech'] == s][i].values[0]*tech_data[tech_data['tech'] == s].cap_MW.values[0] 
    models_co2[i] = model
    
    #add demand constraint: sum of generation must equal demand    
    models_co2[i].demand_constraint = Constraint(expr=sum(models_co2[i].generators[s] for s in models_co2[i].S) == demand[demand.timestep == i].load_MW.values[0]) 


# Solve the models at all timesteps

# In[39]:


for i in timesteps:
    SolverFactory('cbc').solve(models_co2[i]).write()


# Extract optimum dispatch in MW for each timestep from the results

# In[40]:


dispatch_co2 = pd.Series(models_co2[timesteps[0]].generators.get_values(),name=timesteps[0]).to_frame()
for i in timesteps[1:]:
    d = pd.Series(models_co2[i].generators.get_values(),name=i).to_frame()
    dispatch_co2 = dispatch_co2.join(d)

dispatch_co2.to_csv('results/dispatch_co2.csv')
dispatch_co2 #MW


# Calculate dispatch expenditure [€] for each timestep: electricity price [€/MWh] * load [MW] * length of timestep [h]

# In[41]:


dispatch_costs_co2 = duration.copy()
dispatch_costs_co2['load_MW'] = demand['load_MW']
dispatch_costs_co2['electricity_price'] = [models_co2[t].dual[models_co2[t].demand_constraint] for t in timesteps]
dispatch_costs_co2['expenditure'] = [models_co2[t].cost.expr() for t in timesteps]*dispatch_costs_co2['length']
dispatch_costs_co2.to_csv('results/dispatch_costs_co2.csv')
dispatch_costs_co2


# Calculate total annual expenditure and average electricity price

# In[42]:


print("Total annual expenditure with 120€/tCO2 price [billion €]:", round((dispatch_costs_co2.expenditure).sum()/1e9,3))
print("Average electricity price with 120€/tCO2 price [€/MWh]:", round(((dispatch_costs_co2.electricity_price*dispatch_costs.length).sum()/dispatch_costs.length.sum()),2))


# Calculate dispatch emissions per timestep in tCO2: (EMF [tCO2/MWh_th]/eta [MW_el/MW_th] * dispatch [MW])) * length of timestep [h]

# In[43]:


dispatch_emissions_co2 = dispatch_co2.copy()


# In[44]:


for i in timesteps:
    dispatch_emissions_co2[i] = (tech_data.set_index(dispatch_co2.index)['emf'].values*dispatch_co2[i]/tech_data.set_index(dispatch_co2.index)['eta'])*duration[duration['timestep'] == i].length.values[0]

dispatch_emissions_co2.to_csv('results/dispatch_emissions_co2.csv')
dispatch_emissions_co2  #total emissions per timestep, tCO2


# In[45]:


print('Total emissions with 120€/tCO2 price [million tCO2]:', round(dispatch_emissions_co2.sum().sum()/1e6,2))


# Calculate annual contribution margin for the individual technologies

# In[46]:


operational_costs_co2 = dispatch_co2.multiply(tech_data_CO2price.set_index('tech')['marginal_cost'],axis=0)
operational_costs_co2


# In[47]:


revenue_co2 = dispatch_co2.copy()
for i in timesteps:
    revenue_co2[i] = revenue_co2[i]*dispatch_costs_co2[dispatch_costs_co2["timestep"] == i]['electricity_price'].values[0]
revenue_co2


# In[48]:


profit_co2 = revenue_co2 - operational_costs_co2
profit_co2


# In[49]:


profit_co2.sum(axis=1).to_frame(name='Contribution Margin [€]').to_csv('results/profit_co2.csv')
(round(profit_co2.sum(axis=1)/1e6,2)).to_frame(name='Contribution Margin [mil €]')


# In[50]:


print("Total Annual Profit with 120 €/tCO2 CO2 price [million €]:", round(profit_co2.sum(axis=1).sum()/1e6,3))


# Calculate dual prices/shadow prices for each generator for all the timesteps: (how much more expensive the objective function would be if you increased the capacity by one unit)

# In[51]:


dual_prices_co2 = pd.Series({s: models_co2[timesteps[0]].dual[models_co2[timesteps[0]].generator_limit[s]] for s in models_co2[timesteps[0]].S},name=timesteps[0]).to_frame()
for i in timesteps[1:]:
    dp = pd.Series({s: models_co2[i].dual[models_co2[i].generator_limit[s]] for s in models_co2[i].S},name=i).to_frame()
    dual_prices_co2 = dual_prices_co2.join(dp)

dual_prices_co2.to_csv('results/dual_prices_co2.csv')
dual_prices_co2


# ### Comparison

# In[52]:


contr_margin = (round(profit.sum(axis=1)/1e6,2)).to_frame(name='Contribution Margin [mil €]')
contr_margin['Contribution Margin (CO2 price 120 €/tCO2) [mil €]'] = (round(profit_co2.sum(axis=1)/1e6,2)).to_frame()[0]
contr_margin.to_csv('results/contribution_margin.csv')
contr_margin


# Marginal Prices - the marginal prices for carbon generating generators increase with the CO2 price of 120 €/tCO2

# In[53]:


marginal_price = (tech_data_noCO2price[['tech', 'marginal_cost']]).set_index('tech').rename(columns={'marginal_cost': 'Marginal Price'})
marginal_price['Marginal Price with CO2 price'] = tech_data_CO2price[['tech', 'marginal_cost']].set_index('tech')
marginal_price


# In[54]:


plt.style.use('bmh')
ax = marginal_price.plot.bar(figsize=(9,4),ylabel = 'Price [€/MWh]',xlabel='',rot=0)
ax.figure.savefig('marginal_price.pdf',bbox_inches='tight')


# Plot for merit order

# In[55]:


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
        plt.ylim(0, m_o['marginal_cost'].max()+20)

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
        #plt.savefig('figure.pdf', format='pdf')

    merit_order_curve(demand=demand)


# In[56]:


#merit order plot with CO2 price for time step 1
#merit_order(dispatch_co2,cf,tech_data_CO2price, 83115)


# In[57]:


#merit order plot without CO2 for time step 1
merit_order(dispatch,cf, tech_data_noCO2price, 83115)

