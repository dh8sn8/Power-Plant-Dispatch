import pyomo.environ as pyo
import pandas as pd
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory

demand = pd.read_csv('load.csv',header=None)
demand.rename(columns={0:"timestep", 1:"load_MW"}, inplace=True)
demand