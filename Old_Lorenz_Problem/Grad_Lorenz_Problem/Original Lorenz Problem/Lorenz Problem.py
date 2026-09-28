# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 20:13:20 2023

@author: holli
"""

# import libraries
from Lorenz_Class import Lorenz

# constants
S = SIGMA = 10
R = RHO = 28
B = BETA = 8/3
s0 = S + 100 # initial estimate

# taken from the https://github.com/unis-ing/lorenz-parameter-learning/tree/main repo
mu1 = 500
mu2 = 0
mu3 = 0

# algorithm parameters
t0 = 0
tf = 100 # final time
dt = 0.0001 # forward Euler timestep
dt_obs = 0.1 # how often to observe solution
dt_par = 0.1 # how often to update param
p_tol = 0.0001 # tolerance for parameter switching
       
# Lorenz(SIGMA, RHO, BETA) 
lorenz_1 = Lorenz(SIGMA, RHO, BETA)
lorenz_1.get_ic_list(60, 59.9, 10)

# lorenz_1.(initial_estimate, SIG)
lorenz_1.solve_lorenz(s0, 60, 59.9, 10, 1.) # would the parameters of the inital SIGMA, RHO and BETA be different that what is in the solve lorenz equation

lorenz_2 = Lorenz(SIGMA, RHO, BETA)
lorenz_2.get_ic_list(60, 59.9, 10)
lorenz_2.solve_lorenz(s0, 60, 59.9, 10, .8)

lorenz_1.plot_absolute_errors()
lorenz_1.plot_absolute_errors_figure_1()
lorenz_2.plot_absolute_errors_figure_5()

lorenz_1.plot_Lorenz_3d(60, 59.9, 10)