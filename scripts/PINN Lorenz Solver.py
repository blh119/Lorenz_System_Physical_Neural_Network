#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:15:32 2026

@author: brianholliday
"""

import pandas as pd
import numpy as np

# make linear space function

def linear_space(start, end, steps):
    
    step_length = (end - start) / steps

    even_spaced_nums = [start]
    
    # keeps track if we have reached 
    tracker = start
    
    # add values to the even_spaced_nums list
    while tracker < end:
        tracker = tracker + step_length
        even_spaced_nums.append(tracker)
        
    return np.array(even_spaced_nums) # return whole list as numpy array


class Lorenz:
    
    def __init__(self, sigma = 10., beta = 8/3, rho = 28., dt = [.001, .0001, .00001, .000001]):
        
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.dt_list = dt
        self.lorenz_list = []
        self.sample_list = []
        self.lorenz_df = pd.DataFrame(data=None)
        
    def __str__(self):
        
        output_string = "Sigma: " + str(self.sigma) + "\nRho: " + str(self.rho) + "\nBeta: " + str(self.beta)
        return output_string
    
    def get_ic_list(self, x0 = 1., y0 = 1., z0 = 1., dt=1/10000, T=1.0): # simulating 1 seconds
        
        N = int(T / dt)
        x, y, z = x0, y0, z0
        
        self.x_list = [x]
        self.y_list = [y]
        self.z_list = [z]

        for _ in range(N):
     
            dx = self.sigma * (y - x)
            dy = self.rho * x - y - x * z
            dz = x * y - self.beta * z

            x += dt * dx
            y += dt * dy
            z += dt * dz
        
            self.x_list.append(x)
            self.y_list.append(y)
            self.z_list.append(z)
            
    def get_lorenz_system(self):
    
        for dt in self.dt_list:
            
            self.get_ic_list(dt = dt)
            
            self.lorenz_list.append({"sigma" : [self.sigma for i in range(len(self.x_list))],
                                     "beta" : [self.beta for i in range(len(self.x_list))],
                                     "rho" : [self.rho for i in range(len(self.x_list))],
                                     "x_pos" : self.x_list,
                                     "y_pos" : self.y_list,
                                     "z_pos" : self.z_list,
                                     "dt" : [dt for i in range(len(self.x_list))]})
            
    def get_dataframe(self):
        
        for current_list in self.lorenz_list:
            
            self.lorenz_df = pd.concat([self.lorenz_df, pd.DataFrame(current_list)])
            
    def lorenz_pipeline(self):
        
        self.get_lorenz_system()
        self.get_dataframe()
        
        
class Lorenz_PINN_solver(Lorenz):
    
    def sample_system(self, dt = .0001, frac_sample = .10):
        
        current_df = (self.lorenz_df.loc[self.lorenz_df["dt"] == dt, ].reset_index(names="index"))
        current_df["t"] = current_df["index"] * current_df["dt"]
        
        self.sample_list.append(current_df.sample(frac = frac_sample, replace = False, random_state = 99))
        
    def get_samples(self):
        
        for current_dt in self.dt_list:
            
            self.sample_system(current_dt) 
            
    def sample_pipeline(self):
        
        self.lorenz_pipeline()
        self.get_samples()
        
lorenz = Lorenz_PINN_solver()
lorenz.sample_pipeline()
    

    
        
        

