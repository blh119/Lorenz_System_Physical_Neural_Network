# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 08:24:34 2023

@author: holli
"""

 # -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:32:51 2023

@author: holli
"""

import numpy as np
import matplotlib.pyplot as plt

sigma = 16.
beta = 4.
rho = 45.92

mu0 = 1800.
mu2 = 1800. 

dt = 0.0009

def get_ic(x0, y0, z0, dt): 
     # = 2/mu
    N = int(5/dt)+1
    
    x, y, z = xp, yp, zp = [x0, y0, z0]
        
    for i in range(N):     
        
        x += dt * sigma*(yp - xp)
        y += dt * (rho*xp - yp - xp*zp)
        z += dt * (xp*yp - beta*zp)
        xp, yp, zp = x, y, z
        
    return x, y, z


def get_ic_list(x0, y0, z0, dt): 
    
    N = int(10/dt)+1
    x, y, z = xp, yp, zp = [x0, y0, z0]

    x_list = [x]
    y_list = [y]
    z_list = [z]
        
    for i in range(N):     
        
        x += dt * sigma*(yp - xp)
        y += dt * (rho*xp - yp - xp*zp)
        z += dt * (xp*yp - beta*zp)
        xp, yp, zp = x, y, z

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    return x_list, y_list, z_list

def solve_lorenz_split_1_first(time_step, init_x, init_y, init_z):
     
    sol = np.empty((time_step, 3))   
    ndg = np.empty((time_step, 3))
    sol_g = np.empty(time_step)
    ndg_g = np.empty(time_step)
        
    sol[0] = np.array([init_x,
                       init_y, 
                       init_z])
        
    ndg[0] = np.array([init_x, 
                       0., # nudging equation for y starts at zero
                       init_z])
        
    sol_g[0] = -init_x*init_z
        
    ndg_g[0] = 0.

    
    for i in range(1, time_step):
        
        u = sol[i-1]
        v = ndg[i-1]
        
        ndg_g_ = 0.
        
        u_ = np.array([sigma*(u[1]-u[0]), rho*u[0]-u[1]-u[0]*u[2], u[0]*u[1]-beta*u[2]])
        # unknown solution
        v_ = np.array([sigma*(v[1]-v[0])-mu0*(v[0]-u[0]), rho*v[0]-v[1]+ndg_g_, v[0]*v[1]-beta*v[2]-mu2*(v[2]-u[2])])
        
        sol[i] = u + (dt*u_)
        ndg[i] = v + (dt*v_)
        sol_g[i] = -u[0]*u[2]
        ndg_g[i] = ndg_g_
        
        print(sol[i], ndg[i], sol_g[i], ndg_g[i])
        
    return [np.array(sol), np.array(ndg), np.array(sol_g), np.array(ndg_g)]


def solve_lorenz_split_1(time_step, init_x, init_y, init_z, init_ndg_y, init_sol_g, ndg_list):
    
    sol = np.empty((time_step, 3))   
    ndg = np.empty((time_step, 3))
    sol_g = np.empty(time_step)
    ndg_g = np.empty(time_step)
    
    sol[0] = np.array([init_x, 
                       init_y, 
                       init_z])
    ndg[0] = np.array([init_x, 
                       init_ndg_y, 
                       init_z])
    
    sol_g[0] = init_sol_g
    ndg_g[0] = -ndg_list[0][0]*ndg_list[0][2]
    
    for i in range(1, time_step):
        
        u = sol[i-1]
        v = ndg[i-1]
        
        u_ = np.array([sigma*(u[1]-u[0]), rho*u[0]-u[1]-u[0]*u[2], u[0]*u[1]-beta*u[2]])
        # unknown solution
        v_ = np.array([sigma*(v[1]-v[0])-mu0*(v[0]-u[0]), rho*v[0]-v[1]+ndg_g[i-1], v[0]*v[1]-beta*v[2]-mu2*(v[2]-u[2])])
        
        sol[i] = u + (dt*u_)
        ndg[i] = v + (dt*v_)
        sol_g[i] = -u[0]*u[2]
        ndg_g[i] = -ndg_list[i][0]*ndg_list[i][2]
        
        print(sol[i], ndg[i], sol_g[i], ndg_g[i])
        
    return [np.array(sol), np.array(ndg), np.array(sol_g), np.array(ndg_g)]
        
def nudged_system(tf, step):
    
    time_steps = np.arange(0, tf + 1, step)[::-1]
    time_steps = time_steps[time_steps != 0]
    
    init_values_index = np.arange(0, tf, step)
    init_values_index = init_values_index[init_values_index != 0]-1
    
    sol_full_set = []
    ndg_full_set = []
    sol_g_full_set = []
    ndg_g_full_set = []
    
    # initial run
    counter = 0 # this helps keep track of our correct initial position
    
    for time_step in time_steps:
        
        if counter == 0:
            
            init_x, init_y, init_z = get_ic(60., 59.9, 10., dt)
            
            current_sol, current_ndg, current_sol_g, current_ndg_g = solve_lorenz_split_1_first(time_step, init_x, init_y, init_z)
            
            sol_full_set.append(current_sol)
            ndg_full_set.append(current_ndg)
            sol_g_full_set.append(current_sol_g)
            ndg_g_full_set.append(current_ndg_g)
            
            counter = counter + 1
            
        else:
            
            init_x = sol_full_set[counter-1][step][0]
            init_y = sol_full_set[counter-1][step][1]
            init_z = sol_full_set[counter-1][step][2]
            
            init_ndg_y = ndg_full_set[counter-1][step][1]
            init_sol_g = -sol_full_set[counter-1][step-1, 0]*sol_full_set[counter-1][step-1, 2]
            ndg_list = ndg_full_set[counter-1][step-1:]
            
            current_sol, current_ndg, current_sol_g, current_ndg_g = solve_lorenz_split_1(time_step, init_x, init_y, init_z, init_ndg_y, init_sol_g, ndg_list)
            
            sol_full_set.append(current_sol)
            ndg_full_set.append(current_ndg)
            sol_g_full_set.append(current_sol_g)
            ndg_g_full_set.append(current_ndg_g)
            
            counter = counter + 1
            
    return sol_full_set, ndg_full_set, sol_g_full_set, ndg_g_full_set

def get_endpoints(sol, ndg, sol_g, ndg_g, step):
    
    sol_endpoints = []
    ndg_endpoints = []
    sol_g_endpoints = []
    ndg_g_endpoints = []
    
    for i in range(len(sol)):
        sol_endpoints.append(sol[i][step-1])
        
    for i in range(len(ndg)):
        ndg_endpoints.append(ndg[i][step-1])
        
    for i in range(len(sol_g)):
        sol_g_endpoints.append(sol_g[i][step-1])
        
    for i in range(len(ndg_g)):
        ndg_g_endpoints.append(ndg_g[i][step-1])
        
    return np.array(sol_endpoints), np.array(ndg_endpoints), np.array(sol_g_endpoints), np.array(ndg_g_endpoints)

def plot_absolute_errors(sol_endpoints, ndg_endpoints, sol_g_endpoints, ndg_g_endpoints):
    
    abs_u = abs(sol_endpoints[:,0]-ndg_endpoints[:,0])
    abs_v = abs(sol_endpoints[:,1]-ndg_endpoints[:,1])
    abs_w = abs(sol_endpoints[:,2]-ndg_endpoints[:,2])
    abs_g = abs(sol_g_endpoints-ndg_g_endpoints)
    
    t = np.arange(0, len(abs_g))
    
    plt.figure(figsize=(8,3))
    plt.plot(t, abs_u, lw=1, label=r'$|u|$') 
    plt.plot(t, abs_v, lw=1, label=r'$|v|$')
    plt.plot(t, abs_w, lw=1, label=r'$|w|$')
    plt.plot(t, abs_g, lw=1, label=r'$|g|$')
    
    #plt.plot(self.t, self.delta_S, 'k', lw=1, label=r'$|\tilde{\sigma}-\sigma|$')
    plt.title('Evolution of absolute errors'); plt.xlabel('t = 10000 td = 10 mu = 1000 dt = 0.001')
    plt.legend(); plt.grid(); plt.tight_layout(); plt.yscale("log")
    plt.show()
       
sol, ndg, sol_g, ndg_g = nudged_system(5000, 10)
sol_endpoints, ndg_endpoints, sol_g_endpoints, ndg_g_endpoints = get_endpoints(sol, ndg, sol_g, ndg_g, 10)
plot_absolute_errors(sol_endpoints, ndg_endpoints, sol_g_endpoints, ndg_g_endpoints)
  

