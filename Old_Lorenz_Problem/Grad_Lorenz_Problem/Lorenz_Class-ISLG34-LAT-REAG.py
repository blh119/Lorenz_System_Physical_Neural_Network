# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:32:51 2023

@author: holli
"""

import numpy as np
import matplotlib.pyplot as plt

dt_obs = 0.1 # how often to observe solution
dt_par = 0.1 # how often to update param

class Lorenz:
    
    def __init__(self, SIGMA, RHO, BETA):
        
        self.sigma = SIGMA
        self.beta = BETA
        self.rho = RHO
    
    def __str__(self):
        
        output_string = "Sigma: " + str(self.sigma) + "\nRho: " + str(self.rho) + "\nBeta: " + str(self.beta)
        return output_string
    
    def get_ic(self, x0, y0, z0): 
        
        dt = 0.0001 
        N = int(5/dt)+1
        
        x, y, z = xp, yp, zp = [x0, y0, z0]
        
        for i in range(N):     
            x += dt * self.sigma*(yp - xp)
            y += dt * (self.rho*xp - yp - xp*zp)
            z += dt * (xp*yp - self.beta*zp)
            xp, yp, zp = x, y, z

        self.last_x, self.last_y, self.last_z = x, y, z
        
    
    def get_ic_list(self, x0, y0, z0):

        dt = 0.0001 
        N = int(5/dt)+1
        
        x, y, z = xp, yp, zp = [x0, y0, z0]

        x_list = [x]
        y_list = [y]
        z_list = [z]
        
        for i in range(N):     
            x += dt * self.sigma*(yp - xp)
            y += dt * (self.rho*xp - yp - xp*zp)
            z += dt * (xp*yp - self.beta*zp)
            xp, yp, zp = x, y, z

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

        self.x_list, self.y_list, self.z_list = x_list, y_list, z_list
        
    def solve_lorenz_split_1(self, t0, tf, dt, mu1, mu2):
        
        t = np.arange(t0, tf, dt)
        dt_obs_int = int(dt_obs / dt)
        #dt_par_int = int(dt_par / dt)
        
        
        N = t.size
        sol = np.empty((N, 3))
        ndg = np.empty((N, 3))
        g = np.empty((N, 3))
        g_act = np.empty((N, 3))
        
        self.get_ic(60, 59.9, 10) # get initial conditions for the lorenz system
        
        sol[0] = [self.last_x, self.last_y, self.last_z]
        g[0] = [0., 0., 0.] # initialize parameter estimate
        ndg[0] = np.array([self.sigma*(self.last_y - self.last_x), self.rho*self.last_x - self.last_y, -self.beta*self.last_y]) + g[0] - np.array([mu1*((self.sigma*(self.last_y - self.last_x)) - sol[0][0]), mu2*((self.rho*self.last_x - self.last_y) - sol[0][1]), 0])
        
        for i in range(1, N):
            
            u = sol[i-1]
            v = ndg[i-1]
            g_ = g[i-1]
            g_act_ = g_act[i-1]
            
            
            if i % dt_obs_int == 0: # observe data
                #v = [self.sigma*(v[1] - v[0]), self.rho*v[0] - v[1], -self.beta*v[1]] + g_ + np.array([mu1*(v[0] - u[0]), mu2*(v[1] - u[1]), 0])
                # new position vector
                new_pos = np.array([u[0], u[1], 0]) + np.array([0, 0, -self.beta*v[2] + u[0]*u[1]])
                g_ = new_pos - np.array([self.sigma*(v[1] - v[0]), self.rho*v[0] - v[1], -self.beta*v[2]])# new g

            # known solution 
            u_ = np.array([self.sigma*(u[1] - u[0]), self.rho*u[0] - u[1] - u[0]*u[2], -self.beta*u[2] + u[0]*u[1]])
            
            f_ = np.array([self.sigma*(v[1] - v[0]), self.rho*v[0] - v[1], -self.beta*v[2]])
            v_ = f_ + g_ - np.array([mu1*(v[0] - u[0]), mu2*(v[1] - u[1]), 0])
            
            g_act_ = np.array([self.sigma*(u[1] - u[0]), -u[0]*u[2], u[0]*u[1]])
            
            g[i] = g_ + (dt*g_)
            g_act[i] = g_act_ + (dt*g_act_)
            sol[i] = u + (dt*u_)
            ndg[i] = v + (dt*v_)
            
            
            
        self.t, self.g, self.g_act, self.sol, self.ndg = t, g, g_act, sol, ndg
        
        
        self.idx = np.arange(0, self.t.size, 1000) # plot every 1000 timesteps
        
        # these equations represent delta t
        self.abs_u = abs(self.g[self.idx,0]-self.g_act[self.idx,0])
        self.abs_v = abs(self.g[self.idx,1]-self.g_act[self.idx,1])
        self.abs_w = abs(self.g[self.idx,2]-self.g_act[self.idx,2])
        
        
    def plot_absolute_errors(self):
        
        plt.figure(figsize=(8,3))
        plt.plot(self.t[self.idx], self.abs_u, lw=1, label=r'$|u|$')
        plt.plot(self.t[self.idx], self.abs_v, lw=1, label=r'$|v|$')
        plt.plot(self.t[self.idx], self.abs_w, lw=1, label=r'$|w|$')
        plt.title('Evolution of absolute errors'); plt.xlabel('t'); plt.yscale('log')
        plt.legend(); plt.grid(); plt.tight_layout()
        plt.show()   
        
        
lorenz_1 = Lorenz(10, 28, 8/3)
lorenz_1.get_ic(60, 59.9, 10)
lorenz_1.solve_lorenz_split_1(0, 100, 0.0001, 100., 50.)
lorenz_1.plot_absolute_errors()

