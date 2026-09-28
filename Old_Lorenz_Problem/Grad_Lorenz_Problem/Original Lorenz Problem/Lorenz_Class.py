# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 20:19:56 2023

@author: holli
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams["font.family"] = "serif"; plt.rcParams["font.size"] = 11

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
        
class Lorenz:
    
    def __init__(self, sigma, rho, beta):
        
        self.S = sigma
        self.R = rho
        self.B = beta

    def __str__(self):
        
        output_string = "Sigma: " + str(self.S) + " Rho: " + str(self.R) + " Beta: " + str(self.B)
        return output_string

    def get_ic(self, x0, y0, z0): 
        
        dt = 0.0001 
        N = int(5/dt)+1
        
        x, y, z = xp, yp, zp = [x0, y0, z0]
        
        for i in range(N):     
            x += dt * self.S*(yp - xp)
            y += dt * (self.R*xp - yp - xp*zp)
            z += dt * (xp*yp - self.B*zp)
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
            x += dt * self.S*(yp - xp)
            y += dt * (self.R*xp - yp - xp*zp)
            z += dt * (xp*yp - self.B*zp)
            xp, yp, zp = x, y, z

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

        self.x_list, self.y_list, self.z_list = x_list, y_list, z_list

    def solve_lorenz(self, s0, x0, y0, z0, sig_par): # sigma parameter
        
        t = np.arange(t0, tf, dt)
        dt_obs_int = int(dt_obs / dt)
        dt_par_int = int(dt_par / dt)
        
        N = t.size
        sol = np.empty((N, 3))
        ndg = np.empty((N, 3))
        par = np.empty(N)
        
        self.get_ic(x0, y0, z0) # get initial conditions for the lorenz system
        
        sol[0] = [self.last_x, self.last_y, self.last_z]
        ndg[0] = [0.1, 0.1, 0.1]
        par[0] = s0
        
        s = s0 # initialize parameter estimate
        for i in range(1, N):
            
            u = sol[i-1]
            v = ndg[i-1]
            
            if i % dt_par_int == 0: # update
                if abs(v[1] - u[0]) > p_tol:
                    new_s = s * (v[1] - v[0]) / (v[1] - u[0])
                    if new_s > 0: 
                        s = new_s
                        
            if i % dt_obs_int == 0: # observe data
                v[0] = u[0] # direct insertion
                
            u_ = np.array([self.S*(u[1] - u[0]), self.R*u[0] - u[1] - u[0]*u[2], u[0]*u[1] - self.B*u[2]])
            v_ = np.array([(s*sig_par)*(v[1] - v[0]), self.R*v[0] - v[1] - v[0]*v[2], v[0]*v[1] - self.B*v[2]])
            
            
            sol[i] = u + dt*u_
            ndg[i] = v + dt*v_
            par[i] = (s*sig_par)
            
            
        self.t , self.par, self.sol, self.ndg = t, par, sol, ndg
        
        
        self.idx = np.arange(0, self.t.size, 1000) # plot every 1000 timesteps
        self.abs_u = abs(self.ndg[self.idx,0]-self.sol[self.idx,0])
        self.abs_v = abs(self.ndg[self.idx,1]-self.sol[self.idx,1])
        self.abs_w = abs(self.ndg[self.idx,2]-self.sol[self.idx,2])
        self.delta_S = abs(self.par - self.S)
        
        # using equation 2.5 
        self.abs_u_err = self.delta_S[self.idx]*(self.ndg[self.idx, 1] - self.ndg[self.idx,0]) \
        + (self.S * self.abs_v) - (mu1 + self.S)*self.abs_u
        
        self.abs_v_err = (self.R - self.R)*self.ndg[self.idx, 0] \
        + self.R*self.abs_u - (self.abs_u*self.sol[self.idx, 2]) \
        - (self.ndg[self.idx, 0] * self.abs_w) - (1. + mu2)*self.abs_v
        
        self.abs_w_err = -1.*(self.B - self.B)*self.ndg[self.idx, 2] \
        + (self.abs_u*self.sol[self.idx, 1]) + (self.ndg[self.idx, 0]*self.abs_v) \
        - (mu3 + self.B)*self.abs_w
        
        self.abs_error_all = np.sqrt(self.abs_u**2  + self.abs_v**2 + self.abs_w**2)
        self.abs_error_all_vel = np.sqrt(self.abs_u_err**2  + self.abs_v_err**2 + self.abs_w_err**2)
        
        # how do I get Cor IV.2 and Cor IV.4    
        
    def plot_absolute_errors(self):
        
        plt.figure(figsize=(8,3))
        plt.plot(self.t[self.idx], self.abs_u, lw=1, label=r'$|u|$')
        plt.plot(self.t[self.idx], self.abs_v, lw=1, label=r'$|v|$')
        plt.plot(self.t[self.idx], self.abs_w, lw=1, label=r'$|w|$')
        plt.plot(self.t, self.delta_S, 'k', lw=1, label=r'$|\tilde{\sigma}-\sigma|$')
        plt.title('Evolution of absolute errors'); plt.xlabel('t'); plt.yscale('log')
        plt.legend(); plt.grid(); plt.tight_layout()
        plt.show()
        
    def plot_absolute_errors_figure_1(self):
        
        # Create a figure and subplots
        fig = plt.figure(figsize=(10,8))
        gs = GridSpec(2, 2)

        # Customize each subplot
        ax1 = fig.add_subplot(gs[0, :])  # Top plot spanning across both columns
        ax2 = fig.add_subplot(gs[1, 0])  # Bottom-left plot
        ax3 = fig.add_subplot(gs[1, 1])  # Bottom-right plot

        
        ax1.plot(self.t[self.idx], self.abs_error_all, lw=1, label=r'$||(u, v, w)||$')
        ax1.plot(self.t[self.idx], self.abs_error_all_vel, lw=1, 
                 label=r'$||(\dot{u}, \dot{v}, \dot{w})||$')
        
        ax1.plot(self.t, self.delta_S, 'k', lw=1, label=r'$|\tilde{\sigma}-\sigma|$')
        ax1.set_title("Evolution of absolute errors")
        ax1.set_xlabel("Time t")
        ax1.legend(); ax1.grid(); ax1.set_yscale("log")
        
        ax2.plot(self.t[self.idx], self.abs_error_all, lw=1, label=r'$||(u, v, w)||$')
        ax2.set_title("Position error")
        ax2.set_xlabel("Time t")
        ax2.legend(); ax2.grid(); ax2.set_yscale("log")
        
        ax3.plot(self.t[self.idx], self.abs_error_all_vel, lw=1, 
                 label=r'$||(\dot{u}, \dot{v}, \dot{w})||$',
                color = "orange")        
        ax3.set_title("Velocity error")
        ax3.set_ylabel("Time t")
        ax3.legend(); ax3.grid(); ax3.set_yscale("log")

        # Adjust the layout and spacing
        plt.tight_layout()

        # Show the figure
        plt.show()
        
    def plot_absolute_errors_figure_5(self):
        
        plt.figure(figsize = (8,3))
        plt.plot(self.t[self.idx], self.abs_error_all, lw=1, label=r'$||(u, v, w)||$')
        plt.plot(self.t, self.delta_S, 'k', lw=1, label=r'$|\Delta{\sigma}|$')
        plt.xlabel("time")
        plt.ylabel("error")
        plt.yscale("log")
        plt.legend()
        plt.title(r'$\tilde{\sigma} = 0.8\sigma, \tilde{\rho} = \rho, \tilde{\beta} = \beta$')
        
    
    def plot_Lorenz_3d(self, x0, y0, z0):
        
        # call ic_list function
        self.get_ic_list(x0, y0, z0)
        
        fig = plt.figure(figsize = (5, 5))
        ax = fig.add_subplot(111, projection = "3d")
        
        ax.plot(self.x_list, self.y_list, self.z_list)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        ax.set_title("Lorenz Position")
        
        plt.show()
        
        