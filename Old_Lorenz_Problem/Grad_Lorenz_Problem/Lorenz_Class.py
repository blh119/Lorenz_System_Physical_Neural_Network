 # -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:32:51 2023

@author: holli
"""

import numpy as np
import matplotlib.pyplot as plt

dt_obs = 0.1 # how often to observe solution
dt_par = 0.1 # how often to update param

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

        self.init_x, self.init_y, self.init_z = x, y, z
        
    
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
        
    def solve_lorenz_split_1(self, t0, tf, steps, mu1):
        
        #dt_par_int = int(dt_par / dt)
        
        sol_full_set = []
        ndg_full_set = []
        g_full_set = []
        
        time_steps = abs(linear_space(t0, tf, steps) - tf) # This function gives us our our time steps to solve the problem
        
        counter = 0
        
        for time_step in time_steps:
            
            dt = 1/time_step
            
            print("dt: ", dt)
                
            for i in range(5): 
                
                print("Total Time: ", abs(time_step - tf), "\t", "Time Step: ", i + 1)
                
                sol = np.empty((int(time_step), 3))   
                ndg = np.empty((int(time_step), 3))   
                
                if time_step == tf:
                    
                    sol[0] = np.array([self.sigma*(-self.init_x + self.init_y), 
                                       -self.init_y + self.rho*self.init_x - self.init_x*self.init_z,
                                       -self.beta*self.init_z + self.init_x*self.init_y])
                    
                    ndg[0] = np.array([self.sigma*(-self.init_x + self.init_y), 
                                       -self.init_y + self.rho*self.init_x,
                                       -self.beta*self.init_z + self.init_x*self.init_y])
                    
                    g_ = 0
                    
                    u = sol[i]
                    v = ndg[i]
                    
                    print("U: ", u)
                    print("V: ", v)
                
                    # known solution 
                    u_ = np.array([self.sigma*(-u[0] + u[1]), -u[1] + self.rho*u[0] - u[0]*u[2], -self.beta*u[2] + u[0]*u[1]])
                
                    # unknown solution
                    f_ = np.array([self.sigma*(-v[0] + v[1]), -v[1] + self.rho*v[0], -self.beta*v[2] + v[0]*v[1]])
                    v_ = f_ + np.array([-mu1*(v[0] - u[0]), g_, 0])
                    
                    sol[i+1] = u + (dt*u_)
                    ndg[i+1] = v + (dt*v_)
                    
                    print(sol[i+1])
                    print(ndg[i+1])
                
                else:
                    
                    print(sol_full_set)
                    print(ndg_full_set)
                    
                    print(len(sol_full_set))
                    print(len(ndg_full_set))
                    
                    sol[0] = np.array([sol_full_set[counter][len(sol_full_set[counter]) - 1][0], 
                                       sol_full_set[counter][len(sol_full_set[counter]) - 1][1], 
                                       sol_full_set[counter][len(sol_full_set[counter]) - 1][2]])                    
                    
                    ndg[0] = np.array([ndg_full_set[counter][len(ndg_full_set[counter]) - 1][0],
                                       ndg_full_set[counter][len(ndg_full_set[counter]) - 1][1],
                                       ndg_full_set[counter][len(ndg_full_set[counter]) - 1][2]])
                    
                    g_ = -ndg[0,0]*ndg[0,2] # 
                    
                    counter = counter + 1
                    
                    u = sol[i]
                    v = ndg[i]
                    
                    print("U: ", u)
                    print("V: ", v)
                
                    # known solution 
                    u_ = np.array([self.sigma*(-u[0] + u[1]), -u[1] + self.rho*u[0] - u[0]*u[2], -self.beta*u[2] + u[0]*u[1]])
                
                    # unknown solution
                    f_ = np.array([self.sigma*(-v[0] + v[1]), -v[1] + self.rho*v[0], -self.beta*v[2] + v[0]*v[1]])
                    v_ = f_ + np.array([-mu1*(v[0] - u[0]), g_, 0])
                    
                    sol[i+1] = u + (dt*u_)
                    ndg[i+1] = v + (dt*v_)
                    
                    print("Sol: ", sol[i+1])
                    print("Ndg: ", ndg[i+1])
  
                if i == 4:
                    
                    sol_full_set.append(sol) 
                    ndg_full_set.append(ndg) 
                    g_full_set.append(g_)
                                 
        self.g, self.sol, self.ndg = g_full_set, sol_full_set, ndg_full_set
        
        
        # self.idx = np.arange(0, self.t.size, 1000) # plot every 1000 timesteps
        # these equations represent delta t
        self.abs_u = abs(self.ndg[0]-self.sol[0])
        self.abs_v = abs(self.ndg[1]-self.sol[1])
        self.abs_w = abs(self.ndg[2]-self.sol[2])
        
        
    def plot_absolute_errors(self):
        
        plt.figure(figsize=(8,3))
        plt.plot(self.t[self.idx], self.abs_u, lw=1, label=r'$|u|$')
        plt.plot(self.t[self.idx], self.abs_v, lw=1, label=r'$|v|$')
        plt.plot(self.t[self.idx], self.abs_w, lw=1, label=r'$|w|$')
        plt.title('Evolution of absolute errors'); plt.xlabel('t'); plt.yscale('log')
        plt.legend(); plt.grid(); plt.tight_layout()
        plt.show()
        
        
lorenz_1 = Lorenz(10., 28., 8/3)
lorenz_1.get_ic(60., 59.9, 10.)
test = lorenz_1.solve_lorenz_split_1(0, 100000, 100, 100.)
lorenz_1.plot_absolute_errors()

print(lorenz_1.init_x)
print(lorenz_1.init_y)
print(lorenz_1.init_z)


