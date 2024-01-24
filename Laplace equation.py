"""
Laplace equation in Python \nabla^2 V = 0
https://www.youtube.com/watch?v=f4Xnz7BHhpE

As a setup, we discretize the second order derivatoves forming the laplace operator in 3d
setting dx=dy=dz, we find that:
    
    V_{i,j,k} = 1/6 (V_{i+1,j,k}+V_{i-1,j,k}+V_{i,j+1,k}+V_{i,j-1,k}+V_{i,j,k+1}+V_{i,j,k-1})
    
Therefore, if taken an initial guess we average the potential with these 6 points given a set of boundary conditions,
it will settle to the solution given enought time.
"""

import numpy as np
from scipy.ndimage import convolve, generate_binary_structure
import matplotlib.pyplot as plt


"""
We solve for parallel conducting plates inside an insulating box, this has Neumann boundary condition dV/dn = 0
BY THE WAY, THIS IS A GOOD APPROXIMATION FOR INFINITE BOUNDARY CONDITIONS. 
IT IS LIKE SETTING V=0 AT INFINITY WHICH CANNOT BE DONE ON A PC

1) First, we will construct the grid for the potential. 
2) Then, rather than solving for \nabla^2 V = 0, we will solve for \nabla^2 (V/V_0) = 0, which is equivalent, to keep it dimensionless.

"""

# Set V/V_0 = 0.5 and create a 100x100x100 grid
N = 100
grid = np.zeros((N,N,N)) + 0.5 

# Now we set the location of the parallel plates equal to V/V_0 = 1 and V/V_0 = 0. 
# Then get the location in boolean array format.
grid[30:70, 30:70, 20] = 1
grid[30:70, 30:70, 80] = 0

mask_pos = grid==1 # True at the positive potential plate
mask_neg = grid==0 # True at the negative potential plate


# Now use a meshgrid to come up with an initial guess for the potential.
# The meshgrid returns 3 3-dimensional arrays. The information at each voxel tells you what value of x,y,z you are at at that point
yv, xv, zv = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))

# Now we set the initial geuss for the potential. One can see that this does well for 200 iterations
# If we comment this line out, we need about 4000 iterations to get the same result.
grid = 1 - zv/100 
plt.contour(grid[0])


# Now that we have an initial guess, we can average the potential over the 6 points with a convolution.
kern = generate_binary_structure(3, 1).astype(float)/6
kern[1,1,1] = 0 # The center point must be set manually
kern


''' Boundary Condition:
    Discretised Neumann conditions are equivaklent to setting the last 2 point before the boundary
    to be equal to each other. In this way the discretised derivative \Delta V / \Delta x = 0 
    satifies the condition. For exaple, at x=0 sets V_{0,j,k}=V{1,j,k}'''

# Before proceeding we create a function implementing boundary conditions

def neumann(a):
    a[0,:,:] = a[1,:,:]; a[-1,:,:] = a[-2,:,:]
    a[:,0,:] = a[:,1,:]; a[:,-1,:] = a[:,-2,:]
    a[:,:,0] = a[:,:,1]; a[:,:,-1] = a[:,:,-2]
    return a

# Now we do the convolution.
# This funciton implements the updating of the grid by applying a sequence of convolutions.
# We also keep track of the error between consecutive iterations using the err list
err =[]
iters = 2000
for i in range(iters):
    grid_updated = convolve(grid,kern, mode='constant') # mode is for boundary conditions, but we deal with them on our own so no need to worry
    
    # Boundary conditions (Neumman)
    grid_updated = neumann(grid_updated)
    
    # Boundary conditions (Dirichlett) on the plates
    grid_updated[mask_pos] = 1
    grid_updated[mask_neg] = 0
    
    # See what mean square error is between consecutive arrays
    err.append(np.mean(grid - grid_updated)**2)
    grid = grid_updated
  
# %%    
# Now plot the potential at a gicven slice
plot_slice = 50

plt.figure(figsize=(10,6))
CS = plt.contour(np.arange(100)/100, np.arange(100)/100, grid[:,plot_slice], levels=40)
plt.clabel(CS, CS.levels, inline=True, fontsize=6)
plt.xlabel('$z/z_0$')
plt.ylabel('$y/y_0$')
plt.axvline(0.2, ymin=0.3, ymax=0.7, color='r')
plt.axvline(0.8, ymin=0.3, ymax=0.7, color='g')
plt.show()

# %%
''' Plot error '''
plt.semilogy(np.sqrt(np.array(err)))
plt.xlabel('Iteration', fontsize=20)
plt.ylabel(r'MSE')
plt.grid()
plt.show()
