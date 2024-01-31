"""
In this program, we solve the 1-dimensional heat equation numerically.
This equation represents how heat diffuses as time passes given an intial heat profile at t=0

We use the finite difference method and show the difference between the explicit 
and implicit method.
While the explicit method is easier to construct, the implcit method is always stable
while the explicit method is stable only fo small number of steps.

We can clearly see that one method is much more stable. 

Stable vs Unstable:

For the grid spacing, look at the difference between 

N_timesteps=1000, N_steps=10  (both stable)

and

N_timesteps=1000, N_steps=100 (only implicit is stable)

and

N_timesteps=100, N_steps=1000 (only implicit is stable)

One sees that the explicit method quikly becomes unstable but the implicit method
improves while always being stable.
    
    
"""

# Import libraries
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% Set up the grid

N_timesteps = 1000          # Steps in the time dimension t 
N_points = 100              # Steps in the spatial dimension x
X_max = 1
T_max = 1

dx = X_max / N_points       # Spatial step size
dt = T_max / N_timesteps    # Time step size
r = dt / dx**2              # This is the stability error rate

"""
The one dimensional heat equation takes the form:

du / dt = d^2u / dx^2 

with boundary conditions

u(t,0) = u(t,1) = 0

and initial conditions

u(x,0) = u_0(x)
"""

# %% Explicit Method

"""
For the explicit method, we approximate the time derivative with a forward difference:

du / dt = (u_i^{k+1} - u_i^k) / dt

and the space derivative with a second order central difference:

d^2u / dx^2  = (u^k_{i+1}-2u_i^k+u^k_{i-1}) / dx^2 

Hence, the heat equation can be rewritten as an **EXPLICIT** equation:

u_i^{k+1} = \left(1-\frac{2\delta t}{(\delta x)^2}\right)u_i^k + \frac{\delta t}{(\delta x)^2}(u^k_{i+1} + u^k_{i-1})

This method is stable for dt / dx^2 <= 1/2
"""

# Check stability conditoon
if r <= 1/2:
    print('The Explicit method is: Stable')
else:
    print('The Explicit method is: Unstable')
    
# Create axes
x = np.linspace(0, X_max, N_points)
t = np.linspace(0, T_max, N_timesteps)

# SINE WAVE INITIAL CONDITION
u1 = np.zeros([N_timesteps, N_points])
u1[0,:] = np.round(np.sin(np.pi*x + np.pi/2), 6)

# Solve equations (boundary conditions are zero so it is enough not to update them)
for k in range(N_timesteps-1):
    for i in range(1, N_points-1):
        u1[k+1,i] = (1-2*r)*u1[k,i] + r*(u1[k,i+1] + u1[k,i-1])

# Plot SUBE WAVE INITIAL CONDITION
X, T = np.meshgrid(x, t)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(X, T, np.round(u1, 3), cmap='viridis')
ax.set_title('Explicit: Sine Initial Condition', fontsize=14)
ax.set_xlabel('Length', fontsize=12)
ax.set_ylabel('Time', fontsize=12)
ax.view_init(elev=20, azim=30)
fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to provide scale
plt.show()

# DIRAC DELTA IMPULSE INITIAL CONDITION
u2 = np.zeros([N_timesteps, N_points])
u2[0, round(N_points/2)] = 1

# Solve equations (boundary conditions are zero so it is enough not to update them)
for k in range(N_timesteps-1):
    for i in range(1, N_points-1):
        u2[k+1,i] = (1 - 2*r) * u2[k,i] + r*(u2[k,i+1] + u2[k,i-1])

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(X, T, np.round(u2, 3), cmap='viridis')
ax.set_title('Explicit: Dirac Initial Condition', fontsize=14)
ax.set_xlabel('Length', fontsize=12)
ax.set_ylabel('Time', fontsize=12)
ax.view_init(elev=20, azim=30)
fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to provide scale
plt.show()

# GAUSSIAN INITIAL CONDITION

u3=np.zeros([N_timesteps,N_points])
mean=0.5
var=0.01
u3[0,:]=np.round((1/np.sqrt(2*np.pi*var))*np.exp(-((x-mean)**2)/var),3)

# Solve equations (boundary conditions are zero so it is enough not to update them)
for k in range(N_timesteps-1):
    for i in range(1,N_points-1):
        u3[k+1,i] = (1 - 2*r)*u3[k,i] + r*(u3[k,i+1] + u3[k,i-1])
        
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(X, T, np.round(u3, 3), cmap='viridis')
ax.set_title('Explicit: Gaussian Initial Condition', fontsize=14)
ax.set_xlabel('Length', fontsize=12)
ax.set_ylabel('Time', fontsize=12)
ax.view_init(elev=20, azim=30)
fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to provide scale
plt.show()


#%% IMPLICIT METHOD

"""
For the implicit method, we still approximate the time derivative with a forward difference:

du / dt = (u_i^{k+1}-u_i^k)/ dt

and the space derivative with a second order central difference, but this time we use a backward marching scheme:

d^2u / dx^2  = (u^{k+1}_{i+1}-2u_i^{k+1}+u^{k+1}_{i-1}) / dx^2

Hence, the heat equation can be rewritten as an **IMPLICIT** equation:

\left(1+\frac{2\delta t}{(\delta x)^2}\right)u_i^{k+1}-\frac{\delta t}{(\delta x)^2}(u^{k+1}_{i+1} + u^{k+1}_{i-1})=u_i^k 

This method is **ALWAYS** stable, but more computationally expensive.

This is now solved by matrix inversion. 
At each timestep we invert the matrices at the (k+1) so that we can extract 
u^{k+1} from u^k together with the boundary conditions.
"""

# Create matrix of coefficients M
M = np.diag([1+2*r]*(N_points-2)) + np.diag([-r]*(N_points-3),-1) + np.diag([-r]*(N_points-3),1)

# SINE WAVE INTIAL CONDITION
u1_impl=np.zeros([N_timesteps,N_points])
u1_impl[0,:]=np.round(np.sin(np.pi*x + np.pi/2),5)

for k in range(1, N_timesteps):
    # Now extract initial condition vector without the boundary conditions
    u = u1_impl[k-1,1:-1].copy()

    # Construt right hand side vector of u by adding boundary conditions at next time step
    u[0] = u[0] + r*u1_impl[k,0]
    u[-1] = u[-1] + r*u1_impl[k,-1]
    
    # Solve the matrix inversion equation
    solution = np.linalg.solve(M, u)
    u1_impl[k,1:-1] = solution

# Plot
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(X, T, np.round(u1_impl, 3), cmap='viridis')
ax.set_title('Implicit: Sine Initial Condition', fontsize=14)
ax.set_xlabel('Length', fontsize=12)
ax.set_ylabel('Time', fontsize=12)
ax.view_init(elev=20, azim=30)
fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to provide scale
plt.show()

# DIRAC DELTA INITIAL CONDITION AS A HEATH SPIKE HALF WAY THROUGH THE X-AXIS
u2_impl=np.zeros([N_timesteps, N_points])
u2_impl[0,round(N_points/2)] = 1


for k in range(1, N_timesteps):
    # Now extract initial condition vector without the boundary conditions
    u=u2_impl[k-1,1:-1].copy()

    # Construt right hand side vector of u by adding boundary conditions at next time step
    u[0] = u[0] + r*u2_impl[k,0]
    u[-1] = u[-1] + r*u2_impl[k,-1]
    
    # Solve the matrix inversion equation
    solution = np.linalg.solve(M, u)
    u2_impl[k,1:-1] = solution

# Plot
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(X, T, np.round(u2_impl, 3), cmap='viridis')
ax.set_title('Implicit: DIRAC Initial Condition', fontsize=14)
ax.set_xlabel('Length', fontsize=12)
ax.set_ylabel('Time', fontsize=12)
ax.view_init(elev=20, azim=30)
fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to provide scale
plt.show()

# GAUSSIAN INITIAL CONDITION 
u3_impl = np.zeros([N_timesteps,N_points])
mean = 0.5
var = 0.1
u3_impl[0,:] = np.round((1/np.sqrt(2*np.pi*var))*np.exp(-((x-mean)**2)/var),3)

for k in range(1, N_timesteps):
    # Now extract initial condition vector without the boundary conditions
    u = u3_impl[k-1,1:-1].copy()

    # Construt right hand side vector of u by adding boundary conditions at next time step
    u[0] = u[0] + r*u3_impl[k,0]
    u[-1] = u[-1] + r*u3_impl[k,-1]
    
    # Solve the matrix inversion equation
    solution = np.linalg.solve(M,u)
    u3_impl[k,1:-1] = solution
    
# Plot
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(X, T, np.round(u3_impl, 3), cmap='viridis')
ax.set_title('Implicit: GAUSSIAN Initial Condition', fontsize=14)
ax.set_xlabel('Length', fontsize=12)
ax.set_ylabel('Time', fontsize=12)
ax.view_init(elev=20, azim=30)
fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to provide scale
plt.show()




