"""
Integrals in Python: Symbolic and Numeric

"""
import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt

# For when you know the functional form of the integrand
from scipy.integrate import quad 

# For integrating numerical data without functional form
from scipy.integrate import cumulative_trapezoid

# %%
''' Analytical solution of the integral '''
x = smp.symbols('x', real=True)
f = smp.sin(x)**3 * smp.exp(-5*x)

print(smp.integrate(f, x))

a, b = smp.symbols('a b', real=True, positive=True)
f = smp.cos(b*x) * smp.exp(-a*x)

print(smp.integrate(f, x).simplify())

f = (1+smp.sqrt(x))**smp.Rational(1,3) / smp.sqrt(x)
print(smp.integrate(f, (x,smp.log(4),smp.oo))) # definite integral to infinity

# Unsolvable integrals will not run

# %%
''' When the integral does not have an analytical solution, use quad'''

f = lambda x: np.exp(-np.sin(x)) # Returns the value of f when a number is inserted (ex f(2))
print('The integral result and the error are: ',quad(f, 1, 2*np.pi))

# If the function has constants, this can also be done for many different constant values
def f(x, a, b):
    return 1/((a-np.cos(x))**2 + (b-np.sin(x))**2)

a_array = np.arange(2,10,1)
b_array =np.arange(2,10,1)

integrals = [[a, b, quad(f, 0, 2*np.pi, args=(a,b))] for a in a_array for b in b_array ]
for i in integrals:
    print(i)
    
# %%
""" When instead we only have data, we can use cumulative trapezoid """

x, y = np.loadtxt('C:/Users/Andrea/Desktop/Finance/Python/Data/sample_data3.txt')
plt.plot(x,y)
plt.xlabel('Tiime [ns]')
plt.ylabel('Current [mA]')
plt.show()

# Energy is proportional to the integrated current
integral = cumulative_trapezoid(y, x, initial=0) # y goes first. Initial sets the integration constant
plt.plot(x,integral)
plt.xlabel('Tiime [ns]')
plt.ylabel('Integrated Current [mA]')
plt.show()


