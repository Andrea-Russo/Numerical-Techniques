"""
Derivatives in Python: Symbolic and Numeric
https://www.youtube.com/watch?v=DeeoiE22bZ8
"""

import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
from scipy.misc import derivative

# %%
''' Symbolic derivative '''
x ,a, b, c = smp.symbols(' x a b c', real=True)
f = smp.exp(-a*smp.sin(x**2)) * smp.sin(b**x) * smp.log(c*smp.sin(x)**2/x)

dfdx = smp.diff(f, x)
print('First derivative: ',dfdx)

# To take the n't derivative (ex 4)
d4fdx4 = smp.diff(f, x, 4)
#print('Fourth derivative: ',d4fd4x)
print('That was long...')

# One can then compute numerical values by plugging in numbers
d4fdx4.subs([(x,4), (a,1), (b,2), (c,3)]).evalf() # Without the .evalf() it would just pkug the numbers in the symbolic expression

# Can also convert to a numerical function for plotting
d4fdx4_f = smp.lambdify((x,a,b,c), d4fdx4) # the _f is convention to signal it is a function
x = np.linspace(1,2,100)
y = d4fdx4_f(x, a=1, b=2, c=3)

plt.plot(x,y)
plt.xlabel('$x$', fontsize=12)
plt.ylabel('$d^4 f / dx^4$', fontsize=12)
plt.title('Fourth derivative of f')
plt.show()


# %%
''' Derivative of numerical data'''

x, y = np.loadtxt('C:/Users/Andrea/Desktop/Finance/Python/Data/sample_data1.txt')
plt.plot(x,y, 'o--')

# Basic way: This works well only with smooth data, which is rarely what one has
dydx = np.gradient(y,x)
plt.plot(x, y, label='$y(x)$')
plt.plot(x, dydx, '--o', label='$y\'(x)$')
plt.legend()
plt.show()

# Refined way, one smooths the data before taking the derivative
x, y = np.loadtxt('C:/Users/Andrea/Desktop/Finance/Python/Data/sample_data2.txt')
plt.plot(x,y, 'o--')
plt.show()  # Taking the derivative now will teach you nothing

''' Data smoothing by convolution with a rectangle:
    This consists in smoothing by taking the average in a 15 points window.
    The convolution with a rectangle of area one is essentially a moving average'''
    
x, y = np.loadtxt('C:/Users/Andrea/Desktop/Finance/Python/Data/coviddata.txt')
dydx = np.gradient(y,x)

filter_window = np.ones(15) / 15

y_smooth = np.convolve(y, filter_window, mode='valid') 
''' Mode ‘valid’ returns output of length max(M, N) - min(M, N) + 1. The convolution product is only given for points where the signals overlap completely. 
    Values outside the signal boundary have no effect. '''

dysdx = np.gradient(y_smooth, x[7:-7])

fig, ax = plt.subplots(2, 1)
ax[0].plot(x,y, label='$y(x)$')
ax[0].plot(x[7:-7], y_smooth, label='$y_{smooth}(x)$')

ax[1].plot(x, dydx, label='$y\'(x)$', color='r')
ax[1].plot(x[7:-7], dysdx, label='$y_{smooth}\'(x)$', color='purple')
ax[1].set_ylim(-100,120)
ax[1].grid()
[a.legend() for a in ax]
[a.set_xlabel('Time [Days]') for a in ax]
ax[0].set_ylabel('Cases per day')
ax[1].set_ylabel('$\Delta$ Cases per Day / $\Delta t$')
fig.tight_layout()
plt.show()


# %%
''' Quasi-Symbolic case.
    This is when the function is known, but it is not given in a form that one can cimpute the derivative of as normal
    For example: f(u)=max(|e^{-x_i u^2}-y_i|) for some x_i and y_i'''

x = np.linspace(0, 1, 500)
y = np.exp(-x*2.15**2) + 0.1*np.random.randn(len(x))

plt.scatter(x,y)
plt.xlabel('$x_i$', fontsize=20)
plt.ylabel('$y_i$', fontsize=20)
plt.show()

def f(u):
    return max(np.abs(np.exp(-x*u**2)-y))

u = np.linspace(0, 10, 40)
f_u = np.vectorize(f)(u) # !!! Thiss will create a for loop that evaluates f for all the specified values of u

plt.plot(u, f_u, '--o')
plt.xlabel('$u$', fontsize=20)
plt.ylabel('$f(u)$', fontsize=20)
plt.show()

# To compute the derivative, one could just use np.gradient, but here we have more information and we want to use it
dfdu = np.vectorize(derivative)(f, u, dx=1e-6)

plt.plot(u, dfdu)
plt.xlabel('$u$', fontsize=20)
plt.ylabel('$f(u)$', fontsize=20)
plt.show()




