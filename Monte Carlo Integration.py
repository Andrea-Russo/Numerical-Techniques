"""
Monte Carlo integration

Given that <f(x)> = (1/(b-a)) \int_a^b f(x) dx
one can obtain the integral from (b-a)<f(x)> = \int_a^b f(x) dx.

However, this is very computationally expensive. 
Hence, we discretise the average in N points as
(b-a)<f(x)> = (b-a)(1/N)\sum_i f(x_i)
and then we use Monte Carlo

Let's compute int_0^\pi sin(x)dx = 2

"""

""" 
Quick note on random modules: numpy.random vs scipy.random

    Functionality: Both modules offer a wide range of functions for generating random numbers 
    from various distributions. numpy.random is often sufficient for 
    most basic and intermediate statistical needs. scipy.random, on the other hand, 
    is part of SciPy, which includes more advanced statistical functions and distributions. 
    If your work requires these advanced statistical capabilities, then using SciPy as a whole makes sense.

    Performance: In terms of performance, there isn't a significant difference between the two for generating random numbers. 
    Both are built on well-optimized low-level code.

    API Differences: It's important to note that scipy.random is essentially an alias for numpy.random, 
    as SciPy’s random module doesn’t actually provide its own random number generators but rather imports those from NumPy. 
    Therefore, calling scipy.random is the same as calling numpy.random.

    Updates and Future Compatibility: NumPy's random module underwent a significant overhaul in version 1.17 with the introduction 
    of a new random number generation system, including new generator objects and improved seeding technique. This change was aimed 
    at providing better statistical properties and reproducibility. When using newer features and ensuring future compatibility, 
    it's recommended to use numpy.random with the new random generation system.

"""

import numpy.random as rnd
import numpy as np

# Parameters
N = 10000  # Number of random samples
a, b = 0, np.pi  # Integration limits

# Generate N uniform random samples in the interval [a, b]
xrand = rnd.uniform(a, b, N)

# Compute the function values at these points
f_values = np.sin(xrand)

# Compute the mean of these function values
mean_f = np.mean(f_values)

# Multiply by the range to get the integral estimate
integral_estimate = mean_f * (b - a)

print(f'The integral from 0 to pi of sin(x) is approximately: {integral_estimate}')

