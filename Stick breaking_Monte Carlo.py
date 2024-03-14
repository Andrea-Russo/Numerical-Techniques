"""
Here, we prove the famous Stick Breaking problem witha Monte Carlo simulation.
If a stick of unit lenght is broken in three parts uniform at random,
what is the probability that you could form a triangle with the three pieces?

"""
# Import libraries
import numpy.random as rnd

# Set up number of Monte Carlo simulations
n_samples = 10000

# Generate breaking points of the stick
X = rnd.uniform(0, 1, n_samples)
Y = rnd.uniform(0, 1, n_samples)

# This variable counts the number of triangles formed
triangle = 0

# Monte carlo simulation
for i in range(n_samples):
    
    # It is important to set the correct min/max variable as L1 and L3
    Lenght_1 = min(X[i], Y[i])
    Lenght_2 = abs(Y[i] - X[i])
    Lenght_3 = 1 - max(X[i], Y[i])
    
    # If it is a triangle, count it
    if (Lenght_1 + Lenght_2 > Lenght_3) and  (Lenght_1 + Lenght_3 > Lenght_2) and (Lenght_2 + Lenght_3 > Lenght_1):
        triangle += 1

# Compute probability of triangle forming
prob = round(triangle / n_samples,3)
    
# Print resulting probability
print(f"The probability of getting a triangle is: {prob*100}%")