import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


A = np.array([[0, 1], [-2, -3]])
print(A)