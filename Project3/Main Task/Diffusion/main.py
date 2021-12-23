from functions import *
import explicit_scheme as es
import matplotlib.pyplot as plt
plt.close("all")

# Default preset
# ES = es.Explicit_scheme()
# Plot results
# ES.solver(number_of_lines= 20)
# Write to file
# ES.solver(write= True)

# Definied preset
ES = es.Explicit_scheme(N = 50, xstart = 0, xstop = 1, alpha = 0.01, T = 1)
# Plot results
# ES.solver(number_of_lines= 20)
# Write to file
# ES.solver(write= True, filename= "explicit_scheme.txt")
# For automatically generated filenames
ES.solver(write= True, filename= None)

# Read from file (Default)
# u, time_step, parameters = read_file()
# u, time_step, parameters = read_file(number_of_lines= "all")

# Read from file (user defined)
# filename= ES.filename # The filename for this run no matter the name
# filename= "explicit_scheme.txt"
# u, time_step, parameters = read_file(filename= filename, number_of_lines= 20)

# Plot
# simple_plot_2D(u, time_step, parameters)
# plt.title(f"alpha: {ES.alpha} Nx {ES.Nx} Nt {ES.Nt}")