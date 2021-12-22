import matplotlib.pyplot as plt
import numpy as np
from functions import *

# filename= "explicit_scheme.txt"
# u, time_step, parameters = read_file(filename= filename, number_of_lines= 10)
# print(time_step)

# # Plot
# N = u.shape[0]
# plot_2D(u, time_step, parameters)
# plt.title("Explicit Scheme")


filename= "NN.txt"
# filename= "explicit_scheme_Nx_10_Nt_20_NHL_2_125.txt"
# filename= "explicit_scheme_Nx_10_Nt_20_NHL_2_30.txt"
u, time_step, parameters = read_file(filename= filename, number_of_lines= 10)
print(time_step)

# # Plot
N = u.shape[0]
plot_2D(u, time_step, parameters)
plt.title("FFNN")

plot_line(u, time_step, parameters, idx= round(N/2))
plot_line(u, time_step, parameters, idx= round(-1))