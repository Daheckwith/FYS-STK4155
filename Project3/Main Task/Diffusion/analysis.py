# import numpy as np
import matplotlib.pyplot as plt
from functions import *

plt.close("all")
# Watch for the directories you might need to specify one

# Explicit Scheme Analysis
sub_Dir = "ES/Files/"
Dir = sub_Dir
# Dir = ""

# Read from file explicti scheme
filename= Dir + "explicit_scheme_Nx_10_Nt_10000.txt"
# filename= Dir + "explicit_scheme_Nx_50_Nt_250000.txt"
print(filename)
u, time_step, parameters = read_file(filename= filename, number_of_lines= 20)
print(f"time_step: {time_step} \n")

# # # Plot
N = u.shape[0]
plot_2D(u, time_step, parameters)
plt.title("Explicit Scheme")


# Neural Network analysis
# NHL1
# NHL = 1
# NHN_ = []
# NHN = [10, 20, 40, 60, 100]

#NHL2
NHL = 2
NHN_ = [60]
NHN = [10, 20, 30, 40, 50, 60]

#NHL3
# NHL = 3
# NHN_ = [60, 60]
# NHN = [60]

avg_MSE_NHN = []
for i in range(len(NHN)):
    filename= f"NN_Nx_10_Nt_20_NHL_{NHL}_{sum(NHN_) + NHN[i]}.txt"
    
    print("\n-----------------------------------------------------------")
    print(f"{filename}")
    u, time_step, parameters = read_file(filename= filename, number_of_lines= 20)
    print(f"time_step: {time_step} \n")
    
    # # # Plot
    
    avg_mse= plot_2D(u, time_step, parameters, NHL= NHL, NHN= NHN[i])
    plt.title("FFNN")
    avg_MSE_NHN.append(avg_mse)
    
    
    plot_lines(u, time_step, parameters, NHL= NHL, NHN= NHN[i], number_of_lines= 10)
    sub_plots(u, time_step, parameters, NHL= NHL, NHN= NHN[i], number_of_lines= 4)
    N = u.shape[0]
    plot_line(u, time_step, parameters, idx= round(N/2), NHL= NHL, NHN= NHN[i])
    plot_line(u, time_step, parameters, idx= round(-1), NHL= NHL, NHN= NHN[i])

plt.figure()
plt.title(f"Average MSE NHL= {NHL}")
plt.plot(NHN, avg_MSE_NHN, "o--")
plt.xlabel("Number of Hidden Nodes "); plt.ylabel("Average Mean Squared Error")
save_fig(f"../Figures/Avg_MSE_NHL_{NHL}.png")
plt.show()

