import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../Diffusion')
from functions import save_fig

plt.close("all")

def max_eigenvalue(model, A, x0, t, x_euler, eig_euler, N, g, eig_nn, Nt):
    s = t.reshape(-1, 1)
    v, w = np.linalg.eig(A)
    
    write_to_file(max_eigenvalue, model, A, x0, v, w, t, x_euler, eig_euler, N, g, eig_nn, Nt)
    
    # Print results
    print()
    print('A =', A)
    print('x0 =', x0)
    print('Eigvals Numpy:', v)
    print('Max Eigval Numpy', np.max(v))
    print('Final Rayleigh Quotient Euler', eig_euler[-1])
    print('Final Rayleigh Quotient FFNN', eig_nn.numpy()[-1])
    print('Absolute Error Euler:', np.abs(np.max(v) - eig_euler[-1]))
    print('Absolute Error FFNN:', np.abs(np.max(v) - eig_nn.numpy()[-1]))
    
    # Plot components of computed steady-state vector
    fig0, ax0 = plt.subplots()
    ax0.axhline(w[0, np.argmax(v)], color='b', ls=':', label=f'Numpy $v_1$={w[0, np.argmax(v)]:.5f}')
    ax0.axhline(w[1, np.argmax(v)], color='g', ls=':', label=f'Numpy $v_2$={w[1, np.argmax(v)]:.5f}')
    ax0.axhline(w[2, np.argmax(v)], color='r', ls=':', label=f'Numpy $v_3$={w[2, np.argmax(v)]:.5f}')
    ax0.axhline(w[3, np.argmax(v)], color='y', ls=':', label=f'Numpy $v_4$={w[3, np.argmax(v)]:.5f}')
    ax0.axhline(w[4, np.argmax(v)], color='k', ls=':', label=f'Numpy $v_5$={w[4, np.argmax(v)]:.5f}')
    ax0.axhline(w[5, np.argmax(v)], color='c', ls=':', label=f'Numpy $v_6$={w[5, np.argmax(v)]:.5f}')
    ax0.plot(t, x_euler[:, 0], color='b', ls='--', label=f'Euler $v_1$={x_euler[-1, 0]:.5f}')
    ax0.plot(t, x_euler[:, 1], color='g', ls='--', label=f'Euler $v_2$={x_euler[-1, 1]:.5f}')
    ax0.plot(t, x_euler[:, 2], color='r', ls='--', label=f'Euler $v_3$={x_euler[-1, 2]:.5f}')
    ax0.plot(t, x_euler[:, 3], color='y', ls='--', label=f'Euler $v_4$={x_euler[-1, 3]:.5f}')
    ax0.plot(t, x_euler[:, 4], color='k', ls='--', label=f'Euler $v_5$={x_euler[-1, 4]:.5f}')
    ax0.plot(t, x_euler[:, 5], color='c', ls='--', label=f'Euler $v_6$={x_euler[-1, 5]:.5f}')
    ax0.plot(s, g[:, 0], color='b', label=f'FFNN $v_1$={g[-1, 0]:.5f}')
    ax0.plot(s, g[:, 1], color='g', label=f'FFNN $v_2$={g[-1, 1]:.5f}')
    ax0.plot(s, g[:, 2], color='r', label=f'FFNN $v_3$={g[-1, 2]:.5f}')
    ax0.plot(s, g[:, 3], color='y', label=f'FFNN $v_4$={g[-1, 3]:.5f}')
    ax0.plot(s, g[:, 4], color='k', label=f'FFNN $v_5$={g[-1, 4]:.5f}')
    ax0.plot(s, g[:, 5], color='c', label=f'FFNN $v_6$={g[-1, 5]:.5f}')
    ax0.set_title(f"Euler $N_t$={N}, FFNN $N_t$={Nt}")
    ax0.set_ylabel('Components of vector, $v$')
    ax0.set_xlabel('Time, $t$')
    ax0.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, borderaxespad=0, ncol=1)
    save_fig(f"../Figures/max_eigvec_comp_{model.NHL}_{sum(model.NHN)}_{N}_{Nt}.png")
    
    # Plot eigenvalues
    fig, ax = plt.subplots()
    ax.axhline(np.max(v), color='red', ls='--')
    ax.plot(t, eig_euler)
    ax.plot(s, eig_nn)
    # ax.scatter(t_tf, eig_nn_points, color='orange')
    ax.set_title(f"Euler $N_t$={N}, FFNN $N_t$={Nt}")
    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('Rayleigh Quotient, $r$')
    lgd_numpy = "Numpy $\\lambda_{\\mathrm{max}} \\sim$ " + \
        str(round(np.max(v), 5))
    lgd_euler = "Euler $r_{\\mathrm{final}} \\sim$ " + \
        str(round(eig_euler[-1], 5))
    lgd_nn = "FFNN $r_{\\mathrm{final}} \\sim$ " + \
        str(round(eig_nn.numpy()[-1], 5))
    plt.legend([lgd_numpy, lgd_euler, lgd_nn], loc='best')
    save_fig(f"../Figures/max_eigval_{model.NHL}_{sum(model.NHN)}_{N}_{Nt}.png")
    plt.show()


def min_eigenvalue(model, A, x0, t, x_euler, eig_euler, N, g, eig_nn, Nt):
    s = t.reshape(-1, 1)
    v, w = np.linalg.eig(A)
    
    write_to_file(min_eigenvalue, model, A, x0, v, w, t, x_euler, eig_euler, N, g, eig_nn, Nt)
    
    # Print results
    print()
    print('A =', A)
    print('x0 =', x0)
    print('Eigvals Numpy:', v)
    print('Min Eigval Numpy', np.min(v))
    print('Eigvec Numpy:', w[:, 1])
    print('Final Rayleigh Quotient Euler', eig_euler[-1])
    print('Final Rayleigh Quotient FFNN', eig_nn.numpy()[-1])
    print('Absolute Error Euler:', np.abs(np.min(v) - eig_euler[-1]))
    print('Absolute Error FFNN:', np.abs(np.min(v) - eig_nn.numpy()[-1]))
    
    # Plot components of computed steady-state vector
    fig0, ax0 = plt.subplots()
    ax0.axhline(w[0, np.argmin(v)], color='b', ls=':', label=f'Numpy $v_1$={w[0, np.argmin(v)]:.5f}')
    ax0.axhline(w[1, np.argmin(v)], color='g', ls=':', label=f'Numpy $v_2$={w[1, np.argmin(v)]:.5f}')
    ax0.axhline(w[2, np.argmin(v)], color='r', ls=':', label=f'Numpy $v_3$={w[2, np.argmin(v)]:.5f}')
    ax0.axhline(w[3, np.argmin(v)], color='y', ls=':', label=f'Numpy $v_4$={w[3, np.argmin(v)]:.5f}')
    ax0.axhline(w[4, np.argmin(v)], color='k', ls=':', label=f'Numpy $v_5$={w[4, np.argmin(v)]:.5f}')
    ax0.axhline(w[5, np.argmin(v)], color='c', ls=':', label=f'Numpy $v_6$={w[5, np.argmin(v)]:.5f}')
    ax0.plot(t, x_euler[:, 0], color='b', ls='--', label=f'Euler $v_1$={x_euler[-1, 0]:.5f}')
    ax0.plot(t, x_euler[:, 1], color='g', ls='--', label=f'Euler $v_2$={x_euler[-1, 1]:.5f}')
    ax0.plot(t, x_euler[:, 2], color='r', ls='--', label=f'Euler $v_3$={x_euler[-1, 2]:.5f}')
    ax0.plot(t, x_euler[:, 3], color='y', ls='--', label=f'Euler $v_4$={x_euler[-1, 3]:.5f}')
    ax0.plot(t, x_euler[:, 4], color='k', ls='--', label=f'Euler $v_5$={x_euler[-1, 4]:.5f}')
    ax0.plot(t, x_euler[:, 5], color='c', ls='--', label=f'Euler $v_6$={x_euler[-1, 5]:.5f}')
    ax0.plot(s, g[:, 0], color='b', label=f'FFNN $v_1$={g[-1, 0]:.5f}')
    ax0.plot(s, g[:, 1], color='g', label=f'FFNN $v_2$={g[-1, 1]:.5f}')
    ax0.plot(s, g[:, 2], color='r', label=f'FFNN $v_3$={g[-1, 2]:.5f}')
    ax0.plot(s, g[:, 3], color='y', label=f'FFNN $v_4$={g[-1, 3]:.5f}')
    ax0.plot(s, g[:, 4], color='k', label=f'FFNN $v_5$={g[-1, 4]:.5f}')
    ax0.plot(s, g[:, 5], color='c', label=f'FFNN $v_6$={g[-1, 5]:.5f}')
    ax0.set_title(f"Euler $N_t$={N}, FFNN $N_t$={Nt}")
    ax0.set_ylabel('Components of vector, $v$')
    ax0.set_xlabel('Time, $t$')
    ax0.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, borderaxespad=0, ncol=1)
    save_fig(f"../Figures/min_eigvec_comp_{model.NHL}_{sum(model.NHN)}_{N}_{Nt}.png")
    
    # Plot eigenvalues
    fig, ax = plt.subplots()
    ax.axhline(np.min(v), color='red', ls='--')
    ax.plot(t, eig_euler)
    ax.plot(s, eig_nn)
    ax.set_title(f"Euler $N_t$={N}, FFNN $N_t$={Nt}")
    ax.set_xlabel('Time, $t$')
    ax.set_ylabel('Rayleigh Quotient, $r$')
    lgd_numpy = "Numpy $\\lambda_{\\mathrm{min}} \\sim$ " + \
        str(round(np.min(v), 5))
    lgd_euler = "Euler $r_{\\mathrm{final}} \\sim$ " + \
        str(round(eig_euler[-1], 5))
    lgd_nn = "FFNN $r_{\\mathrm{final}} \\sim$ " + \
        str(round(eig_nn.numpy()[-1], 5))
    plt.legend([lgd_numpy, lgd_euler, lgd_nn], loc='best')
    # plt.legend([lgd_numpy, lgd_euler, lgd_nn], loc='center left', bbox_to_anchor=(1.04, 0.5),
    #           fancybox=True, borderaxespad=0, ncol=1)
    save_fig(f"../Figures/min_eigval_{model.NHL}_{sum(model.NHN)}_{N}_{Nt}.png")
    plt.show()


def write_to_file(function, model, A, x0, v, w, t, x_euler, eig_euler, N, g, eig_nn, Nt):
    import sympy as sym
    
    # Dir = "../files/"
    Dir = "../files/"
    file = Dir + function.__name__ + f"_{model.NHL}_{sum(model.NHN)}_{N}_{Nt}"
    f= open(file, "w")
    f.write(f"{function.__name__}, Euler Nt: {N}, NN Nt: {Nt} \n")
    f.write(f"NHL: {model.NHL}, NHN: ")
    f.writelines(str(i) + "|" for i in model.NHN)
    f.write("\n")
    f.write(f"x0= {x0} \n")
    f.write(f"A=\n {A} \n")
    f.write(f"Eigvectors Numpy:\n {w} \n")
    f.write(f"Eigvals Numpy: {v} ")
    if function.__name__ == "max_eigenvalue":
        f.write(f"Max Eigval Numpy: {np.max(v)} \n")
    elif function.__name__ == "min_eigenvalue":
        f.write(f"Min Eigval Numpy: {np.min(v)} \n")
    
    f.write(f"Final Rayleigh Quotient Euler, {eig_euler[-1]} \n")
    f.write(f"Final Rayleigh Quotient FFNN, {eig_nn.numpy()[-1]} \n")
    if function.__name__ == "max_eigenvalue":
        # idx = np.argmax(v)
        np_vec = w[:, np.argmax(v)]
        f.write(f"Absolute Error Euler: {np.abs(np.max(v) - eig_euler[-1])} \n")
        f.write(f"Absolute Error FFNN: {np.abs(np.max(v) - eig_nn.numpy()[-1])} \n")
    elif function.__name__ == "min_eigenvalue":
        # idx = np.argmin(v)
        np_vec = w[:, np.argmin(v)]
        f.write(f"Absolute Error Euler: {np.abs(np.min(v) - eig_euler[-1])} \n")
        f.write(f"Absolute Error FFNN: {np.abs(np.min(v) - eig_nn.numpy()[-1])} \n")
    
    vec_euler = x_euler[-1]; vec_nn = g.numpy()[-1]
    f.write(f"Final Eigenvector Estimate Euler, {vec_euler} \n")
    f.write(f"Final Eigenvector Estimate FFNN, {vec_nn} \n")
    f.write(f"Eigenvector MAE from Numpy Euler: {np.mean(np.abs(vec_euler - np_vec))} \n")
    f.write(f"Eigenvector MAE from Numpy FFNN: {np.mean(np.abs(vec_nn - np_vec))}\n")
    
    # Rounding up to 4h digit before converting to LaTeX
    A_sym = A.round(4); v_sym = v.round(4); w_sym = w.round(4); 
    x_sym = vec_euler.round(4); g_sym = vec_nn.round(4);
    
    f.write("\n######################################## LATEX ########################################\n\n")
    f.write( "A:" + sym.latex(sym.sympify(A_sym)) + "\n" )
    f.writelines(f"v_{i}: " + str(v_sym[i]) + " | " for i in range(len(v_sym)))
    f.write("\n")
    f.writelines(f"w_{i}: " + sym.latex( sym.sympify( w_sym[:,i].reshape([-1,1]) ) ) + "\n" for i in range(w_sym.shape[1]))
    f.write("\n")
    f.write(f"Final Eigenvector Euler: {sym.latex( sym.sympify( x_sym.reshape([-1,1]) ) )} \n")
    f.write(f"Final Eigenvector FFNN: {sym.latex( sym.sympify( g_sym.reshape([-1,1]) ) )} \n")
    f.close()