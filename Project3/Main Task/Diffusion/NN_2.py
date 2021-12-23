import tensorflow as tf
import numpy as np
import math as m


# Define trial solution, ODE rhs, loss function, and gradient method
# @tf.function
def initial_condition(x):
    print("1here")
    pi = tf.constant(m.pi, dtype=tf.float64)
    return tf.math.sin(pi*x)

# @tf.function
def trial_solution(model, point):
    print("2here")
    # print(point)
    # print(point, "\n next \n", point[0][0], "\n next \n", point[1][0])
    # print(point, "\n next \n", point[0], "\n next \n", point[1])
    # x, t = point
    # print(point, point)
    x = point[0][0].numpy(); t = point[1][0].numpy()
    # return (1-point[1])*initial_condition(point[0]) + point[0]*(1-point[0])*point[1]*model(point)
    return (1-t)*initial_condition(x) + x*(1-x)*t*model(point)
    # return tf.einsum('i..., j->...ij', tf.exp(-t), x0)[0] + tf.einsum('i...,ij->...ij', (1-tf.exp(-t)), model(t))[0]

# @tf.function
def lhs(model, point):
    print("3here")
    # x, t = point
    x = point[0]; t = point[1]
    with tf.GradientTape() as tape:
        tape.watch(point[1][0])
        trial = trial_solution(model, point)
    d_trial_dt = tape.gradient(trial, point[1][0])
    return d_trial_dt

# @tf.function
def rhs(model, point):
    print("4here")
    # x, t = point
    x = point[0]; t = point[1]
    with tf.GradientTape() as g:
        g.watch(point[0])
        with tf.GradientTape() as gg:
            gg.watch(point[0])
            trial = trial_solution(model, point)
        # d_trial_dx = gg.gradient(trial, x)
    d2_trial_dx2 = g.gradient(trial, point[0])
    return d2_trial_dx2


# @tf.function
def loss(model, point):
    print("5here")
    # x, t = point
    x = point[0]; t = point[1]
    t_deriv = lhs(model, point)
    x_deriv = rhs(model, point)
    print(f"t_deriv: {t_deriv}")
    # print(f"x_deriv: {x_deriv}")
    return tf.losses.MSE(t_deriv, x_deriv)

# @tf.function
def grad(model, point):
    print("6here")
    print(point)
    # x, t = point
    # x = point[0]; t = point[1]
    
    with tf.GradientTape() as tape:
        loss_value = loss(model, point)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Define model
class DNModel(tf.keras.Model):
    def __init__(self, x, t, NHN, Nt= 11, T= 1):
        super(DNModel, self).__init__()
        
        # Problem formulation for tensorflow
        self.x = tf.convert_to_tensor(x, dtype=tf.float64)
        self.t = tf.convert_to_tensor(t, dtype=tf.float64)
        
        self.NHL = len(NHN) # Number of Hidden Layers
        self.NHN = NHN # Number of Hidden Nodes
        for i in range(self.NHL):
            var = 'dense_' + str(i+1)
            setattr(self, var, tf.keras.layers.Dense(NHN[i], activation=tf.nn.sigmoid, dtype=tf.float64))
            
        self.out = tf.keras.layers.Dense(1, name="output", dtype=tf.float64)

    # def call(self, inputs):
    #     x = inputs
    #     for i in range(self.NHL):
    #         var = 'dense_' + str(i+1)
    #         x = self.__dict__[var](x)
        
    #     return self.out(x)
    
    def call(self, inputs):
        print("call here")
        x = inputs
        for i in range(self.NHL):
            var = 'dense_' + str(i+1)
            x = self.__dict__[var](x)
        
        return self.out(x)
    
    def solver(self, model, num_epochs= 2000, alpha= 0.01):
        optimizer = tf.keras.optimizers.Adam(alpha)
        for epoch in range(num_epochs):
            for t_ in self.t:
                for x_ in self.x:
                    # point_x = model.x[6]; point_t =  model.t[4]
                    # point = tf.convert_to_tensor([point_x, point_t], dtype= tf.float64)
                    point = [x_, t_]
                    point = tf.reshape(point, [-1,1])
                    # point = tf.convert_to_tensor(point, dtype=tf.float64)
                    print(point, "\n next \n", point[0][0], "\n next \n", point[1][0])
                    cost, gradients = grad(model, point)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    step = optimizer.iterations.numpy()
                    if step == 1:
                        print(f"Step: {step}, " + f"Loss: {tf.math.reduce_mean(cost.numpy())}")
                    if step % 100 == 0:
                        print(f"Step: {step}, " + f"Loss: {tf.math.reduce_mean(cost.numpy())}")


if __name__ == '__main__':
    print("wow")
    # Finding the minimum eigenvalue
    # Initial model and optimizer (NN)
    xstart = 0; xstop = 1
    L = xstop - xstart    
    Nx = 10
    dx = L/Nx
    x_idx = np.arange(Nx+1)
    x = x_idx*dx
    # print(f"index x: {x}")
    
    
    T = 1 # period [s] 
    Nt = 10
    dt = T/Nt
    t_idx = np.arange(Nt+1)
    t = t_idx*dt
    # print(f"index t: {t}")
    
    # NHN = [100]
    NHN = [100, 10]
    # NHN = [100, 50, 25]
    num_epochs = 2000
    
    
    model = DNModel(x, t, NHN, Nt= Nt, T= T)
    model.solver(model)
    # point_x = model.x[6]; point_t =  model.t[4]
    # point = tf.convert_to_tensor([point_x, point_t], dtype= tf.float64)
    # point = tf.reshape(point, [-1,1])
    # out = model(point)
    # model.solver(model, num_epochs= num_epochs)