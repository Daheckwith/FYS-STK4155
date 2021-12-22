import tensorflow as tf
import numpy as np


# Define trial solution, ODE rhs, loss function, and gradient method
@tf.function
def trial_solution(model, x0, t):
    return tf.einsum('i..., j->...ij', tf.exp(-t), x0)[0] + tf.einsum('i...,ij->...ij', (1-tf.exp(-t)), model(t))[0]
    # return tf.einsum('i..., j->ij', tf.exp(-t), x0) + tf.einsum('i...,ij->ij', (1-tf.exp(-t)), model(t)) 

@tf.function
def rhs(model, A, x0, t):
    g = trial_solution(model, x0, t)
    return tf.einsum('ij,ij,kl,il->ik', g, g, A, g) - tf.einsum('ij,jk,ik,il->il', g, A, g, g)

@tf.function
def loss(model, A, x0, t):
    with tf.GradientTape() as tape:
        tape.watch(t)
        trial = trial_solution(model, x0, t)
    d_trial_dt = tape.batch_jacobian(trial, t)[:, :, 0]
    return tf.losses.MSE(d_trial_dt, rhs(model, A, x0, t))


@tf.function
def grad(model, A, x0, t):
    with tf.GradientTape() as tape:
        loss_value = loss(model, A, x0, t)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Define Rayleigh quotient
def ray_quo(A, x):
    return tf.einsum('ij,jk,ik->i', x, A, x) / tf.einsum('ij,ij->i', x, x)

# Define model
class DNModel(tf.keras.Model):
    def __init__(self, A, x0, NHN, Nt= 11, T= 1):
        super(DNModel, self).__init__()
        
        # Problem formulation for tensorflow
        self.A = tf.convert_to_tensor(A, dtype=tf.float64)
        self.n = A.shape[0]
        x0 = x0 / np.linalg.norm(x0)
        self.x0 = tf.convert_to_tensor(x0, dtype=tf.float64)
        start = tf.constant(0, dtype=tf.float64)
        stop = tf.constant(T, dtype=tf.float64)
        t = tf.linspace(start, stop, Nt)
        self.t = tf.reshape(t, [-1, 1])
        
        self.NHL = len(NHN) # Number of Hidden Layers
        self.NHN = NHN # Number of Hidden Nodes
        for i in range(self.NHL):
            var = 'dense_' + str(i+1)
            setattr(self, var, tf.keras.layers.Dense(NHN[i], activation=tf.nn.sigmoid, dtype=tf.float64))
            
        self.out = tf.keras.layers.Dense(self.n, name="output", dtype=tf.float64)

    def call(self, inputs):
        x = inputs
        for i in range(self.NHL):
            var = 'dense_' + str(i+1)
            x = self.__dict__[var](x)
        
        return self.out(x)
    
    def solver(self, model, num_epochs= 2000, alpha= 0.01):
        optimizer = tf.keras.optimizers.Adam(alpha)
        
        for epoch in range(num_epochs):
            cost, gradients = grad(model, self.A, self.x0, self.t)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            step = optimizer.iterations.numpy()
            if step == 1:
                print(f"Step: {step}, " + f"Loss: {tf.math.reduce_mean(cost.numpy())}")
            if step % 100 == 0:
                print(f"Step: {step}, " + f"Loss: {tf.math.reduce_mean(cost.numpy())}")