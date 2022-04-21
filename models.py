""" Defines simple neural ODE model. """
from functools import partial
from math import sqrt
import tensorflow as tf
from odeint import odeint


class LeakyComponent(tf.keras.layers.Layer):
  def __init__(self):
    super(LeakyComponent, self).__init__()

  def build(self, input_shape):
    tau_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
    self.tau = tf.Variable(
        initial_value=1-tau_init(shape=(1, input_shape[-1]),
                            dtype='float32'),
        trainable=True)
    self.mult = tf.keras.layers.Multiply()
    super(LeakyComponent, self).build(input_shape)  # Be sure to call this at the end

  def call(self, x):
    result = self.mult([x,-1/self.tau])
    return result

  def compute_output_shape(self, input_shape):
    return input_shape[0]

class ConductanceConnectivity(tf.keras.layers.Layer):
  def __init__(self):
    super(ConductanceConnectivity, self).__init__()

  def build(self, input_shape):
    bias_init = tf.random_normal_initializer(mean=1.0, stddev=0.5)
    self.bias = tf.Variable(
        initial_value=bias_init(shape=(1, input_shape[-1]),
                            dtype='float32'),
        trainable=True)
    self.add = tf.keras.layers.Add()
    super(ConductanceConnectivity, self).build(input_shape)  # Be sure to call this at the end

  def call(self, x):
    result = self.add([-x,self.bias])
    return result

  def compute_output_shape(self, input_shape):
    return input_shape[0]

class MLP(tf.keras.Sequential):
  """ Simple MLP model. """
  def __init__(self,
               output_units,
               num_layers=3,
               hidden_units=64,
               activation=tf.nn.tanh):
    super().__init__([
        tf.keras.layers.Dense(
            units=hidden_units if i < num_layers else output_units,
            activation=activation if i < num_layers else None,
            kernel_initializer=(
                tf.initializers.orthogonal(sqrt(2) if i < num_layers else 1)),
            bias_initializer=tf.initializers.zeros(),
        ) for i in range(1, num_layers + 1)
    ])


class ODEModel(tf.keras.Model):
  """ ODE model that wraps state, dynamics and output models. """
  def __init__(self, state, dynamics, outputs,
               time=(0., 1.), rtol=1e-3, atol=1e-3):
    super().__init__()
    self.state = state
    self.dynamics = dynamics
    self.outputs = outputs
    self.time = tf.cast(tf.convert_to_tensor(time), tf.float32)
    self.odeint = partial(odeint, rtol=rtol, atol=atol)

  def call(self, inputs, training=True, mask=None):
    _ = training, mask

    def dynamics(inputs, time):
      time = tf.cast([[time]], tf.float32)
      inputs = tf.concat([inputs, tf.tile(time, [inputs.shape[0], 1])], -1)
      return self.dynamics(inputs)

    state = self.state(inputs)
    hidden = self.odeint(dynamics, state, self.time)[-1]
    out = self.outputs(hidden)
    return out


class ODEMLP(ODEModel):
  """ Basic MLP model with ode. """
  def __init__(self, output_units, hidden_units=64,
               num_state_layers=1, num_dynamics_layers=1, num_output_layers=1,
               time=(0., 1.), rtol=1e-3, atol=1e-3):

    def make_sequential(num_layers, **layer_kws):
      return tf.keras.Sequential(
          [tf.keras.layers.Dense(**layer_kws) for _ in range(num_layers)])

    layer_kws = dict(
        units=hidden_units,
        activation=tf.nn.tanh,
        kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
        bias_initializer=tf.initializers.zeros())

    state = make_sequential(num_state_layers, **layer_kws)
    dynamics = make_sequential(num_dynamics_layers, **layer_kws)

    layer_kws.update(units=output_units, activation=None,
                     kernel_initializer=tf.initializers.orthogonal(1))
    output = make_sequential(num_output_layers, **layer_kws)
    super().__init__(state, dynamics, output, time=time, rtol=rtol, atol=atol)

class CTRNN(ODEModel):
  """ Basic CT-RNN model using the ode wrapper. """
  def __init__(self, output_units, hidden_units=64,
               num_state_layers=1, num_dynamics_layers=1, num_output_layers=1,
               time=(0., 1.), rtol=1e-3, atol=1e-3, tau = 0.9):

    def make_sequential(num_layers, **layer_kws):
      return tf.keras.Sequential(
          [tf.keras.layers.Dense(**layer_kws) for _ in range(num_layers)])

    layer_kws = dict(
        units=hidden_units,
        activation=tf.nn.tanh,
        kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
        bias_initializer=tf.initializers.zeros())

    state = make_sequential(num_state_layers, **layer_kws)
    dynamics = make_sequential(num_dynamics_layers, **layer_kws)

    layer_kws.update(units=output_units, activation=None,
                     kernel_initializer=tf.initializers.orthogonal(1))
    output = make_sequential(num_output_layers, **layer_kws)
    
    super().__init__(state, dynamics, output, time=time, rtol=rtol, atol=atol)
    
    self.leakyComponent = LeakyComponent()
    
  # implements a leaky integrator
  def call(self, inputs, training=True, mask=None):
    _ = training, mask

    def dynamics(inputs, time):
      time = tf.cast([[time]], tf.float32)
      inputs_padded = tf.concat([inputs, tf.tile(time, [inputs.shape[0], 1])], -1)
      return tf.keras.layers.Add()([self.dynamics(inputs_padded),self.leakyComponent(inputs)])  # TODO : implement learnable tau vector

    state = self.state(inputs)
    hidden = self.odeint(dynamics, state, self.time)[-1]
    out = self.outputs(hidden)
    return out

class LTC(ODEModel):
  """ Basic LTC model using the ode wrapper. """
  
  def __init__(self, output_units, hidden_units=64,
               num_state_layers=1, num_dynamics_layers=1, num_output_layers=1,
               time=(0., 1.), rtol=1e-3, atol=1e-3, tau = 0.9):


    def make_sequential(num_layers, **layer_kws):
      return tf.keras.Sequential(
          [tf.keras.layers.Dense(**layer_kws) for _ in range(num_layers)]) 

    layer_kws = dict(
        units=hidden_units,
        activation=tf.nn.tanh,
        kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
        bias_initializer=tf.initializers.zeros())

    state = make_sequential(num_state_layers, **layer_kws)
    dynamics = make_sequential(num_dynamics_layers, **layer_kws)

    layer_kws.update(units=output_units, activation=None,
                     kernel_initializer=tf.initializers.orthogonal(1))
    output = make_sequential(num_output_layers, **layer_kws)
    
    super().__init__(state, dynamics, output, time=time, rtol=rtol, atol=atol)
    
    self.add = tf.keras.layers.Add()
    self.mult = tf.keras.layers.Multiply()
    self.leakyComponent = LeakyComponent()
    self.conductance = ConductanceConnectivity()
    
  # implements a leaky integrator
  def call(self, inputs, training=True, mask=None):
  
    _ = training, mask

    def dynamics(inputs, time):
      time = tf.cast([[time]], tf.float32)
      non_linearity = self.dynamics(tf.concat([inputs, tf.tile(time, [inputs.shape[0], 1])], -1))
      non_linearity = self.mult([non_linearity,self.conductance(inputs)])
      return self.add([non_linearity,self.leakyComponent(inputs)])

    state = self.state(inputs)
    hidden = self.odeint(dynamics, state, self.time)[-1]
    out = self.outputs(hidden)
    return out


class ContinuousActorCriticModel(tf.keras.Model):
  """ Adds variance variable to policy and value models to create new model. """
  def __init__(self, input_shape, action_dim, policy, value, logstd=None):
    super().__init__()
    self.input_tensor = tf.keras.layers.Input(input_shape)
    self.policy = policy
    self.value = value
    if logstd is not None:
      if tf.shape(logstd) != [action_dim]:
        raise ValueError(f"logstd has wrong shape {tf.shape(logstd)}, ",
                         f"expected 1-d tensor of size action_dim={action_dim}")
      self.logstd = logstd
    else:
      self.logstd = tf.Variable(tf.zeros(action_dim), trainable=True,
                                name="logstd")

  @property
  def input(self):
    return self.input_tensor

  def call(self, inputs, training=True, mask=None):
    _ = training, mask
    inputs = tf.cast(inputs, tf.float32)
    batch_size = tf.shape(inputs)[0]
    logstd = tf.tile(self.logstd[None], [batch_size, 1])
    return self.policy(inputs), tf.exp(logstd), self.value(inputs)
