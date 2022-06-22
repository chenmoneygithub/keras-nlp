# import keras_nlp
# import tensorflow as tf
# from tensorflow import keras

# x, y = tf.random.uniform(shape=[2, 5]), tf.constant([1, 0])

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
#     tpu="chenmoney-nimei", project="keras-team-gcp", zone="us-east1-d",
# )
# strategy = tf.distribute.TPUStrategy(resolver)

# dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2).repeat()
# dist_dataset = strategy.experimental_distribute_dataset(dataset)

# with strategy.scope():
#   # Create a tiny transformer.
#   inputs = keras.Input(shape=(5,), dtype=tf.float32)
#   outputs = keras.layers.Dense(1, activation="sigmoid")(inputs)
#   model = keras.Model(inputs, outputs)
#   optimizer = tf.keras.optimizers.experimental.Adam()
#   loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

# @tf.function
# def train_step(iterator):
#   """The step function for one training step."""

#   def step_fn(inputs):
#     """The computation to run on each TPU device."""
#     x, y = inputs
#     with tf.GradientTape() as tape:
#       pred = model(x, training=True)
#       loss = loss_fn(y, pred)
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

#   strategy.run(step_fn, args=(next(iterator),)) 

# train_iterator = iter(dist_dataset)
# for epoch in range(5):
#   print('Epoch: {}/5'.format(epoch))

#   for step in range(5):
#     train_step(train_iterator)
#   print('Current step: {}%'.format(
#       optimizer.iterations.numpy()))

import tensorflow as tf
print("Tensorflow version " + tf.__version__)

@tf.function
def add_fn(x,y):
  z = x + y
  return z

resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
    tpu="chenmoney-geez-pod", project="keras-team-gcp", zone="us-east1-d",
)
strategy = tf.distribute.TPUStrategy(resolver)

x = tf.constant(1.)
y = tf.constant(1.)
z = strategy.run(add_fn, args=(x,y))
print(z)