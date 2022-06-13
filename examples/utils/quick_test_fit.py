import keras_nlp
import tensorflow as tf
from tensorflow import keras

x, y = tf.random.uniform(shape=[2, 5]), tf.constant([1, 0])

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu="chenmoney-nimei", project="keras-team-gcp", zone="us-east1-d", 
    # coordinator_address="10.142.0.38:8470", coordinator_name="coordinator",
)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
#     tpu="chenmoney-nimei", project="keras-team-gcp", zone="us-east1-d",
# )
strategy = tf.distribute.TPUStrategy(resolver)

dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2).repeat()
dist_dataset = strategy.experimental_distribute_dataset(dataset)

with strategy.scope():
  # Create a tiny transformer.
  inputs = keras.Input(shape=(5,), dtype=tf.float32)
  outputs = keras.layers.Dense(1, activation="sigmoid")(inputs)
  model = keras.Model(inputs, outputs)
  model.compile(optimizer="adam", loss="binary_crossentropy")

model.fit(dist_dataset, steps_per_epoch=5)

