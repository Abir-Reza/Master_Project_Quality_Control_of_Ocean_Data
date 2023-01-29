import tensorflow as tf
from tensorflow import keras
from keras import layers
import json
import time
import numpy as np
import data_utils

tf.config.experimental_run_functions_eagerly(True)

settings_path = '/Users\macio\Desktop\MAD-GAN_migrated\our_code\settings\outlier_detection.txt'
settings = json.load(open(settings_path, 'r'))


batch_size = settings['batch_size']
seq_length = settings['seq_length']
latent_dim = settings['latent_dim']
num_epoch = settings['num_epoch']
num_signal = settings['num_pc_dimention']
noise_dim = settings['noise_dim']
num_of_generated_examples = settings['num_of_generated_examples']
seq_step = settings['seq_step']

samples,labels = data_utils.process_test_data(num_signal,seq_length,seq_step)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
seed = tf.random.normal([batch_size, seq_length, latent_dim])
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
d_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)


# Define the generator

from keras.layers import LSTM, Dense, Input
from keras.models import Model

latent_dim = 100

loaded_discriminator = tf.saved_model.load(settings['saved_model_path'] + 'discriminators\model_seq_' + str(seq_length) + settings['exp'] + '/')
discriminator = loaded_discriminator.signatures["serving_default"]

# generator = create_generator()

# discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3), loss='binary_crossentropy')
# generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3), loss='binary_crossentropy')

def D_loss(real_data):
    real_loss = cross_entropy(tf.zeros_like(real_data), real_data)
    return real_loss
  
def G_loss(fake_data):
    return tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_data), fake_data))

@tf.function
def train_step(batch_data):
    noise = tf.random.normal([batch_size, seq_length, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generated_data = generator(noise, training=True)
        batch_data = tf.convert_to_tensor(batch_data, dtype=tf.float32)
        real_output = discriminator(input_1=batch_data)
        predictions = real_output["dense_1"]
        real_output = tf.convert_to_tensor(predictions, dtype=tf.float32)
        
        # fake_output = discriminator(generated_data, training=True)
        
        # gen_loss = G_loss(fake_output)
        disc_loss = D_loss(real_output)

        print('Discriminator loss: ',disc_loss.numpy())
        # gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train_GAN(epochs):
    for epoch in range(epochs):
        start = time.time()
        for batch_idx in range(int(samples.shape[0])):
            train_batch = data_utils.get_batch(samples,batch_size,batch_idx)
            train_step(train_batch)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
train_GAN(num_epoch)

    