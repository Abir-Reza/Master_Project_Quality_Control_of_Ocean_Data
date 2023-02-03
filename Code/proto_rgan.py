import tensorflow as tf
from tensorflow import keras
from keras import layers
import json
import time
import numpy as np
import data_utils

tf.config.experimental_run_functions_eagerly(True)

settings_path = '/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/settings/gan_train.txt'
settings = json.load(open(settings_path, 'r'))


batch_size = settings['batch_size']
seq_length = settings['seq_length']
latent_dim = settings['latent_dim']
num_epoch = settings['num_epoch']
num_signal = settings['num_pc_dimention']
noise_dim = settings['noise_dim']
num_of_generated_examples = settings['num_of_generated_examples']
seq_step = settings['seq_step']

samples,labels = data_utils.process_train_data(num_signal,seq_length,seq_step)

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

# Define the generator
def create_generator():
    generator_input = Input(batch_input_shape=(None,seq_length, latent_dim))
    x = LSTM(256, return_sequences=True)(generator_input)
    generator_output = Dense(num_signal)(x)
    generator = Model(generator_input, generator_output)
    return generator

def create_discriminator():
    discriminator_input = Input(batch_input_shape=(None,seq_length, num_signal))
    x = LSTM(256, return_sequences=True)(discriminator_input)
    x = LSTM(256)(x)
    x = Dense(256)(x)
    discriminator_output = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, discriminator_output)
    return discriminator

discriminator = create_discriminator()
generator = create_generator()

discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3), loss='binary_crossentropy')
generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3), loss='binary_crossentropy')
discriminator.trainable = False

def D_loss(real_data, fake_data):
    real_loss = tf.math.reduce_mean(cross_entropy(tf.zeros_like(real_data), real_data))
    fake_loss = tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_data), fake_data))
    total_loss = real_loss + fake_loss
    return total_loss
  
def G_loss(fake_data):
    return tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_data), fake_data))

@tf.function
def train_step(batch_data):
    noise = tf.random.normal([batch_size, seq_length, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)

        real_output = discriminator(batch_data, training=True)
        fake_output = discriminator(generated_data, training=True)
        
        gen_loss = G_loss(fake_output)
        disc_loss = D_loss(real_output, fake_output)

        print('Generator loss: ',gen_loss.numpy(),' Discriminator loss: ',disc_loss.numpy())
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train_GAN(epochs):
    for epoch in range(epochs):
        start = time.time()
        for batch_idx in range(int(len(samples) / batch_size)):
            train_batch = data_utils.get_batch(samples,batch_size,batch_idx)
            train_step(train_batch)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    tf.saved_model.save(discriminator, '/Users\macio\Desktop\MAD-GAN_migrated\our_code\saved_model\discriminators\model_seq_' + str(seq_length) + settings['exp'] + '/')
    tf.saved_model.save(generator, '/Users\macio\Desktop\MAD-GAN_migrated\our_code\saved_model\generators\model_seq_' + str(seq_length) + settings['exp'] + '/')

    
train_GAN(num_epoch)

    