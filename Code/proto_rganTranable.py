import tensorflow as tf
import json
import time
import numpy as np
import data_utils
from keras.layers import LSTM, Dense, Input, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import regularizers
import os

settings_path = '/Users\macio\Desktop\MAD-GAN_migrated\our_code\settings\gan_train.txt'
settings = json.load(open(settings_path, 'r'))

batch_size = settings['batch_size']
seq_length = settings['seq_length']
latent_dim = settings['latent_dim']
num_epoch = settings['num_epoch']
num_signal = settings['num_pc_dimention']
num_of_generated_examples = settings['num_of_generated_examples']
seq_step = settings['seq_step']
proximity = settings['proximity']
optimizer_call_threshold = settings['optimizer_call_threshold']
d_path = os.path.join('/Users\macio\Desktop\MAD-GAN_migrated\our_code\saved_model\discriminators\model_seq_' + str(seq_length) + '_' + settings['exp'] + '/')


samples,labels = data_utils.process_train_data(num_signal,seq_length,seq_step)

test_sample,test_labels, test_index = data_utils.process_test_data(num_signal,seq_length,seq_step)

generator_optimizer = tf.keras.optimizers.Adam(.05)
discriminator_optimizer = tf.keras.optimizers.Adam(.01)
seed = tf.random.normal([batch_size, seq_length, latent_dim],mean= -15.75,
    stddev=7,
    dtype=tf.dtypes.float32,
    seed=50)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def create_generator():
    inputs = Input(batch_input_shape=(None,seq_length, latent_dim))
    x = LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.0057))(inputs) 
    x = Dropout(0.33)(x)
    x = LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l1(0.76))(inputs) 
    x = Dropout(0.25)(x)
    x = LSTM(16, activation='tanh',return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(x)
    outputs = Dense(num_signal, activation='sigmoid')(x)
    generator = Model(inputs, outputs)
    return generator

def create_discriminator():
    discriminator_input = Input(batch_input_shape=(None,seq_length, num_signal))
    x = LSTM(128, return_sequences=True, activation='relu')(discriminator_input)
    x = Dropout(0.15)(x)
    x = LSTM(64, return_sequences=True,activation='relu')(x)
    x = BatchNormalization()(x)
    # x = Dense(16, activation='relu')(x)
    discriminator_output = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, discriminator_output)
    return discriminator

discriminator = create_discriminator()
generator = create_generator()

discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.005), loss='binary_crossentropy')
generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0020), loss='binary_crossentropy')

def D_loss(real_data, fake_data, batch_data):
    # real_data = tf.clip_by_value(real_data, 1e-4, 1 - 1e-4)
    # fake_data = tf.clip_by_value(fake_data, 1e-4, 1 - 1e-3)
    real_loss = tf.math.reduce_mean(cross_entropy(tf.ones_like(real_data), real_data))
    fake_loss = tf.math.reduce_mean(cross_entropy(tf.zeros_like(fake_data), fake_data))
    total_loss = real_loss + fake_loss
    print('\nReal loss: ---',real_loss)
    print('Fake loss: ---',fake_loss,'\n')
    return total_loss

discriminator.trainable = False
  
def G_loss(fake_data,real_data):
    g_loss = tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_data), fake_data))
    clipped_loss = tf.clip_by_value(g_loss, 1e-4, 1e5)
    return clipped_loss

def train_step():
    total_batch = int(samples.shape[0] / batch_size)
    total_d_loss = 0
    total_g_loss = 0
    call_optimizer_g_count = 0
    call_optimizer_d_count = 0
    for batch_idx in range(total_batch):
        batch_data = tf.convert_to_tensor((data_utils.get_batch(samples,batch_size,batch_idx)),dtype=tf.float32)
        call_optimizer_d_count += 1
        call_optimizer_g_count += 1
        #batch_noise = tf.random.normal([batch_size,seq_length,num_signal])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(seed, training=True)
            generated_input_predictions = discriminator(generated_data, training=True)
            gen_loss = G_loss(generated_input_predictions,batch_data)

            generator.trainable = True
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            clipped_grads_of_generator = [tf.clip_by_value(grad, -0.01, 200) for grad in gradients_of_generator]
            generator_optimizer.apply_gradients(zip(clipped_grads_of_generator, generator.trainable_variables))
            generator.trainable = False


            real_input_predictions = discriminator(batch_data, training=True)            
            disc_loss = D_loss(real_input_predictions, generated_input_predictions, batch_data)

            discriminator.trainable = True
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            clipped_grads_of_discriminator = [tf.clip_by_value(grad, 0.005, 100) for grad in gradients_of_discriminator]
            discriminator_optimizer.apply_gradients(zip(clipped_grads_of_discriminator, discriminator.trainable_variables))
            discriminator.trainable = False

            g_grads = []
            for grad in clipped_grads_of_discriminator:
                g_grads.append(np.mean(grad))
            print('Dicriminator gradients: ', np.mean(g_grads))

            total_d_loss = total_d_loss + np.mean(disc_loss)
            total_g_loss = total_g_loss + np.mean(gen_loss)
        # print('Generator Output: \t',generated_data[0][0])
    print ('Discriminator Loss: {} \tGenerator Loss: {}'.format((total_d_loss/total_batch), (total_g_loss/total_batch)))
    return (total_d_loss/total_batch)

def train_GAN(epochs):
    loss_tracker = []
    for epoch in range(epochs):
        start = time.time()
        print('Epoch {} started'.format(epoch))
        print('====================================================================================================================')
        avg_d_loss = train_step()
        loss_tracker.append(avg_d_loss)
        if(len(loss_tracker)>4):
            avg = np.average(loss_tracker)
            diff = abs(avg_d_loss - avg)
            if diff < 0.000000000001:
                print('BREAKING EPOCH LOOP!!!!!VANISHING GRADIENT DETECTED')
                break
            else:
               loss_tracker.pop(0)
        print ('Epoch Finished. Time for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    #save trained models
    # g_path = os.path.join('/Users\macio\Desktop\MAD-GAN_migrated\our_code\saved_model\generators\model_seq_' + str(seq_length) + '_' + settings['exp'] + '/')
    tf.saved_model.save(discriminator, d_path)
    # tf.saved_model.save(generator, g_path)


def get_final_prediction(batch):
    loaded_discriminator = tf.saved_model.load(d_path)
    discriminator = loaded_discriminator.signatures["serving_default"]
    batch_data = tf.convert_to_tensor(batch ,dtype=tf.float32)
    predictions = discriminator(input_1=batch_data)
    predictions = predictions["dense_1"]
    return predictions

def generate_result():
        num_samples_t = test_sample.shape[0]
        D_test = np.empty([num_samples_t, seq_length, 1])
        R_labels = np.empty([num_samples_t, seq_length, 1])
        I_mb = np.empty([num_samples_t, seq_length, 1])
        batch_times = int(num_samples_t / batch_size)

        for batch_idx in range(0, batch_times):
            start_pos = batch_idx * batch_size
            end_pos = start_pos + batch_size

            batch = test_sample[start_pos:end_pos, :, :]
            T_labels = test_labels[start_pos:end_pos, :, :]
            I_mmb = test_index[start_pos:end_pos, :, :]

            P_labels = get_final_prediction(batch)
            D_test[start_pos:end_pos, :, :] = P_labels
            R_labels[start_pos:end_pos, :, :] = T_labels
            I_mb[start_pos:end_pos, :, :] = I_mmb

        # Left over sample from batch iteration
        start_pos = (batch_times) * batch_size
        end_pos = start_pos + batch_size

        size = test_sample[start_pos:end_pos, :, :].shape[0]
        fill = np.ones([batch_size - size, samples.shape[1], samples.shape[2]])
        batch = np.concatenate([samples[start_pos:end_pos, :, :], fill], axis=0)

        P_labels = get_final_prediction(batch)
        T_labels = test_labels[start_pos:end_pos, :, :]
        I_mmb = test_index[start_pos:end_pos, :, :]

        D_test[start_pos:end_pos, :, :] = P_labels[:size, :, :]
        R_labels[start_pos:end_pos, :, :] = T_labels
        I_mb[start_pos:end_pos, :, :] = I_mmb

        np.save('./predictions/prediction_sequence_seq_length_'+ str(settings['seq_length'])+ '_' + str(1) ,D_test)
        np.save('./predictions/real_sequence_seq_length_'+ str(settings['seq_length'])+ '_' + str(1) ,R_labels)

        results = np.zeros([10, 4])
        org_shape = data_utils.de_shape(D_test, R_labels, I_mb, seq_step)
        tao_min = np.min(org_shape) - 0.03
        for i in range(1, 10):
            tao = float(tao_min + (0.01*i))
            Accu, Pre, Rec, F1 = data_utils.get_evaluation(D_test, R_labels, I_mb, seq_step, tao)
            print('Final Evaluation: tao={:.2}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}\n'
                  .format(tao, Accu, Pre, Rec, F1))
            results[i-1 , :] = [Accu, Pre, Rec, F1]
        return results

train_GAN(num_epoch)
generate_result()

    