import tensorflow as tf
import json
import time
import numpy as np
import data_utils
from keras.layers import LSTM, Dense, Input
from keras.models import Model


settings_path = '/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/settings/gan_train.txt'
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


samples,labels = data_utils.process_train_data(num_signal,seq_length,seq_step)
noise = data_utils.seeder(num_signal,seq_length,seq_step)

test_sample,test_labels, test_index = data_utils.process_test_data(num_signal,seq_length,seq_step)

generator_optimizer = tf.keras.optimizers.Adam(.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(.05)
seed = tf.random.normal([batch_size, seq_length, latent_dim])
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def create_generator():
    inputs = Input(batch_input_shape=(None,seq_length, latent_dim))
    x = LSTM(128, return_sequences=True)(inputs)
    x = LSTM(64, activation='tanh',return_sequences=True)(x)
    outputs = Dense(num_signal, activation='tanh')(x)
    generator = Model(inputs, outputs)
    return generator

def create_discriminator():
    discriminator_input = Input(batch_input_shape=(None,seq_length, num_signal))
    x = LSTM(128, return_sequences=True, activation='relu')(discriminator_input)
    x = LSTM(64, return_sequences=True,activation='relu')(x)
    x = LSTM(32, return_sequences=True,activation='relu')(x)
    x = Dense(10, activation='sigmoid')(x)
    discriminator_output = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, discriminator_output)
    return discriminator

discriminator = create_discriminator()
generator = create_generator()

discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.005), loss='binary_crossentropy')
generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0001), loss='binary_crossentropy')
# discriminator.trainable = False

def D_loss(real_data, fake_data):
    real_loss = tf.math.reduce_mean(cross_entropy(tf.zeros_like(real_data), real_data))
    fake_loss = tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_data), fake_data))
    total_loss = real_loss + fake_loss
    return total_loss
  
def G_loss(fake_data):
    return tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_data), fake_data))

def train_step():
    total_batch = int(samples.shape[0] / batch_size)
    total_d_loss = 0
    total_g_loss = 0
    call_optimizer_count = 0
    temp_g_loss = []
    temp_d_loss = []
    for batch_idx in range(total_batch):
        batch_data = tf.convert_to_tensor((data_utils.get_batch(samples,batch_size,batch_idx)),dtype=tf.float32)
        batch_noise = tf.convert_to_tensor((data_utils.get_batch(noise,batch_size,0)),dtype=tf.float32)
        call_optimizer_count += 1
        # batch_noise = tf.random.normal([batch_size,seq_length,num_signal])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(seed, training=True)

            real_input_predictions = discriminator(batch_data, training=True)
            generated_input_predictions = discriminator(generated_data, training=True)
    
            MSE = tf.keras.losses.mean_squared_error(batch_data, generated_data)
            print('MSE after batch lot {}: {}'.format(batch_idx, (np.mean(MSE))))
            
            gen_loss = G_loss(generated_input_predictions)
            disc_loss = D_loss(real_input_predictions, generated_input_predictions)

            temp_g_loss.append(gen_loss)
            temp_d_loss.append(disc_loss)

            if call_optimizer_count > optimizer_call_threshold :
                print('\n**********\tOptimizing Model\t**********\n')
                temp_avg_g_loss = tf.math.reduce_mean(temp_g_loss)
                temp_avg_d_loss = tf.math.reduce_mean(temp_d_loss)

                gradients_of_generator = gen_tape.gradient(temp_avg_g_loss, generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(temp_avg_d_loss, discriminator.trainable_variables)
                
                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

                call_optimizer_count = 0
                temp_g_loss = []
                temp_d_loss = []

            total_d_loss = total_d_loss + np.mean(disc_loss)
            total_g_loss = total_g_loss + np.mean(gen_loss)
    print ('Discriminator Loss: {} \tGenerator Loss: {}'.format((total_d_loss/total_batch), (total_g_loss/total_batch)))

def train_GAN(epochs):
    for epoch in range(epochs):
        start = time.time()
        print('Epoch {} started'.format(epoch))
        print('====================================================================================================================')
        train_step()
        generate_result(epoch)
        print ('Epoch Finished. Time for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
            
    #save trained models
    d_path = os.path.join('/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/saved_model/discriminators/model_seq_' + str(seq_length) + '_' + settings['exp'] + '/')
    g_path = os.path.join('/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/saved_model/generators/model_seq_' + str(seq_length) + '_' + settings['exp'] + '/')
    tf.saved_model.save(discriminator, d_path)
    tf.saved_model.save(generator, g_path)

def get_final_prediction(batch):
    predictions = discriminator(batch)
    return predictions

def generate_result(epoch):
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

        # np.save('./predictions/prediction_sequence_seq_length_'+ str(settings['seq_length'])+ '_' + str(epoch) ,D_test)
        # np.save('./predictions/real_sequence_seq_length_'+ str(settings['seq_length'])+ '_' + str(epoch) ,R_labels)

        results = np.zeros([6, 4])
        for i in range(2, 8):
            tao = 0.1 * i
            Accu, Pre, Rec, F1 = data_utils.get_evaluation(D_test, R_labels, I_mb, seq_step, tao)
            print('seq_length:', seq_length)
            print('Comb-logits-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}\n'
                  .format(epoch, tao, Accu, Pre, Rec, F1))
            results[i-2 , :] = [Accu, Pre, Rec, F1]
        return results

train_GAN(num_epoch)