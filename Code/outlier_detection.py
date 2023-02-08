import tensorflow as tf
from tensorflow import keras
from keras import layers
import json
import time
import numpy as np
import data_utils
from sklearn.metrics import precision_recall_fscore_support

tf.config.experimental_run_functions_eagerly(True)

settings_path = '/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/settings/outlier_detection.txt'
settings = json.load(open(settings_path, 'r'))


batch_size = settings['batch_size']
seq_length = settings['seq_length']
latent_dim = settings['latent_dim']
num_epoch = settings['num_epoch']
num_signal = settings['num_pc_dimention']
noise_dim = settings['noise_dim']
num_of_generated_examples = settings['num_of_generated_examples']
seq_step = settings['seq_step']

samples,labels, index = data_utils.process_test_data(num_signal,seq_length,seq_step)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
seed = tf.random.normal([batch_size, seq_length, latent_dim])
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
d_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# load saved generator and discriminators
loaded_discriminator = tf.saved_model.load('/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/saved_model/discriminators/model_seq_' + str(seq_length)+ '_' + settings['exp'] + '/')
discriminator = loaded_discriminator.signatures["serving_default"]

loaded_generator = tf.saved_model.load('/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/saved_model/generators/model_seq_' + str(seq_length) + '_' + settings['exp'] + '/')
generator = loaded_generator.signatures["serving_default"]

def D_loss(real_data,fake_data):
    real_loss = cross_entropy(tf.zeros_like(real_data), real_data)
    fake_loss = cross_entropy(tf.ones_like(fake_data), fake_data)
    total_loss = real_loss + fake_loss
    return total_loss
  
def G_loss(fake_data):
    return tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_data), fake_data))

def train_step():
    total_d_loss = 0
    total_g_loss = 0
    total_batch = int(samples.shape[0] / batch_size)
    for batch_idx in range(total_batch):
        batch_data = tf.convert_to_tensor((data_utils.get_batch(samples,batch_size,batch_idx)),dtype=tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            g_output = generator(input_2=batch_data)
            g_predictions = g_output["dense_2"]
            g_output = tf.convert_to_tensor(g_predictions, dtype=tf.float32)
            
            real_output = discriminator(input_1=batch_data)
            d_predictions = real_output["dense_1"]
            real_output = tf.convert_to_tensor(d_predictions, dtype=tf.float32)
            
            fake_output = discriminator(input_1=g_output)
            d_predictions_fake = fake_output["dense_1"]
            fake_output = tf.convert_to_tensor(d_predictions_fake, dtype=tf.float32)
            
            gen_loss = G_loss(g_output)
            disc_loss = D_loss(real_output,fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            total_d_loss = total_d_loss + np.mean(disc_loss)
            total_g_loss = total_g_loss + np.mean(gen_loss)
    print ('Discriminator Loss: {} \tGenerator Loss: {}'.format((total_d_loss/total_batch), (total_g_loss/total_batch)))

def train_GAN(epochs):
    for epoch in range(epochs):
        start = time.time()
        print('------------------------------------------------------------------------------')
        print('Epoch {} started'.format(epoch))
        train_step()
        print ('Epoch Finished. Time for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    generate_result(epoch)


def get_label(batch):
    batch = tf.convert_to_tensor(batch, dtype=tf.float32)
    label = discriminator(input_1=batch)
    d_predictions = label["dense_1"]
    return d_predictions

def generate_result(epoch):
        num_samples_t = samples.shape[0]
        D_test = np.empty([num_samples_t, seq_length, 1])
        R_labels = np.empty([num_samples_t, seq_length, 1])
        I_mb = np.empty([num_samples_t, seq_length, 1])
        batch_times = int(num_samples_t / batch_size)

        for batch_idx in range(0, batch_times):
            start_pos = batch_idx * batch_size
            end_pos = start_pos + batch_size

            batch = samples[start_pos:end_pos, :, :]
            T_labels = labels[start_pos:end_pos, :, :]
            I_mmb = index[start_pos:end_pos, :, :]

            P_labels = get_label(batch)
            D_test[start_pos:end_pos, :, :] = P_labels
            R_labels[start_pos:end_pos, :, :] = T_labels
            I_mb[start_pos:end_pos, :, :] = I_mmb

        # Left over sample from batch iteration
        start_pos = (batch_times) * batch_size
        end_pos = start_pos + batch_size

        size = samples[start_pos:end_pos, :, :].shape[0]
        fill = np.ones([batch_size - size, samples.shape[1], samples.shape[2]])
        batch = np.concatenate([samples[start_pos:end_pos, :, :], fill], axis=0)

        P_labels = get_label(batch)
        D_test[start_pos:end_pos, :, :] = P_labels[:size, :, :]
        R_labels[start_pos:end_pos, :, :] = T_labels[:size, :, :]

        # np.save('./predictions/prediction_sequence_seq_length_'+ str(settings['seq_length'])+ '_' + str(epoch) ,D_test)
        # np.save('./predictions/real_sequence_seq_length_'+ str(settings['seq_length'])+ '_' + str(epoch) ,R_labels)

        results = np.zeros([6, 4])
        for i in range(2, 8):
            tao = 0.1 * i
            Accu, Pre, Rec, F1 = get_evaluation(D_test, R_labels, I_mb, seq_step, tao)
            print('seq_length:', seq_length)
            print('Comb-logits-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}'
                  .format(epoch, tao, Accu, Pre, Rec, F1))
            results[i-2 , :] = [Accu, Pre, Rec, F1]
        return results

def get_evaluation(Label_test, Label_real, I_mb, seq_step, tao):
    aa = Label_test.shape[0]
    bb = Label_test.shape[1]
    
    LL = (aa-1)*seq_step+bb
    

    Label_test = abs(Label_test.reshape([aa, bb]))
    Label_real = Label_real .reshape([aa, bb])
    I_mb = I_mb .reshape([aa, bb])

    D_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])

    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L[i*seq_step+j] += Label_test[i, j]
            L_L[i * seq_step + j] += Label_real[i, j]
            Count[i * seq_step + j] += 1

    D_L /= Count
    L_L /= Count
    
    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(LL):
        if D_L[i] < tao:
            D_L[i] = 1
        else:
            D_L[i] = 0

    cc = (D_L == L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    L_L = L_L.astype(bool)

    Accu = float((N / LL) * 100)
    precision, recall, f1, _ = precision_recall_fscore_support(L_L, D_L, average='binary')

    return Accu, precision, recall, f1,

train_GAN(num_epoch)



    