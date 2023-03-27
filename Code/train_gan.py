import tensorflow as tf
import keras.backend as K
import json
import time
import numpy as np
import data_utils
from keras.layers import LSTM, Dense, Input, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import regularizers
import os
# from keras.regularizers import l2,l1
from matplotlib import pyplot as plt
import random

# settings_path = '/Users\macio\Desktop\MAD-GAN_migrated\our_code\settings\gan_train.txt'
# settings_path = '/home/abir/Study/Winter22_23/Project/TEST/gan_train.txt'
settings_path = '/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/settings/gan_train.txt'
settings = json.load(open(settings_path, 'r'))

history_random_g_label = []
history_augment_g_label = []
history_real_d_label = []
history_augment_d_label = []
history_d_loss = []
history_g_loss = []
history_g_grads = []
history_d_grads = []

batch_size = settings['batch_size']
seq_length = settings['seq_length']
latent_dim = settings['latent_dim']
num_epoch = settings['num_epoch']
num_signal = settings['num_pc_dimention']
num_of_generated_examples = settings['num_of_generated_examples']
seq_step = settings['seq_step']
proximity = settings['proximity']
optimizer_call_threshold = settings['optimizer_call_threshold']
# d_path = os.path.join('/home/abir/Study/Winter22_23/Project/TEST/model_seq_' + str(seq_length) + '_' + settings['exp'] + '/')
d_path = os.path.join('/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/saved_model/discriminators/model_seq_' + str(seq_length) + '_' + settings['exp'] +'100' + '/')


samples,labels = data_utils.process_train_data(num_signal,seq_length,seq_step)

augmented_samples, augmented_labels = data_utils.process_train_data(num_signal,seq_length,seq_step,True)

test_sample,test_labels, test_index = data_utils.process_test_data(num_signal,seq_length,seq_step)

generator_optimizer = tf.keras.optimizers.Adam(.05)
discriminator_optimizer = tf.keras.optimizers.Adam(.01)

noise = tf.random.normal([batch_size, seq_length, latent_dim],mean= -4,
    stddev=5,
    dtype=tf.dtypes.float32,
    seed=8)

def get_seed(std,mn,s):
    seed = tf.random.normal([batch_size, seq_length, latent_dim],mean= mn,
    stddev=std,
    dtype=tf.dtypes.float32,
    seed=s)
    return seed
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
catagorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False) 
mean_squared_error = tf.keras.losses.MeanSquaredError()


def create_generator():
    inputs = Input(batch_input_shape=(None,seq_length, latent_dim))
    x = LSTM(32, return_sequences=True)(inputs)
    x = LeakyReLU(alpha=0.4)(x)
    x = LSTM(16, return_sequences=True)(x)
    x = LeakyReLU(alpha=0.4)(x)
    x = LSTM(8, return_sequences=True)(x)
    x = LeakyReLU(alpha=0.4)(x)
    x = LSTM(4, return_sequences=True)(x)
    x = LeakyReLU(alpha=0.4)(x)
    x = Dense(32,activation='leaky_relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(12,activation='leaky_relu')(x)
    outputs = Dense(num_signal, activation='leaky_relu')(x)
    generator = Model(inputs, outputs)
    return generator

def create_discriminator():
    discriminator_input = Input(batch_input_shape=(None,seq_length, num_signal))
    x = LSTM(64, return_sequences=True)(discriminator_input)
    x = LeakyReLU(alpha=0.4)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = LeakyReLU(alpha=0.4)(x)
    x = LSTM(16, return_sequences=True)(x)
    x = LeakyReLU(alpha=0.4)(x)
    x = LSTM(8, return_sequences=True)(x)
    x = LeakyReLU(alpha=0.4)(x)
    x = Dense(4,activation='leaky_relu')(x)
    discriminator_output = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, discriminator_output)
    return discriminator

discriminator = create_discriminator()
generator = create_generator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.06,epsilon=1), loss='binary_crossentropy')
generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.04,epsilon=1), loss='binary_crossentropy')


def D_loss(real_data, fake_data, augment_label):
    real_data_loss = tf.math.reduce_mean(cross_entropy(tf.zeros_like(real_data), real_data))
    fake_data_loss = 5*(tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_data), fake_data)))
    augment_loss = tf.math.reduce_mean(cross_entropy(real_data, augment_label))

    total_loss = real_data_loss + fake_data_loss + augment_loss
    print('6. Real_data_loss',real_data_loss)
    print('\t7. D Augment loss',augment_loss)
    print('6.Discriminator loss:', total_loss )
    return total_loss

discriminator.trainable = False
  
def G_loss(generated_label,generated_data,batch_data):
    g_loss = tf.math.reduce_mean(cross_entropy(tf.zeros_like(generated_label), generated_label))
    penalty = tf.math.reduce_mean(mean_squared_error(batch_data, generated_data)) 
    print('\t3. G Augment loss',penalty)
    print('\t4. G real loss',g_loss)
    return g_loss + penalty

def train_step(train_with_augmented = False):
    total_batch = int(samples.shape[0] / batch_size)
    direction_flag = False
    for batch_idx in range(total_batch):
        if (train_with_augmented):
            batch_data = tf.convert_to_tensor((data_utils.get_batch(augmented_samples,batch_size,batch_idx)),dtype=tf.float32)
            batch_label = tf.convert_to_tensor((data_utils.get_batch(augmented_labels,batch_size,batch_idx)),dtype=tf.float32)
        else:
            batch_data = tf.convert_to_tensor((data_utils.get_batch(samples,batch_size,batch_idx)),dtype=tf.float32)
        real_data = tf.convert_to_tensor((data_utils.get_batch(samples,batch_size,batch_idx)),dtype=tf.float32)
        # if (train_with_augmented):
        #     seed = tf.convert_to_tensor((data_utils.get_batch(augmented_samples,batch_size,2)),dtype=tf.float32)
        #     seed_label = tf.convert_to_tensor((data_utils.get_batch(augmented_labels,batch_size,batch_idx)),dtype=tf.float32)
        # else:
        seed = noise
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(seed,training =True)
            generated_input_predictions = discriminator(generated_data,training =True)
            # print('\n1. Generator Output: \t',generated_data[0][0])
            print('\n\n1. Faka data label: ',np.mean(generated_input_predictions[0]))
            
            # if train_with_augmented:
            #     gen_loss = G_loss_augment(generated_input_predictions,seed_label)
            # else:
            gen_loss = G_loss(generated_input_predictions,generated_data,real_data)
            # gen_loss = wasserstein_loss_g(generated_input_predictions)
            generator.trainable = True
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            # clipped_grads_of_generator = [tf.clip_by_value(grad, 100 , 30) for grad in gradients_of_generator]
            for i in range(settings['g_round']):
                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            generator.trainable = False
            direction_flag = not direction_flag

            real_input_predictions = discriminator(batch_data, training = True)
            real_input_predictions = tf.clip_by_value(real_input_predictions, 1e-5,  1) 
#             print('5. Realabel: ', np.mean(real_input_predictions[0]))
            # print('6. Augmented data label: ', np.mean(augmented_input_predictions[0]))

            # if train_with_augmented:
            #     disc_loss = D_loss_augment(real_input_predictions,batch_label,gen_loss)
            # else:
            disc_loss = D_loss(real_input_predictions, generated_input_predictions, batch_label)
            # disc_loss = wasserstein_loss_d(real_input_predictions, generated_input_predictions)

            discriminator.trainable = True
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            for i in range(settings['d_round']):
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            discriminator.trainable = False

            g_grads = []
            d_grads = []
            for grad in gradients_of_generator:
                g_grads.append(np.mean(grad))
            for grad in gradients_of_discriminator:
                d_grads.append(np.mean(grad))

            history_g_loss.append(gen_loss)
            history_d_loss.append(disc_loss)
            history_g_grads.append(np.mean(g_grads))
            history_d_grads.append(np.mean(d_grads))

#             if(train_with_augmented):
#                 history_augment_d_label.append(real_input_predictions[0])
#                 history_augment_g_label.append(generated_input_predictions[0])
#             else:
#                 history_random_g_label.append(generated_input_predictions[0])
#                 history_real_d_label.append(real_input_predictions[0])
    print ('Discriminator Loss: {} \tGenerator Loss: {}'.format(disc_loss, gen_loss))

    return (disc_loss)

def D_loss_train(real_data, fake_data):
    real_data_loss = tf.math.reduce_mean(cross_entropy(tf.zeros_like(real_data), real_data))
    fake_data_loss = 2*(tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_data), fake_data)))
    total_loss = real_data_loss + fake_data_loss
    return total_loss

def train_dicriminator():
    total_batch = int(samples.shape[0] / batch_size)
    for batch_idx in range(total_batch):
        normal_data = tf.convert_to_tensor((data_utils.get_batch(samples,batch_size,batch_idx)),dtype=tf.float32)
        std = random.randint(2,10)
        mean = random.randint(-2,12)
        s = random.randint(1,20)
        seed = get_seed(std,mean,s)
        with tf.GradientTape() as disc_tape_train:
            normal_data_prediction = discriminator(normal_data, training =True)
            outlier_prediction = discriminator(seed, training =True)
            
            loss = D_loss_train(normal_data_prediction,outlier_prediction)
            print('1. Normal data label: ', np.mean(normal_data_prediction[0]))
            print('2. Noise data label: ', np.mean(outlier_prediction[0]))
            discriminator.trainable = True
            D_grad = disc_tape_train.gradient(loss, discriminator.trainable_variables)
            for i in range(settings['d_round']):
                discriminator_optimizer.apply_gradients(zip(D_grad, discriminator.trainable_variables))
            discriminator.trainable = False
        print ('3. Loss: {} \n'.format(loss))

def train_GAN(epochs):
    loss_tracker_1st_step = []
    loss_tracker_second_step = []
    print('1st stage training')
    for epoch in range(int(epochs/2)):
        start = time.time()

        print('Epoch {} started'.format(epoch))
        print('====================================================================================================================')
        avg_d_loss = train_step(True)
        loss_tracker_1st_step.append(avg_d_loss)
        print ('Epoch Finished. Time for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    #save trained models
    # g_path = os.path.join('/Users\macio\Desktop\MAD-GAN_migrated\our_code\saved_model\generators\model_seq_' + str(seq_length) + '_' + settings['exp'] + '/')

    print('2nd stage training')    
    for epoch in range(int(epochs/2)):
        start = time.time()
        print('Epoch {} started'.format(epoch))
        print('====================================================================================================================')
        avg_d_loss = train_step(True)
        loss_tracker_second_step.append(avg_d_loss)
        print ('Epoch Finished. Time for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

    tf.saved_model.save(discriminator, d_path)
    # np.save('./predictions/history/history_random_g_label' ,history_random_g_label)
    # np.save('./predictions/history/history_augmented_g_label' ,history_augment_g_label)
    # np.save('./predictions/history/history_real_d_label' ,history_real_d_label)
    # np.save('./predictions/history/history_augment_d_label' ,history_augment_d_label)
    # np.save('./predictions/history/history_d_loss' ,history_d_loss)
    # np.save('./predictions/history/history_g_loss' ,history_g_loss)
    # np.save('./predictions/history/history_d_grad' ,history_d_grads)
    # np.save('./predictions/history/history_g_grad' ,history_g_grads)

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

        # np.save('./predictions/prediction_sequence_seq_length_'+ str(settings['seq_length'])+ '_' + str(2) ,D_test)
        # np.save('./predictions/real_sequence_seq_length_'+ str(settings['seq_length'])+ '_' + str(2) ,R_labels)

        results = np.zeros([10, 4])
        org_shape = data_utils.de_shape(D_test, R_labels, I_mb, seq_step)
        tao_max = np.max(org_shape)                       
        print(tao_max)
        # exit()
        for i in range(1, 10):
            tao = float(tao_max - (0.00595*i))
            Accu, Pre, Rec, F1 = data_utils.get_evaluation(D_test, R_labels, I_mb, seq_step, tao)
            print('Final Evaluation: tao={:.10}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}\n'
                  .format(tao, Accu, Pre, Rec, F1))
            results[i-1 , :] = [Accu, Pre, Rec, F1]
        return results

train_GAN(num_epoch)
generate_result()

    
