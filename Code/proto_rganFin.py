import tensorflow as tf
import json
import time
import numpy as np
import data_utilsFin
from keras.layers import LSTM, Dense, Input, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import regularizers
import os
from keras.regularizers import l2,l1
from matplotlib import pyplot as plt
import random

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
d_path = os.path.join('/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/saved_model/discriminators/model_seq_' + str(seq_length) + '_' + settings['exp'] + '/')


samples,labels = data_utilsFin.process_train_data(num_signal,seq_length,seq_step)

augmented_samples, augmented_labels = data_utilsFin.process_train_data(num_signal,seq_length,seq_step,True)

test_sample,test_labels, test_index = data_utilsFin.process_test_data(num_signal,seq_length,seq_step)

generator_optimizer = tf.keras.optimizers.Adam(.05)
discriminator_optimizer = tf.keras.optimizers.Adam(.01)

def get_seed(std,mn,s):
    seed = tf.random.normal([batch_size, seq_length, latent_dim],mean= mn,
    stddev=std,
    dtype=tf.dtypes.float32,
    seed=s)
    return seed
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
catagorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False) 

def create_generator():
    inputs = Input(batch_input_shape=(None,seq_length, latent_dim))
    x = LSTM(256, return_sequences=True)(inputs) 
    # x = BatchNormalization()(x)
    # x = Dropout(0.40)(x)
    x = LSTM(128, return_sequences=True)(inputs) 
    x = Dropout(0.25)(x)
    x = LSTM(16, activation='tanh',return_sequences=True)(x)
    outputs = Dense(num_signal, activation='tanh')(x)
    generator = Model(inputs, outputs)
    return generator

def create_discriminator():
    discriminator_input = Input(batch_input_shape=(None,seq_length, num_signal))
    x = LSTM(128, return_sequences=True, activation='tanh')(discriminator_input)
    # x = BatchNormalization()(x)
    x = LSTM(64, return_sequences=True,activation='tanh')(x)
    x = LeakyReLU()(x)
#     x = LSTM(32, return_sequences=True,activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    discriminator_output = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, discriminator_output)
    return discriminator

discriminator = create_discriminator()
generator = create_generator()

# g_weights = generator.trainable_weights
# d_weights = discriminator.trainable_weights

# np.save('./saved_parameters/g_weight_'+ str(settings['seq_length'])+ '_' +settings['exp'],g_weights)
# np.save('./saved_parameters/d_weight_'+ str(settings['seq_length'])+ '_' +settings['exp'],d_weights)
# np.save('./saved_parameters/seed_'+ str(settings['seq_length'])+ '_' +settings['exp'],seed)

# g_saved_weight = np.load('/Users\macio\Desktop\MAD-GAN_migrated\saved_parameters\set_1\g_weight_128_23.npy', allow_pickle=True)
# d_saved_weight = np.load('/Users\macio\Desktop\MAD-GAN_migrated\saved_parameters\set_1\d_weight_128_23.npy', allow_pickle=True)
# saved_seed = np.load('/Users\macio\Desktop\MAD-GAN_migrated\saved_parameters\set_1\seed_128_23.npy', allow_pickle=True)

# discriminator.set_weights(d_saved_weight)
# generator.set_weights(g_saved_weight)
# seed = tf.convert_to_tensor(saved_seed,dtype=tf.float32)


discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.05), loss='binary_crossentropy')
generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0022), loss='binary_crossentropy')

def D_loss(real_data, fake_data, augmented_data, augmented_label):
    real_data_loss = tf.math.reduce_mean(cross_entropy(tf.ones_like(real_data), real_data))
    fake_data_loss = tf.math.reduce_mean(cross_entropy(tf.zeros_like(fake_data), fake_data))
    augmented_data_loss = tf.math.reduce_mean(cross_entropy(augmented_label, augmented_data))
    
    real_data_loss = tf.clip_by_value(real_data_loss, 1e-4, 1e4)
    fake_data_loss = tf.clip_by_value(fake_data_loss, 1e-4, 1e4)
    augmented_data_loss = tf.clip_by_value(augmented_data_loss, 1e-4, 5e4)

    total_loss = real_data_loss + fake_data_loss + augmented_data_loss
    # print('7. g_loss',fake_data_loss)
    # print('8. d_loss',real_data_loss)
    # print('9. a_loss',augmented_data_loss)
    return total_loss

discriminator.trainable = False
  
def G_loss(fake_data):
    g_loss = tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_data), fake_data))
    g_loss = tf.clip_by_value(g_loss, 1e-5,  1.10983e4) 
    print('3. gen_loss',g_loss)
    return g_loss

def train_step():
    total_batch = int(samples.shape[0] / batch_size)
    for batch_idx in range(total_batch):
        batch_data = tf.convert_to_tensor((data_utilsFin.get_batch(samples,batch_size,batch_idx)),dtype=tf.float32)
        augmented_batch_data = tf.convert_to_tensor((data_utilsFin.get_batch(augmented_samples,batch_size,batch_idx)),dtype=tf.float32)
        augmented_label_data = tf.convert_to_tensor((data_utilsFin.get_batch(augmented_labels,batch_size,batch_idx)),dtype=tf.float32)
        direction_flag = False
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            std = random.randint(0,5)
            mean = random.randint(-2,4)
            s = random.randint(0,20)
            seed = get_seed(std,mean,s)
            generated_data = generator(seed)
            generated_input_predictions = discriminator(generated_data)
            print('\n1. Generator Output: \t',generated_data[0][0])
            print('2. Faka data label: ',np.mean(generated_input_predictions[0]))
            gen_loss = G_loss(generated_input_predictions)

            if direction_flag:
                min = 4
            else:
                min = -1.5
            generator.trainable = True
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            clipped_grads_of_generator = [tf.clip_by_value(grad, min , 30) for grad in gradients_of_generator]
            generator_optimizer.apply_gradients(zip(clipped_grads_of_generator, generator.trainable_variables))
            generator.trainable = False
            direction_flag = not direction_flag

            # g_grads = []
            # d_grads = []
            # for grad in gradients_of_generator:
            #     g_grads.append(np.mean(grad))
            # print('4. Generator gradients: ', np.mean(g_grads))

            real_input_predictions = discriminator(batch_data)
            real_input_predictions = tf.clip_by_value(real_input_predictions, 1e-5,  1) 
            augmented_input_predictions = discriminator(augmented_batch_data)
            augmented_input_predictions = tf.clip_by_value(augmented_input_predictions, 1e-5,  1)
            # print('5. Real data label: ', np.mean(real_input_predictions[0]))
            # print('6. Augmented data label: ', np.mean(augmented_input_predictions[0]))
            disc_loss = D_loss(real_input_predictions, generated_input_predictions, augmented_input_predictions, augmented_label_data)

            discriminator.trainable = True
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            # clipped_grads_of_discriminator, _ = tf.clip_by_global_norm(gradients_of_discriminator, 5e2)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            discriminator.trainable = False

            # for grad in gradients_of_discriminator:
            #     d_grads.append(np.mean(grad))
            # print('10. Discriminator gradients: ', np.mean(d_grads))
            # print ('Discriminator Loss: {} \tGenerator Loss: {}'.format(disc_loss, gen_loss))
    # plot_gradient(gradhistory, losshistory)
    print ('Discriminator Loss: {} \tGenerator Loss: {}'.format(disc_loss, gen_loss))

    return (disc_loss)

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

        np.save('/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/saved_model/prediction_sequence_seq_length_'+ str(settings['seq_length'])+ '_' + str(1) ,D_test)
        np.save('/home/students/MAD-GAN/Master_Project_Quality_Control_of_Ocean_Data/Code/saved_model/real_sequence_seq_length_'+ str(settings['seq_length'])+ '_' + str(1) ,R_labels)

        results = np.zeros([10, 4])
        org_shape = data_utilsFin.de_shape(D_test, R_labels, I_mb, seq_step)
        tao_max = .5                             
        # np.max(org_shape)
        # # print(D_test)
        # exit()
        for i in range(1, 10):
            tao = float(tao_max - (0.0015*i))
            Accu, Pre, Rec, F1 = data_utilsFin.get_evaluation(D_test, R_labels, I_mb, seq_step, tao)
            print('Final Evaluation: tao={:.2}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}\n'
                  .format(tao, Accu, Pre, Rec, F1))
            results[i-1 , :] = [Accu, Pre, Rec, F1]
        return results

# train_GAN(num_epoch)
generate_result()

    
