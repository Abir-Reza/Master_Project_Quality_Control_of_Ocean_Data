import numpy as np
import scipy.io 
import pandas as pd
import math
from matplotlib import pyplot as plt
import json
from pandas.plotting import parallel_coordinates
# import sys

input_directory = '/Users\macio\Desktop\MAD-GAN_migrated\data\RAW\MSS'
output_directory = '/Users\macio\Desktop\MAD-GAN_migrated\data\Processed\MSS'

diss_features = ['t','depth','nu','tdiff','eps1','kmin1','kmax1','acc','sinkvel','spikeflag1']
mat_features = ['she1']
file_dict = {
    'm135_1_mss026_':{
                                'data_frame': 'd',
                                'feature' : mat_features,
                                'directory': 'MAT'},
            'm135_1_':{
                                'data_frame': 'df',
                                'feature' : diss_features,
                                'directory': 'DISS'}
            }


def create_test_train_data():
    train_path = '/Users\macio\Desktop\MAD-GAN_migrated\data\Processed\MSS\combined_filter_data_diss_mat_train.npy'
    test_path = '/Users\macio\Desktop\MAD-GAN_migrated\data\Processed\MSS\combined_filter_data_diss_mat_test.npy'
    
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)

    temp_train_data = train_data
    indeces_to_remove = np.array([],int)

    #Remove outlires from train data.
    print('Before removing 1 shape: ',train_data.shape)
    for i in range(0, len(temp_train_data)-1, 1):
        if train_data[i][-1] > 0.0:
            indeces_to_remove = np.append(indeces_to_remove,i)

    temp_train_data = np.delete(temp_train_data,indeces_to_remove, axis=0)
    print('After removing 1 shape: ',temp_train_data.shape)
    train_data = temp_train_data
    print('Train Data shape',train_data.shape)
    print('Test Data shape',test_data.shape)
    np.save('/Users\macio\Desktop\MAD-GAN_migrated\data\kdd99_train.npy', train_data, allow_pickle=True)
    np.save('/Users\macio\Desktop\MAD-GAN_migrated\data\kdd99_test.npy', test_data, allow_pickle=True)

def get_windowed_data(data):
    mapped_data = []
    temp  = []
    data = data[0]
    window_size = 1024
    line_start = 23330
    window_start = line_start
    window_end = window_start + window_size
    total_window = int((len(data) - 23330)/window_size) 
    for index in range(1,total_window):
        temp_window = data[window_start:window_end]
        median = np.median(temp_window)
        temp.append(median)
        window_start = window_end
        window_end = window_end + window_size
    mapped_data.append(temp)
    return mapped_data

def preapre_data_from_file(number):
    if number < 10:
        number = '00' + str(number)
    else:
        number = '0' + str(number)
    for item in file_dict:
        if item == 'm135_1_mss026_':
            file_name = item + number
        else:
            file_name = item + number + 'dissfinal'
        
        item_dict = file_dict[item]
        path = input_directory + '\\' + item_dict['directory'] + '\\'+ file_name
        data_frame_name = item_dict['data_frame']
        features = item_dict['feature']

        file_data = scipy.io.loadmat(path)
        data_frame = file_data[data_frame_name]
        raw_data = data_frame[0][0]
        extracted_raw_data = []

        print('extracting data from ', file_name)
        for sensor in features:
            if item == 'spikeflag1':
                feature_data = np.array(raw_data[sensor][0][0]['opti'])
            else:
                feature_data = np.array(raw_data[sensor])
            
            m,n = feature_data.shape
            if m > 1:
                feature_data = np.reshape(feature_data,(n,m))
                windowed_data = get_windowed_data(feature_data)
            extracted_raw_data.append(windowed_data)

    return extracted_raw_data

def get_minimum_data_points(data):
    num_features = len(diss_features) + len(mat_features)
    size_list = []
    for item in data:
        size_list.append(np.size(item))
    min_size = min(size_list)
    minimum_data = np.empty((min_size,0), float)
    # print('Minimum datapoint ', min_size)
    for item in data:
        temp = np.array(item)
        m,n = temp.shape
        temp = temp.reshape(n,m)
        minimum_data = np.append(minimum_data, np.array(temp), axis=1)
    # print('After minimizing data shape: ', minimum_data.shape)
    return minimum_data,min_size

def read_file(number_of_files):
    train_file_amount = int(math.floor(number_of_files*.80)) 
    train_file_numbers = np.arange(1,train_file_amount,1, dtype=int)
    test_file_numbers = np.arange(train_file_amount,number_of_files,1,dtype=int)
    save_data_file(train_file_numbers,'train')
    save_data_file(test_file_numbers,'test')
    # print('After merging minimum data from all file the final data shape: ', final_data.shape)

def save_to_excel(data):
    df = pd.DataFrame(data).T
    df.to_excel(excel_writer = "/Users\macio\Desktop\MAD-GAN_migrated\data\Processed\MSS\DISS\Test2.xlsx")

def save_data_file(files,type):
    final_data = []
    for number in files:
        data = preapre_data_from_file(number)
        minimum_data_points,min = get_minimum_data_points(data)
        if len(final_data) > 1:
            final_data = np.append(final_data,minimum_data_points,axis = 0)
        else:
            temp_final = np.empty((min,0),float)
            final_data = np.append(temp_final,minimum_data_points,axis = 1)
    
    np.save(output_directory +'\\' + 'combined_filter_data_diss_mat_' + type, final_data)

def get_primary_shape(data,step):
    arr = data
    arr_2 = []
    for i in range(0,len(arr),1):
        if i == 0:
            m,n = arr[i].shape
            temp = np.array(np.reshape(arr[i],m))
            arr_2 = np.append(arr_2,temp)
            
        else:
            m,n = arr[i][-step:,:].shape
            temp = np.array(np.reshape(arr[i][-step:,:],m))
            arr_2 = np.append(arr_2,temp)
    return arr_2

def get_normal(data):
    min_val = min(data)
    max_val = max(data)
    for i in range(0,len(data),1):
         normalized_value = float((data[i]-min_val)/(max_val-min_val))
         data[i] = normalized_value

    return data
    
def show_with_real(real,pridected,epoch,tao):
    if tao:
        apply_threshold(pridected,tao)
    plot_data(pridected, real,epoch)


def plot_data(predicted,real,epoch):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.plot(predicted, color='red', label='Predicted Label')
    ax.plot(real, color='Green', label='Real Label')
    ax.set_ylim(0, 1.5)
    ax.set(ylabel='Boolean label',xlabel = 'Data points')
    ax.legend(loc='upper right') 
    ax.title.set_text('Real and Predicted data spikeflag')
    fig.tight_layout(pad=5.0)
    fig.savefig('test/normalized/threshold_test_real_fake_together' + str(epoch) +'.png')
    

def show(seq_lengths,num_epoch,tao = False):
    real_path = '././predictions/real_sequence_seq_length_'+ str(seq_lengths) +'_'
    prediction_path = '././predictions/prediction_sequence_seq_length_'+ str(seq_lengths) +'_'
    for i in range(0,num_epoch,1):
        prediction = np.load(prediction_path + str(i) + '.npy')
        reshaped_predictions = get_primary_shape(prediction,8)
        norm_reshaped_predictions = get_normal(reshaped_predictions)

        real = np.load(real_path + str(i) + '.npy')
        reshaped_real = get_primary_shape(real,8)
        epoch = str(i)+'_normalized'
        show_with_real(reshaped_real,norm_reshaped_predictions,epoch,tao)
    
def apply_threshold(data,tao):
    for i in range(0,len(data),1):
        if data[i] > tao:
            data[i] = 1
        else:
            data[i] = 0
    return data
# arr = np.array([[[1],[2],[3],[4],[5],[6]],[[4],[5],[6],[7],[8],[9]],[[7],[8],[9],[10],[11],[12]]])

def eval_result_whole(sub_id,settings,test_number):
    res_path = '././experiments/plots/Results' + '_' + sub_id + '_' + str(settings['sequence_size']) + '.npy'
    results = np.load(res_path)

    # Take only D_Test data results
    results = results[:,10:12,:]

    counter = 0
    m = results.shape[0]
    
    survay = np.empty(shape=(m, 4))
    for epoch in results:
        avg_results = []
        mean_acc = np.mean(epoch[:,:1,])
        mean_acc = mean_acc/100
        avg_results.append(mean_acc)
        mean_pre = np.mean(epoch[:, 1:2,])
        avg_results.append(mean_pre)
        mean_rec = np.mean(epoch[:,2:3,])
        avg_results.append(mean_rec)
        mean_f = np.mean(epoch[:,-1:,])
        avg_results.append(mean_f)
        survay[counter] = avg_results
        counter += 1

    fig, ax = plt.subplots(2,2, figsize=(10, 10))
    figure_title = read_dict(settings)
    fig.suptitle(figure_title)
    creiterias = [['Accuracy','red'],['Precision','blue'],['Recall','green'],['F_score','orange']]
    counter = 0
    for i in range(2):
        for j in range(2):
            if i==0 and j == 0:
                start = 0
            end = start + 1
            data = survay[:,start:end]
            text = creiterias[start][0]
            start = end
            ax[i][j].plot(data,color = colors[counter])
            ax[i][j].set(ylabel=text,xlabel = 'Number of Epoch')
            counter += 1
    fig.tight_layout(pad=3.0)
    fig.savefig('test/evaluations/evaluation_test_'+ test_number +'.png')
    path = 'test/evaluations/evaluation_test_'+ test_number +'.png'
    np.save(path,survay)

def eval_result_threshold(sub_id,settings,test_number):
    res_path = '././experiments/plots/Results' + '_' + sub_id + '_' + str(settings['sequence_size']) + '.npy'
    results = np.load(res_path)

    # Take only D_Test data results
    results = results[:,6:12,:]
    tao_count = 6
    survay = np.empty(shape=(tao_count, 4))
    epoch = results
    for i in range(0,tao_count):
        if i == 0:
            start = 0
        end = start + 1
        avg_results = []
        mean_acc = np.mean(epoch[:,start:end,:1])/100
        avg_results.append(mean_acc)
        mean_pre = np.mean(epoch[:,start:end, 1:2,])
        avg_results.append(mean_pre)
        mean_rec = np.mean(epoch[:,start:end,2:3,])
        avg_results.append(mean_rec)
        mean_f = np.mean(epoch[:,start:end ,-1:,])
        avg_results.append(mean_f)
        start = end
        survay[i][:] = avg_results

    fig, ax = plt.subplots(2,2, figsize=(10, 10))
    figure_title = read_dict(settings)
    fig.suptitle(figure_title)
    creiterias = [['Accuracy','red'],['Precision','blue'],['Recall','green'],['F_score','orange']]
    counter = 0
    for i in range(2):
        for j in range(2):
            if i==0 and j == 0:
                start = 0
                tao = 0.2
            end = start + 1
            tao = [[0.2],[0.3],[0.4],[0.5],[0.6],[0.7]]
            data = survay[:,start:end]
            text = creiterias[start][0]
            start = end
            ax[i][j].plot(tao,data, color = colors[counter]) 
            ax[i][j].set(ylabel=text,xlabel = 'Threshold')
            counter += 1
    fig.tight_layout(pad=3.0)
    fig.savefig('test/evaluations/thresholds/evaluation_test_'+ test_number +'.png')
    path = 'test/evaluations/thresholds/evaluation_test_'+ test_number +'.py'
    np.save(path,survay)

def read_dict(dict):
    text = ''
    for key in dict:
        text = text + key + ': ' + str(dict[key]) + '; '
    return text

def find_feature_corelation():
    path = '/Users\macio\Desktop\MAD-GAN_migrated\data\kdd99_test.npy'
    data = np.load(path, allow_pickle=True)
    feature_coeffs = []
    start = 0
    for i in range(0,len(diss_features),1):
        end = start + 1 
        norm_col = get_normal(data[:,start:end])
        coeff = np.corrcoef(np.array(norm_col),data[:,2:3],rowvar=False)
        start = end
        feature_coeffs.append(coeff)
    for i in range(0,len(diss_features)):
        print(diss_features[i])
        print(feature_coeffs[i][0][1])



    
colors = ['red','green','orange','blue']
settings_path = '././experiments/settings/kdd99_test.txt'
settings_loaded = json.load(open(settings_path, 'r'))
settings = {
    'learning_rate': settings_loaded['learning_rate'],
    'batch_size':settings_loaded['batch_size'],
    'sequence_size': settings_loaded['seq_length'],
    'sequence_step': settings_loaded['seq_step'],
    'hidden_units_g':settings_loaded['hidden_units_g'],
    'hidden_units_d':settings_loaded['hidden_units_d'],

}

read_file(98)
create_test_train_data()
# eval_result_whole('kdd99',settings,'65')
# eval_result_threshold('kdd99',settings,'65')
# show(settings['sequence_size'],30)
# find_feature_corelation()