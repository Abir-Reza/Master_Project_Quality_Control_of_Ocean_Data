import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

def process_train_data(num_signals,seq_length,seq_step, augment = None):
    if augment:
        train = np.load('././data/contaminated_train.npy',allow_pickle=True)
    else:
        train = np.load('././data/kdd99_train.npy',allow_pickle=True)
    print('Loaded train data')
    m, n = train.shape  # m=562387, n=35

    samples = train[:, 0:n - 1]
    samples = np.interp(samples, (samples.min(), samples.max()), (-1, 1))  #Normalize in range(-1 to 1)

    labels = train[:, n - 1]  # the last colummn is label
    
    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    X_n = samples
    # -- the best PC dimension is chosen pc=6 -- #
    n_components = num_signals
    pca = PCA(n_components, svd_solver='full')
    pca.fit(X_n)
    ex_var = pca.explained_variance_ratio_
    pc = pca.components_
    # projected values on the principal component
    # T_n = np.matmul(X_n, pc.transpose(1, 0))
    # print('After main sample multiplicated with PCA component the sample shape is:\n',T_n.shape)
    print('Varience:\n',ex_var)
    # samples = T_n
    num_samples = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples, seq_length, num_signals])
    bb = np.empty([num_samples, seq_length, 1])

    for j in range(num_samples):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb
    return samples, labels

def process_test_data(num_signals,seq_length,seq_step):
    test = np.load('././data/contaminated_train.npy',allow_pickle=True)
    print('load kdd99_test from .npy')

    m, n = test.shape  # m1=494021, n1=35
    samples = test[:, 0:n - 1]
    samples = np.interp(samples, (samples.min(), samples.max()), (-1, 1))  #Normalize in range(-1 to 1)
    
    labels = test[:, n - 1]
    labels = labels.astype(int)
    np.save('./data/labels/real_label_' + str(len(labels)), labels)
    idx = np.asarray(list(range(0, m)))  # record the idx of each point
    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    X_a = samples
    # -- the best PC dimension is chosen pc=6 -- #
    n_components = num_signals
    pca_a = PCA(n_components, svd_solver='full')
    pca_a.fit(X_a)
    pc_a = pca_a.components_
    # projected values on the principal component
    T_a = np.matmul(X_a, pc_a.transpose(1, 0))
    # print('After main sample multiplicated with PCA component the sample shape is:\n',T_a.shape)
    # samples = T_a
    num_samples_t = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples_t, seq_length, num_signals])
    bb = np.empty([num_samples_t, seq_length, 1])
    bbb = np.empty([num_samples_t, seq_length, 1])

    for j in range(num_samples_t):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        bbb[j, :, :] = np.reshape(idx[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb
    index = bbb
    return samples, labels, index

def get_batch(samples, batch_size, batch_idx, labels=None):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    # print('-----------------start position:', start_pos)
    # print('-----------------end position:', end_pos)
    if labels is None:
        return samples[start_pos:end_pos]
    else:
        if type(labels) == tuple: # two sets of labels
            assert len(labels) == 2
            return samples[start_pos:end_pos], labels[0][start_pos:end_pos], labels[1][start_pos:end_pos]
        else:
            assert type(labels) == np.ndarray
            return samples[start_pos:end_pos], labels[start_pos:end_pos]

def seeder(num_signals,seq_length,seq_step,proximity):
    test = np.load('././data/kdd99_test.npy',allow_pickle=True)
    m, n = test.shape 
    samples = test[:, 0:n - 1]
    seed = np.zeros([m,n-1])
    
    for i in range(0,n-2):
        column = samples[:,i:i+1]
        mean = np.mean(column)
        var = np.var(column)
        temp_col = np.random.normal(loc=(mean*proximity),scale=(0.65*proximity),size=[m,1])
        seed[:,i:i+1] = temp_col
    
    num_samples_t = (seed.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples_t, seq_length, num_signals])
    for j in range(num_samples_t):
        for i in range(num_signals):
            aa[j, :, i] = seed[(j * seq_step):(j * seq_step + seq_length), i]
    seed = aa
    return seed

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

    np.save('/Users\macio\Desktop\MAD-GAN_migrated\our_code\labels\prediction_sequence_seq_length_256_exp_25' ,D_L)
    np.save('/Users\macio\Desktop\MAD-GAN_migrated\our_code\labels\\real_sequence_seq_length_256_exp_25',L_L)
    plt.hist(D_L, bins=50, density=True, color='blue')
    for i in range(LL):
        # print('Real: {} \tPredicted: {}'.format(L_L[i],D_L[i]))
        if D_L[i] > tao:
            D_L[i] = 1
        else:
            D_L[i] = 0
    cc = (D_L == L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    L_L = L_L.astype(bool)
    Accu = float((N / LL) * 100)
    precision, recall, f1, _ = precision_recall_fscore_support(L_L, D_L, average='binary', zero_division=0)
    return Accu, precision, recall, f1,

def de_shape(Label_test, Label_real, I_mb, seq_step):
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

    return D_L