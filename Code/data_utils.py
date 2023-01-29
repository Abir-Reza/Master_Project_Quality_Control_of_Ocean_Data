import numpy as np
from sklearn.decomposition import PCA

def process_train_data(num_signals,seq_length,seq_step):
    train = np.load('././data/kdd99_train.npy',allow_pickle=True)
    print('Loaded train data')
    m, n = train.shape  # m=562387, n=35

    # normalization
    for i in range(n - 1):
        # print('i=', i)
        A = max(train[:, i])
        # print('A=', A)
        if A != 0:
            train[:, i] /= max(train[:, i])
            # scale from -1 to 1
            train[:, i] = 2 * train[:, i] - 1
        else:
            train[:, i] = train[:, i]

    samples = train[:, 0:n - 1]
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
    T_n = np.matmul(X_n, pc.transpose(1, 0))
    print('After main sample multiplicated with PCA component the sample shape is:\n',T_n.shape)
    samples = T_n
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
    test = np.load('././data/kdd99_test.npy',allow_pickle=True)
    print('load kdd99_test from .npy')

    m, n = test.shape  # m1=494021, n1=35

    for i in range(n - 1):
        B = max(test[:, i])
        if B != 0:
            test[:, i] /= max(test[:, i])
            # scale from -1 to 1
            test[:, i] = 2 * test[:, i] - 1
        else:
            test[:, i] = test[:, i]

    samples = test[:, 0:n - 1]

    labels = test[:, n - 1]
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
    print('After main sample multiplicated with PCA component the sample shape is:\n',T_a.shape)
    samples = T_a
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
    return samples, labels

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