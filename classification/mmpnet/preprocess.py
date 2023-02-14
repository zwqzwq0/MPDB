import numpy as np
from sklearn.model_selection import train_test_split
from numpy.fft import fft
from mne.io import concatenate_raws, read_raw_edf, read_epochs_eeglab


def get_label(epochs):
    true_label = []
    dic = {v: k for k, v in epochs.event_id.items()}
    test = epochs.events[:, -1]
    for record in epochs.events:
        if dic[record[2]] in ['139', '141', '145']:  # brake
            true_label.append(0)
        elif dic[record[2]] in ['125', '127']:  # turn
            true_label.append(1)
        elif dic[record[2]] in ['129', '131']:  # change
            true_label.append(2)
        elif dic[record[2]] in ['137', '143']:  # throttle
            true_label.append(3)
        elif dic[record[2]] in ['133']:  # stable
            true_label.append(4)
    return true_label


def find_class_index(label, c):
    y = []
    for i in range(len(label)):
        if label[i] == c:
            y.append(i)
    return y


def data_preprocess():
    DATA = np.zeros(shape=(5234, 64, 2000))
    y = []
    pointer = 0
    datapath = '/home/Users/zwq/EEGclassify/data_arrange/'
    for i in range(1, 31):
        EEG_temp = read_epochs_eeglab(datapath+'EEG/EEG_' + str(i) + '.set')
        EMG_temp = read_epochs_eeglab(datapath+'EMG/EMG_' + str(i) + '.set')
        GSR_temp = read_epochs_eeglab(datapath+'GSR/GSR_' + str(i) + '.set')
        label_EEG = get_label(EEG_temp)
        label_EMG = get_label(EMG_temp)
        label_GSR = get_label(GSR_temp)

        EEG = EEG_temp.get_data()
        EMG = EMG_temp.get_data()
        GSR = GSR_temp.get_data()

        EEG_index, EMG_index, GSR_index = [], [], []

        for i in range(5):
            EEG_index.append(find_class_index(label_EEG, i))
            EMG_index.append(find_class_index(label_EMG, i))
            GSR_index.append(find_class_index(label_GSR, i))
        # 按照EEG的数量对齐构建数组
        for i in range(5):
            for j in range(min(len(EEG_index[i]), len(EMG_index[i]), len(GSR_index[i]))):
                sample = np.concatenate(
                    [EEG[EEG_index[i][j], :, :], EMG[EMG_index[i][j], :, :], GSR[GSR_index[i][j], :, :]], axis=0)
                DATA[pointer, :, :] = sample
                pointer += 1
                y.append(i)
    DATA = abs(fft(DATA))
    return train_test_split(DATA, y, test_size=.2, random_state=35)     # X_train, X_test, y_train, y_test



if __name__ == "__main__":
    data_preprocess()
