import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler


freq_bands = ['all', 'delta', 'theta', 'alpha', 'beta', 'gamma']
subjects = ['zuoyaxi', 'zhangliuxin', 'zhaosijia', 'zhuyangxiangru', 'pujizhou', 'chenbingliang', 'mengdong', 'zhengxucen',
            'hexingtao', 'wanghuiling', 'panshuyi', 'wangsifan', 'zhaochangquan', 'wuxiangyu', 'xiajingtao', 'liujiaxin',
            'wangyanchu', 'liyizhou', 'weifenfen', 'chengyuting', 'chenjiajing', 'matianfang', 'liuledian', 'zuogangao', 'feicheng', 'xuyutong']

data_path_temp = r'/mnt/xlancefs/home/gwl20/code/data/features/{}s/{}_data_de_{}s.npy'
label_path_temp = r'/mnt/xlancefs/home/gwl20/code/data/features/{}s/{}_label_{}s.npy'
eye_feature_temp = r'/mnt/xlancefs/home/gwl20/code/data/features/eyefeature/final/{}_data_de_1s.npy'

eegeye_temp = r'/mnt/xlancefs/home/gwl20/code/data/eegeye_datanewsplit/{}_data_{}_de_{}s.npy'
eegeye_label_temp = r'/mnt/xlancefs/home/gwl20/code/data/eegeye_datanewsplit/{}_label_{}_de_{}s.npy'


slice_length = 5
for subj in subjects:
    print('='*10, subj)
    try:
        # (1800, 23)
        eye = np.load(eye_feature_temp.format(subj))
        eye = eye.reshape(360, -1)

        # (360, )
        label = np.load(label_path_temp.format(slice_length, subj, slice_length))

        # (360, 62, 5, 5)
        eeg = np.load(data_path_temp.format(slice_length, subj, slice_length))[0]
        
        for band in freq_bands:
            if band == 'all':
                eeg_ = eeg.reshape(360, -1)
            else:
                eeg_ = eeg[:, :, freq_bands.index(band)-1, :].reshape(360, -1)

            eegeye = np.concatenate((eeg_, eye), axis=1)
            print('Before SMOTE: ', eegeye.shape, label.shape)
            rs = SMOTE()
            eegeye, label_ = rs.fit_resample(eegeye, label)
            print('After SMOTE: ', eegeye.shape, label_.shape)
            print('Number of cases per class: ', label_.shape[0]/3)
            arr_idx = label_.argsort(kind='mergesort')
            eegeye = eegeye[arr_idx]
            label_ = label_[arr_idx]
            np.save(eegeye_temp.format(subj, band, slice_length), eegeye)
            np.save(eegeye_label_temp.format(subj, band, slice_length), label_)
    except IOError:
        print('=== {} missing'.format(subj))
        
# label = np.load(label_path_temp.format(5, 'zuoyaxi', 5))
# print(label.shape)
# print(label)