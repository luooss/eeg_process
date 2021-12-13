import os
import random
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from imblearn.over_sampling import SMOTE, RandomOverSampler


freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
data_path_temp = r'/mnt/xlancefs/home/gwl20/code/data/features/{}s/{}_data_{}_{}s.npy'
label_path_temp = r'/mnt/xlancefs/home/gwl20/code/data/features/{}s/{}_label_{}s.npy'

img_order = np.load(r'/mnt/xlancefs/home/gwl20/code/data/img_order.npy', allow_pickle=True).item()

# train test set into separate Datasets
class ArtGraphDataset(InMemoryDataset):
    def __init__(self, slice_length, feature, freq_band, subjects, exclude_images, root, oversample=False, transform=None, pre_transform=None):
        """[summary]

        Args:
            slice_length (int): 1s, 3s, 5s
            feature (str): psd, de
            freq_band (str): 
            subjects (list of str): 
            exclude_images (dict):
            root (str): 
            transform ([type], optional): [description]. Defaults to None.
            pre_transform ([type], optional): [description]. Defaults to None.
        """
        self.slice_length = slice_length
        self.feature = feature
        if freq_band == 'all':
            self.idx_band = -1
        else:
            self.idx_band = freq_bands.index(freq_band)
        self.subjects = subjects
        self.exclude_images = exclude_images
        self.oversample = oversample
        super(ArtGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for subject in self.subjects:
            data_path = data_path_temp.format(self.slice_length, subject, self.feature, self.slice_length)
            label_path = label_path_temp.format(self.slice_length, subject, self.slice_length)
            ncases_per_img = int(30 / self.slice_length)
            image_order = img_order[subject]
            excl_imgs = self.exclude_images[subject]
            mask = np.ones((60*ncases_per_img, ), dtype=int)
            for img in excl_imgs:
                idx = image_order.index(img)
                mask[idx*ncases_per_img : (idx+1)*ncases_per_img] = 0
            mask = mask == 1
            # (ncases, 62, 5, nseconds)
            data = np.load(data_path)[0][mask]
            label = np.load(label_path)[mask]
            if self.idx_band != -1:
                data = data[:, :, self.idx_band, :]
            ncases = data.shape[0]
            data = np.reshape(data, (ncases, -1))
            if self.oversample:
                rs = RandomOverSampler()
                data, label = rs.fit_resample(data, label)
            for i in range(data.shape[0]):
                x = torch.tensor(np.reshape(data[i], [62, -1]), dtype=torch.float)
                y = torch.tensor([label[i]], dtype=torch.long)
                edge_index = [[i, j] for i in range(62) for j in range(62)]
                edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                assert edge_index.shape[1] == 62 * 62
                graph_data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(graph_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class ArtDataset(Dataset):
    def __init__(self, slice_length, feature, freq_band, subjects, exclude_images, oversample=False):
        super().__init__()
        if freq_band == 'all':
            idx_band = -1
        else:
            idx_band = freq_bands.index(freq_band)

        self.data = []
        self.label = []
        
        for subject in subjects:
            data_path = data_path_temp.format(slice_length, subject, feature, slice_length)
            label_path = label_path_temp.format(slice_length, subject, slice_length)
            ncases_per_img = int(30 / slice_length)
            image_order = img_order[subject]
            excl_imgs = exclude_images[subject]
            mask = np.ones((60*ncases_per_img, ), dtype=int)
            for img in excl_imgs:
                idx = image_order.index(img)
                mask[idx*ncases_per_img : (idx+1)*ncases_per_img] = 0
            mask = mask == 1
            # (ncases, 62, 5, nseconds)
            data_ = np.load(data_path)[0][mask]
            label_ = np.load(label_path)[mask]
            if idx_band != -1:
                data_ = data_[:, :, idx_band, :]
            
            ncases = data_.shape[0]
            data_ = np.reshape(data_, (ncases, -1))
            if oversample:
                rs = RandomOverSampler()
                data_, label_ = rs.fit_resample(data_, label_)
            self.data.append(data_)
            self.label.append(label_)

        self.data = torch.from_numpy(np.concatenate(self.data, axis=0)).to(torch.float)
        self.label = torch.from_numpy(np.concatenate(self.label, axis=0)).to(torch.long)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]


class SEED_IV(Dataset):
    labels = [[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
              [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
              [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]
    freqs = {'delta':0,
             'theta':1,
             'alpha':2,
             'beta':3,
             'gamma':4}

    def __init__(self, dir_path, feature, smooth_method, freq_bands, subjects, DANN=None):
        """Initialize dataset.

        Args:
            dir_path (str): directory path
            feature (str): either 'de' or 'psd'
            smooth_method (str): either 'movingAve' or 'LDS'
            freq_bands (array): delta, theta, alpha, beta, gamma
            subjects (array): 1-15
        """
        super().__init__()
        self.DANN = DANN
        data_list = []
        label_list = []
        self.fbs = [self.freqs[band] for band in freq_bands]
        fpattern = feature + '_' + smooth_method
        for session in range(0, 3):
            for fname in os.listdir(os.path.join(dir_path, str(session+1))):
                if int(fname[:-4]) in subjects:
                    datap = sio.loadmat(os.path.join(dir_path, str(session+1), fname))
                    for k, v in datap.items():
                        if k.startswith(fpattern):
                            trial_number = int(k[len(fpattern):]) - 1
                            x = torch.from_numpy(v)[:, :, self.fbs].permute(1, 0, 2).flatten(start_dim=1)
                            data_list.append(x)
                            label_list += [self.labels[session][trial_number]] * x.size()[0]
        
        self.data = torch.cat(data_list, 0).to(torch.float)
        self.emotion_label = torch.tensor(label_list).to(torch.long)
        if self.DANN == 'source':
            self.domain_label = torch.zeros(self.data.size()[0]).to(torch.long)
        elif self.DANN == 'target':
            self.domain_label = torch.ones(self.data.size()[0]).to(torch.long)

        print('SEED-IV dataset preparation done:\n{}, {}, {}, {}'.format(fpattern, str(freq_bands), str(subjects), str(self.data.size())))
        unique_label, label_count = torch.unique(self.emotion_label, sorted=True, return_counts=True)
        label_count = torch.true_divide(label_count, label_count.sum())
        print('Label count: {}, {}'.format(str(unique_label), str(label_count)))
    
    def __len__(self):
        return self.data.size()[0]
    
    def __getitem__(self, index):
        if self.DANN is None:
            return self.data[index], self.emotion_label[index]
        else:
            return self.data[index], self.emotion_label[index], self.domain_label[index]
