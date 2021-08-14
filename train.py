import os
import argparse
import time
import random
from pprint import pprint
from numpy.lib.function_base import average
import numpy as np
from scipy.stats.stats import RepeatedResults
from torch._C import dtype
from torch.functional import split

from tqdm import tqdm

from collections import OrderedDict, namedtuple
from itertools import product

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, plot_confusion_matrix, f1_score
# KFold is not enough, need to make sure the ratio between classes is the same in both train set and test set
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, PredefinedSplit

import matplotlib.pyplot as plt

from dset import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--which',
                    help='Which model to train: 0-SVM, 1-DANN, 2-LSTM',
                    default='0',
                    type=int)
parser.add_argument('--maxepochs',
                    help='Maximum training epochs',
                    default=300,
                    type=int)

# LSTM hyper-parameters
parser.add_argument('--hidden_size',
                    help='Hidden size of the LSTM, enter multiple values like [32,64,128] starts hyperparameter test',
                    default='[128]',
                    type=str)
parser.add_argument('--num_layers',
                    help='Number of LSTM layers',
                    default='[3]',
                    type=str)


args = parser.parse_args()

experiments = ['subj_dependent', 'subj_independent']
# features = ['psd', 'de']
features = ['de']
freq_bands = ['all', 'delta', 'theta', 'alpha', 'beta', 'gamma']
subjects = ['chenyi', 'huangwenjing', 'huangxingbao', 'huatong', 'wuwenrui', 'yinhao']
phases = ['train', 'test']
indicators = ['accuracy', 'f1_macro']
data_dir = r'/home/PublicDir/luoshuai/bcmi'

class SVMTrainApp:
    def __init__(self, dset, split_strategy):
        """Train classifier.

        Args:
            dset (torch dataset): the data to be fitted.
            split_strategy (sklearn split generator): controls the type of experiment.
        """
        self.nsamples = dset.data.size()[0]
        self.data = dset.data.numpy().reshape(self.nsamples, -1)
        self.label = dset.label.numpy()
        self.split_strategy = split_strategy

        hyperparams = [{'kernel': ['linear'], 'C': np.logspace(-2, 10, 13)},
                       {'kernel': ['rbf'], 'C': np.logspace(-2, 10, 13), 'gamma': np.logspace(-9, 3, 13)}]
        # refit: after hp is determined, learn the best lp over the whole dataset, this is for prediction
        self.model = GridSearchCV(SVC(),
                                  param_grid=hyperparams,
                                  scoring=['accuracy', 'f1_macro'],
                                  n_jobs=-1,
                                  refit='f1_macro',
                                  cv=self.split_strategy,
                                  return_train_score=True)
    
    def main(self):
        X = StandardScaler().fit_transform(self.data)
        Y = self.label
        self.model.fit(X, Y)
        # pprint(self.model.cv_results_)
        idx = self.model.best_index_

        result = {}
        for ph in phases:
            result[ph] = {}
            for ind in indicators:
                result[ph][ind] = self.model.cv_results_['mean_'+ph+'_'+ind][idx]
        
        print('The best hyper parameters: {}'.format(self.model.best_params_))

        # train_cm_display = plot_confusion_matrix(model, x_train, y_train, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # train_cm_display.figure_.suptitle('{}_{}_{}: Train set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # train_cm_display.figure_.savefig('./figs/{}_{}_{}_train.png'.format(self.feature, self.freq_band, model_name))

        # test_cm_display = plot_confusion_matrix(model, x_test, y_test, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # test_cm_display.figure_.suptitle('{}_{}_{}: Test set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # test_cm_display.figure_.savefig('./figs/{}_{}_{}_test.png'.format(self.feature, self.freq_band, model_name))
        # plt.close('all')

        return result
        


class DANNTrainApp:
    def __init__(self, dset, split_strategy):
        self.dset = dset
        self.split_strategy = split_strategy
        self.nworkers = os.cpu_count()
        self.maxepochs = args.maxepochs
        self.batch_size = 256

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
    
    def get_runs(self):
        params = OrderedDict(lr = np.logspace(-4, -2, 3),
                             lambda_ = np.logspace(-2, 0, 3),
                             alpha = np.logspace(-2, 0, 3))
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

    def getModel(self, lambda_):
        dann = DANN()
        domain_clf = DomainClassifier(lambda_)
        if self.use_cuda:
            print('Using cuda. Total {:d} devices.'.format(torch.cuda.device_count()))
            # if torch.cuda.device_count() > 1:
            #     dann = nn.DataParallel(dann)
            #     domain_clf = nn.DataParallel(domain_clf)
        else:
            print('Using cpu.')

        return dann.to(self.device), domain_clf.to(self.device)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def getDataset(self):
        assert self.split_strategy.get_n_splits() == 1
        for train_indices, test_indices in self.split_strategy.split():
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

        # if self.use_cuda:
        #     self.batch_size *= torch.cuda.device_count()

        # the size of source and target in each batch should be equal, otherwise the model tends to focus on source
        source_dloader = DataLoader(self.dset, batch_size=self.batch_size // 2, num_workers=self.nworkers, pin_memory=self.use_cuda, shuffle=True, sampler=train_sampler)
        target_dloader = DataLoader(self.dset, batch_size=self.batch_size // 2, num_workers=self.nworkers, pin_memory=self.use_cuda, shuffle=True, sampler=test_sampler)
        
        return source_dloader, target_dloader

    def main(self):
        runs = self.get_runs()
        result = {}
        for run in runs:
            result[run] = {}
            for ph in phases:
                result[run][ph] = {}

        for run in runs:
            comment = ' DANN lr={} lambda_={} alpha={}'.format(run.lr, run.lambda_, run.alpha)
            dann_writer = SummaryWriter(comment=comment)

            print('Hyper Parameter test: ' + comment)
            start_outer = time.time()

            source_dloader, target_dloader = self.getDataset()
            nbatches = min(len(source_dloader), len(target_dloader))

            dann, domain_classifier = self.getModel(run.lambda_)
            # weight initialization
            dann.feature_extractor.apply(self.init_weights)
            dann.emotion_classifier.apply(self.init_weights)
            domain_classifier.apply(self.init_weights)

            # optim = torch.optim.Adam(list(dann.parameters()) + list(domain_classifier.parameters()))
            optim = torch.optim.SGD(list(dann.parameters()) + list(domain_classifier.parameters()), lr=run.lr)
            
            for epoch in tqdm(range(1, self.maxepochs+1)):
                loss, emotion_loss, domain_loss = 0, 0, 0
                emotion_pred_accuracy, domain_pred_accuracy = 0, 0

                for (src_data, src_emotion_label), (tgt_data, tgt_emotion_label) in zip(source_dloader, target_dloader):
                    src_domain_label = torch.zeros(src_emotion_label.size()[0]).to(torch.long)
                    tgt_domain_label = torch.ones(tgt_emotion_label.size()[0]).to(torch.long)
                    # (256, 62, 1, 5)
                    x = torch.cat([src_data, tgt_data], 0).to(self.device)
                    y_emotion = src_emotion_label.to(self.device)
                    y_domain = torch.cat([src_domain_label, tgt_domain_label], 0).to(self.device)

                    features = dann.feature_extractor(x)
                    pred_emotion = dann.emotion_classifier(features[:src_data.size()[0]])
                    pred_domain = domain_classifier(features)

                    # for nn.xxx, need to declare and then use because it is class
                    # for nn.functional.xxx, use directly
                    emotion_loss_b = F.nll_loss(pred_emotion, y_emotion)
                    emotion_loss += emotion_loss_b
                    domain_loss_b = F.nll_loss(pred_domain, y_domain)
                    domain_loss += domain_loss_b
                    loss_b = emotion_loss_b + run.alpha * domain_loss_b
                    loss += loss_b

                    optim.zero_grad()
                    loss_b.backward()
                    optim.step()

                    emotion_pred_accuracy += (pred_emotion.max(dim=1)[1] == y_emotion).float().mean().item()
                    domain_pred_accuracy += (pred_domain.max(dim=1)[1] == y_domain).float().mean().item()

                epa_mean = emotion_pred_accuracy / nbatches  # expected to increase
                dpa_mean = domain_pred_accuracy / nbatches  # expected to approximate 50%

                dann_writer.add_scalar('loss/total_loss', loss, epoch)
                dann_writer.add_scalar('loss/emotion_loss', emotion_loss, epoch)
                dann_writer.add_scalar('loss/domain_loss', domain_loss, epoch)
                dann_writer.add_scalar('accuracy/emotion_pred_accuracy', epa_mean, epoch)
                dann_writer.add_scalar('accuracy/domain_pred_accuracy', dpa_mean, epoch)
                dann_writer.flush()
                dann_writer.close()

                if epoch == 1 or epoch % 100 == 0:
                    tqdm.write('Epoch {:6d}: train_emotion_pred_accuracy {:.4f}, train_domain_pred_accuracy {:.4f}'.format(epoch, epa_mean, dpa_mean))
            
            validation_dloaders = [source_dloader, target_dloader]
            # validation on train set, test set
            for ph, der in zip(phases, validation_dloaders):
                with torch.no_grad():
                    y_true, y_pred = [], []
                    for tx, ty_e in der:
                        tx = tx.to(self.device, non_blocking=True)
                        ty_e = ty_e.to(self.device, non_blocking=True)

                        tpred_e = dann(tx)
                        y_p = tpred_e.max(dim=1)[1].to(device='cpu').numpy()
                        y_t = ty_e.to(device='cpu').numpy()
                        y_true.append(y_t)
                        y_pred.append(y_p)
                    
                    y_true = np.concatenate(y_true, axis=0)
                    y_pred = np.concatenate(y_pred, axis=0)
                    acc = accuracy_score(y_true, y_pred)
                    f1_macro = f1_score(y_true, y_pred, average='macro')
                    result[run][ph]['accuracy'] = acc
                    result[run][ph]['f1_macro'] = f1_macro
                    tqdm.write('For this run, {} set accuracy/f1: {:.4f}/{:.4f}'.format(ph, acc, f1_macro))
        
            end_outer = time.time()
            dur_outer = end_outer - start_outer
            print('For this run, train time: {:4d}min {:2d}sec'.format(int(dur_outer // 60), int(dur_outer % 60)))
        
        best_run = runs[0]
        for run in runs:
            if(result[run]['test']['f1_macro'] > result[best_run]['test']['f1_macro']):
                best_run = run
        
        print('Best hyper parameter: lr={} lambda_={} alpha={}'.format(run.lr, run.lambda_, run.alpha))
        return result[best_run]



class LSTMTrainApp:
    def __init__(self):
        self.dir_path = args.datapath
        self.nworkers = os.cpu_count()
        self.lr = args.lr
        self.maxepochs = args.maxepochs
        
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.num_classes = 4

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

    def getModel(self, seq_len, input_size):
        model = LSTM_Classification(seq_len, input_size, self.hidden_size, self.num_layers, self.num_classes)

        if self.use_cuda:
            print('Using cuda. Total {:d} devices.'.format(torch.cuda.device_count()))
            # if torch.cuda.device_count() > 1:
            #     dann = nn.DataParallel(dann)
            #     domain_clf = nn.DataParallel(domain_clf)
        else:
            print('Using cpu.')
        return model.to(self.device)
    
    def getDataset(self, feature, smooth_method, frq_bands, target_subject):
        dset_train = SEED_IV(self.dir_path, feature, smooth_method, frq_bands, [i for i in range(1, 16) if i != target_subject])
        dset_test = SEED_IV(self.dir_path, feature, smooth_method, frq_bands, [target_subject])

        dloader_train = DataLoader(dset_train, batch_size=128, num_workers=self.nworkers, pin_memory=self.use_cuda, shuffle=True)
        dloader_test = DataLoader(dset_test, batch_size=128, num_workers=self.nworkers, pin_memory=self.use_cuda, shuffle=True)
        return dloader_train, dloader_test, dset_train.data.size()[1]

    def main(self):
        print('#'*100)
        print('LSTM\n')

        for feature in ['de', 'psd']:
            for smooth_method in ['movingAve', 'LDS']:
                for frq_bands in [['delta'], ['theta'], ['alpha'], ['beta'], ['gamma'], ['delta', 'theta', 'alpha', 'beta', 'gamma']]:
                    print('\nFeature in use: {}\nFrequency bands in use: {} >>>'.format(feature+'_'+smooth_method, str(frq_bands)))
                    test_accuracies = []
                    start_outer = time.time()
                    for target_subject in range(1, 16):
                        print('Target on {}'.format(target_subject))
                        dloader_train, dloader_test, seq_dim = self.getDataset(feature, smooth_method, frq_bands, target_subject)
                        
                        seq_len = 2
                        input_size = seq_dim // 2

                        model = self.getModel(seq_len, input_size)
                        # optim = opt.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.1)
                        optim = opt.Adam(model.parameters(), lr=self.lr)

                        for epoch in tqdm(range(1, self.maxepochs+1)):
                            train_pred_right = 0
                            train_total = 0
                            for x, y in dloader_train:
                                x = x.view(-1, seq_len, input_size).to(self.device)
                                y = y.to(self.device)

                                pred = model(x)

                                loss = F.nll_loss(pred, y)

                                optim.zero_grad()
                                loss.backward()
                                optim.step()

                                train_pred_right += (pred.max(dim=1)[1] == y).float().sum().item()
                                train_total += x.shape[0]
                            
                            train_pred_acc = train_pred_right / train_total
                            if epoch == 1 or epoch % 100 == 0:
                                tqdm.write('Epoch {:6d}: train_pred_accuracy {:10.4f}'.format(epoch, train_pred_acc))
                        
                        with torch.no_grad():
                            test_pred_right = 0
                            test_total = 0
                            for x, y in dloader_test:
                                x = x.view(-1, seq_len, input_size).to(self.device)
                                y = y.to(self.device)

                                pred = model(x)

                                test_pred_right += (pred.max(dim=1)[1] == y).float().sum().item()
                                test_total += x.shape[0]
                            
                            test_pred_acc = test_pred_right / test_total
                        
                        test_accuracies.append(test_pred_acc)
                    
                    end_outer = time.time()
                    dur_outer = end_outer - start_outer
                    print('Train time: {:4d}min {:2d}sec'.format(int(dur_outer // 60), int(dur_outer % 60)))

                    cv_acc_mean = torch.tensor(test_accuracies).mean().item()
                    cv_acc_std = torch.tensor(test_accuracies).std().item()
                    print('Leave-one-out cross validation, mean: {:10.4f}, std: {:10.4f}'.format(cv_acc_mean, cv_acc_std))



if __name__ == '__main__':
    for exp in experiments:
        print('#'*50, 'Experiment: ', exp)
        for feature in features:
            print('#'*30, 'Feature: ', feature)
            for freq in freq_bands:
                print('#'*20, 'Freq. Band: ', freq)
                model_name = None
                exp_result = {}

                if exp == 'subj_dependent':
                    for subject in subjects:
                        print('#'*10, 'Train on ', subject)

                        data_path = data_dir + '/' + subject + '_data_' + feature + '.npy'
                        label_path = data_dir + '/' + subject + '_label.npy'
                        dset = ArtDataset([data_path], [label_path], freq_band=freq)
                        split_strategy = StratifiedShuffleSplit(n_splits=9, test_size=100)
                        
                        if args.which == 0:
                            model_name = 'SVM'
                            print('>>> Model: SVM')
                            result = SVMTrainApp(dset, split_strategy).main()
                        elif args.which == 1:
                            pass
                        
                        exp_result[subject] = result
                elif exp == 'subj_independent':
                    data_paths = [data_dir + '/' + subj + '_data_' + feature + '.npy' for subj in subjects]
                    label_paths = [data_dir + '/' + subj + '_label.npy' for subj in subjects]
                    # (6*900, ...)
                    dset = ArtDataset(data_paths, label_paths, freq_band=freq)

                    for subject in subjects:
                        print('#'*10, 'Target on ', subject)
                        subj_idx = subjects.index(subject)

                        test_fold = np.empty(5400, dtype=np.int8)
                        test_fold.fill(-1)
                        test_fold[subj_idx*900: (subj_idx+1)*900] = 0
                        split_strategy = PredefinedSplit(test_fold)
                        
                        if args.which == 0:
                            model_name = 'SVM'
                            print('>>> Model: SVM')
                            result = SVMTrainApp(dset, split_strategy).main()
                        elif args.which == 1:
                            model_name = 'DANN'
                            print('>>> Model: DANN')
                            result = DANNTrainApp(dset, split_strategy).main()
                        
                        exp_result[subject] = result

                print('Result:')
                pprint(exp_result)

                subj_train_accs = np.array([round(exp_result[subj]['train']['accuracy'], 4) for subj in subjects])
                subj_train_f1s = np.array([round(exp_result[subj]['train']['f1_macro'], 4) for subj in subjects])
                subj_test_accs = np.array([round(exp_result[subj]['test']['accuracy'], 4) for subj in subjects])
                subj_test_f1s = np.array([round(exp_result[subj]['test']['f1_macro'], 4) for subj in subjects])

                plt.style.use('seaborn')
                x = np.arange(0, (len(subjects)-1)*2.5+1, 2.5)  # the label locations
                width = 1.0  # the width of the bars
                fig, ax = plt.subplots(figsize=(14.8, 7.8))
                acc_train_rect = ax.bar(x - width/2, subj_train_accs, width, label='Train/Acc', fill=False, ls='--')
                acc_test_rect = ax.bar(x - width/2, subj_test_accs, width, label='Test/Acc')
                f1_train_rect = ax.bar(x + width/2, subj_train_f1s, width, label='Train/F1', fill=False, ls='--')
                f1_test_rect = ax.bar(x + width/2, subj_test_f1s, width, label='Test/F1')
                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_xlabel('Subjects')
                ax.set_title('{}_{}_{}_{}'.format(exp, feature, freq, model_name), pad=36)
                ax.set_xticks(x)
                ax.set_xticklabels(subjects)
                ax.set_ylim(0.0, 1.0)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend([acc_train_rect, acc_test_rect, f1_test_rect], ['Train', 'Test/Acc.', 'Test/F1.'], loc='center left', bbox_to_anchor=(1, 0.5))
                ax.bar_label(acc_train_rect, padding=3)
                ax.bar_label(acc_test_rect, padding=3)
                ax.bar_label(f1_train_rect, padding=3)
                ax.bar_label(f1_test_rect, padding=3)
                fig.savefig('./figs/{}_{}_{}_{}.png'.format(exp, feature, freq, model_name))
                plt.close('all')

                print('====Train:\nacc: {:.4f}/{:.4f}\nf1: {:.4f}/{:.4f}'.format(subj_train_accs.mean(), subj_train_accs.std(), subj_train_f1s.mean(), subj_train_f1s.std()))
                print('====Test:\nacc: {:.4f}/{:.4f}\nf1: {:.4f}/{:.4f}'.format(subj_test_accs.mean(), subj_test_accs.std(), subj_test_f1s.mean(), subj_test_f1s.std()))


