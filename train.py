import os
import argparse
import time
from pprint import pprint
import numpy as np
import datetime
from pprint import pprint
import logging
import math

from tqdm import tqdm

from collections import OrderedDict, namedtuple
from itertools import product

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
# KFold is not enough, need to make sure the ratio between classes is the same in both train set and test set
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, PredefinedSplit, StratifiedKFold

import matplotlib
import matplotlib.pyplot as plt

from dset import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--exp',
                    help='What experiment: subj_dep, subj_indep',
                    default='subj_dep',
                    type=str)
parser.add_argument('--which',
                    help='Which model to train: 0-SVM, 1-GNN',
                    default=0,
                    type=int)
parser.add_argument('--slice_length',
                    help='in seconds',
                    default=1,
                    type=int)
parser.add_argument('--feature',
                    help='psd, de',
                    default='de',
                    type=str)
parser.add_argument('--freqbands',
                    help='Freq bands',
                    default="['all', 'delta', 'theta', 'alpha', 'beta', 'gamma']",
                    type=str)


# GNN hyper params
# (5, 1), (5, 3), (5, 5)
parser.add_argument('--maxepochs',
                    help='Maximum training epochs',
                    default=1000,
                    type=int)
parser.add_argument('--lr',
                    help='Learning rate',
                    default='np.logspace(-4, -1, 4)',
                    type=str)
parser.add_argument('--lr_fe',
                    help='Learning rate',
                    default='np.logspace(-4, -1, 4)',
                    type=str)
parser.add_argument('--lr_lc',
                    help='Learning rate',
                    default='np.logspace(-4, -1, 4)',
                    type=str)
parser.add_argument('--weight_decay',
                    help='Weight decay for optimizer',
                    default='1e-5',
                    type=float)
parser.add_argument('--label_type',
                    help='hard or soft',
                    default='soft',
                    type=str)
parser.add_argument('--num_hidden',
                    help='The dimensionality of the learned embedding for nodes',
                    default='[3, 5, 8, 13, 15, 25]',
                    type=str)
parser.add_argument('--K',
                    help='Number of layers',
                    default='[5]',
                    type=str)
parser.add_argument('--dropout',
                    help='Dropout',
                    default='[0.7]',
                    type=str)
parser.add_argument('--epsilon',
                    help='Soft label epsilon',
                    default='np.logspace(-2, -1, 2)',
                    type=str)
parser.add_argument('--lambda_',
                    help='Coefficient of inverse gradient',
                    default='np.logspace(-4, 0, 5)',
                    type=str)
parser.add_argument('--alpha',
                    help='Coefficient of domain prediction loss',
                    default='np.logspace(-4, 0, 5)',
                    type=str)
parser.add_argument('--reclosse',
                    help='Coefficient of reconstruct loss',
                    default='np.logspace(-4, 0, 5)',
                    type=str)
parser.add_argument('--lablosse',
                    help='Coefficient of reconstruct loss',
                    default='np.logspace(-4, 0, 5)',
                    type=str)

args = parser.parse_args()

all_subjects = ['zuoyaxi', 'zhangliuxin', 'zhaosijia', 'zhuyangxiangru', 'pujizhou', 'chenbingliang', 'mengdong', 'zhengxucen',
            'hexingtao', 'wanghuiling', 'panshuyi', 'wangsifan', 'zhaochangquan', 'wuxiangyu', 'xiajingtao', 'liujiaxin',
            'wangyanchu', 'liyizhou', 'weifenfen', 'chengyuting', 'chenjiajing', 'matianfang', 'liuledian', 'zuogangao', 'feicheng', 'xuyutong']
# subjects = ['zuoyaxi', 'zhangliuxin', 'zhaosijia', 'zhuyangxiangru', 'pujizhou', 'chenbingliang', 'zhengxucen',
#             'hexingtao', 'wanghuiling', 'panshuyi', 'zhaochangquan', 'wangyanchu', 'liyizhou', 'chengyuting',
#             'chenjiajing', 'matianfang', 'liuledian', 'zuogangao', 'feicheng', 'xuyutong']
eyedata_missed = ['zhuyangxiangru', 'zhaochangquan']

subjects = ['zuoyaxi', 'zhangliuxin', 'zhaosijia', 'pujizhou', 'chenbingliang', 'zhengxucen',
            'hexingtao', 'wanghuiling', 'panshuyi', 'xiajingtao', 'liujiaxin',
            'wangyanchu', 'liyizhou', 'chengyuting', 'chenjiajing', 'matianfang', 'liuledian', 'zuogangao', 'feicheng', 'xuyutong']

phases = ['train', 'test']
indicators = ['accuracy', 'f1_macro']
data_dir = r'/mnt/xlancefs/home/gwl20/code/data/features/'
graph_data_root_dir = r'/mnt/xlancefs/home/gwl20/code/data/features/graph_data/'
exclude_imgs = np.load(r'/mnt/xlancefs/home/gwl20/code/data/exlcude_img_arousal_3.npy', allow_pickle=True).item()
nimgs_each_subject = []
for subject in subjects:
    nimgs_each_subject.append(60 - len(exclude_imgs[subject]))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter("%(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler('./logs_newdatasplit/eegeye_{}_{}_{}.log'.format(args.exp, args.which, args.slice_length), mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

class SVMTrainApp:
    def __init__(self, dset, split_strategy):
        """Train classifier.

        Args:
            dset (torch dataset): the data to be fitted.
            split_strategy (sklearn split generator): controls the type of experiment.
        """
        self.data = dset.data.numpy()
        self.label = dset.label.numpy()
        self.split_strategy = split_strategy

        # hyperparams = [{'kernel': ['rbf'], 'C': [math.pow(2, p) for p in range(-10, 10)], 'gamma': ['scale', 'auto']},
        #                {'kernel': ['linear'], 'C': [math.pow(2, p) for p in range(-10, 10)]}]
        hyperparams = [{'kernel': ['linear'], 'C': [math.pow(2, p) for p in range(-10, 10)]}]
        # refit: after hp is determined, learn the best lp over the whole dataset, this is for prediction
        self.model = GridSearchCV(SVC(),
                                  param_grid=hyperparams,
                                  scoring=['accuracy', 'f1_macro'],
                                  n_jobs=16,
                                  refit='f1_macro',
                                  cv=self.split_strategy,
                                  verbose=2,
                                  return_train_score=True)
    
    def main(self):
        # 在什么维度上归一化？
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

        logger.info('The best hyper parameters: {}'.format(self.model.best_params_))

        pred = self.model.best_estimator_.predict(X)
        overall_acc = accuracy_score(Y, pred)
        overall_f1 = f1_score(Y, pred, average='macro')
        overall_class_acc = confusion_matrix(Y, pred, normalize='true').diagonal()

        logger.info('Overall acc: {:.4f}'.format(overall_acc))
        logger.info('Overall f1: {:.4f}'.format(overall_f1))
        logger.info('Class Acc: ' + str(overall_class_acc))

        # pic_result = []
        # for pp in range(90):
        #     pic_result.append(self.model.cv_results_['split{}_test_accuracy'.format(pp)][idx])

        # train_cm_display = plot_confusion_matrix(model, x_train, y_train, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # train_cm_display.figure_.suptitle('{}_{}_{}: Train set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # train_cm_display.figure_.savefig('./figs/{}_{}_{}_train.png'.format(self.feature, self.freq_band, model_name))

        # test_cm_display = plot_confusion_matrix(model, x_test, y_test, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # test_cm_display.figure_.suptitle('{}_{}_{}: Test set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # test_cm_display.figure_.savefig('./figs/{}_{}_{}_test.png'.format(self.feature, self.freq_band, model_name))
        # plt.close('all')

        return result


class GNNDomainAdaptationTrainApp:
    def __init__(self, train_dset, test_dset, label_type):
        self.train_dset = train_dset
        self.test_dset = test_dset
        self.label_type = label_type
        self.nworkers = os.cpu_count()
        self.maxepochs = args.maxepochs

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
    
    def get_runs(self):
        params = OrderedDict(lr = eval(args.lr),
                             num_hidden = eval(args.num_hidden),
                             K = eval(args.K),
                             dropout=eval(args.dropout),
                             lambda_ = eval(args.lambda_),
                             alpha = eval(args.alpha),
                             epsilon = eval(args.epsilon))
        
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
    
    def getInitialEdgeWeightMatrix(self):
        adj = np.zeros((62, 62))
        xs, ys = np.tril_indices(62, -1)
        adj[xs, ys] = np.random.uniform(-1, 1, xs.shape[0])
        adj = adj + adj.T + np.identity(len(adj))
        return torch.tensor(adj, dtype=torch.float)

    def getModel(self, num_hidden, K, dropout):
        edge_weight = self.getInitialEdgeWeightMatrix()
        sgc = SGCFeatureExtractor(num_nodes=62,
                                  learn_edge_weight=True,
                                  edge_weight=edge_weight,
                                  num_features=self.train_dset.num_node_features,
                                  num_hidden=num_hidden,
                                  K=K)
        label_classifier = LabelClassifier(62*self.train_dset.num_node_features, dropout)
        domain_classifier = DomainClassifier(62*self.train_dset.num_node_features, dropout)
        
        if self.use_cuda:
            logger.info('Using cuda. Total {:d} devices.'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                sgc = DataParallel(sgc)
                label_classifier = nn.DataParallel(label_classifier)
                domain_classifier = nn.DataParallel(domain_classifier)
        else:
            logger.info('Using cpu.')

        return sgc.to(self.device), label_classifier.to(self.device), domain_classifier.to(self.device)

    def getSoftLabel(self, y, epsilon):
        """[summary]

        Args:
            y : (batch_size, )

        Returns:
            soft_y : (batch_size, 3)
        """
        batch_size = y.size(dim=0)
        soft_y = np.zeros((batch_size, 3))
        for i in range(batch_size):
            if y[i] == 0:
                # negative
                soft_y[i] = [1-2*epsilon/3, 2*epsilon/3, 0]
            elif y[i] == 1:
                # neutral
                soft_y[i] = [epsilon/3, 1-2*epsilon/3, epsilon/3]
            elif y[i] == 2:
                # positive
                soft_y[i] = [0, 2*epsilon/3, 1-2*epsilon/3]
        
        return torch.tensor(soft_y, dtype=torch.float, device=self.device)

    def test(self, sgc, label_classifier, domain_classifier, dloader):
        sgc.eval()
        label_classifier.eval()
        domain_classifier.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for graph_list in dloader:
                # (batch_size, num_hidden)
                sgc_output = sgc(graph_list)
                # (batch_size, 3)
                labcl_output = label_classifier(sgc_output)
                
                y = torch.cat([data.y for data in graph_list]).to(labcl_output.device)
                y_pred.append(torch.argmax(labcl_output, dim=1).detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())
            
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
        
        return acc, f1_macro

    def main(self):
        runs = self.get_runs()
        result = {}
        for run in runs:
            result[run] = {}
            for ph in phases:
                result[run][ph] = {}

        for run in runs:
            comment = ' RGNN_DA lr={} num_hidden={} K={} dropout={} lambda_={} alpha={} epsilon={}'.format(run.lr, run.num_hidden, run.K, run.dropout, run.lambda_, run.alpha, run.epsilon)
            
            # dann_writer = SummaryWriter(comment=comment)

            logger.info('Hyper Parameter test: ' + comment)
            start_outer = time.time()

            train_dloader = DataListLoader(self.train_dset, batch_size=16, shuffle=True, drop_last=True)
            test_dloader = DataListLoader(self.test_dset, batch_size=16, shuffle=True, drop_last=True)

            sgc, label_classifier, domain_classifier = self.getModel(run.num_hidden, run.K, run.dropout)

            optim = torch.optim.Adam(list(sgc.parameters())+list(label_classifier.parameters())+list(domain_classifier.parameters()), lr=run.lr, weight_decay=args.weight_decay)
            
            sgc.train()
            label_classifier.train()
            domain_classifier.train()
            for epoch in tqdm(range(1, self.maxepochs+1)):
                loss, label_loss, domain_loss = 0, 0, 0
                label_pred_accuracy, domain_pred_accuracy = 0, 0

                batches = zip(train_dloader, test_dloader)
                n_batches = min(len(train_dloader), len(test_dloader))
                i = 0
                for train_graph_list, test_graph_list in tqdm(batches, leave=False, total=n_batches):

                    train_domain_label = torch.zeros(16).to(torch.long).to(self.device)
                    test_domain_label = torch.ones(16).to(torch.long).to(self.device)

                    # (batch_size, num_hidden)
                    train_sgc_output = sgc(train_graph_list)
                    test_sgc_output = sgc(test_graph_list)
                    # (batch_size, 3)
                    train_labcl_output = label_classifier(train_sgc_output)
                    # (batch_size, 2)
                    p = float(i + epoch * n_batches) / self.maxepochs / n_batches
                    lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
                    train_domcl_output = domain_classifier(train_sgc_output, lambda_)
                    test_domcl_output = domain_classifier(test_sgc_output, lambda_)
                    
                    y = torch.cat([data.y for data in train_graph_list]).to(train_labcl_output.device)
                    if self.label_type == 'hard':
                        label_loss_b = F.nll_loss(train_labcl_output, y)
                    elif self.label_type == 'soft':
                        soft_y = self.getSoftLabel(y, run.epsilon)
                        label_loss_b = F.kl_div(train_labcl_output, soft_y, reduction='batchmean')
                    
                    label_loss += label_loss_b
                    domain_loss_b = F.nll_loss(train_domcl_output, train_domain_label) + F.nll_loss(test_domcl_output, test_domain_label)
                    domain_loss += domain_loss_b
                    loss_b = label_loss_b + run.alpha * domain_loss_b
                    loss += loss_b

                    optim.zero_grad()
                    loss_b.backward()
                    optim.step()

                    label_pred_accuracy += (train_labcl_output.max(dim=1)[1] == y).float().mean().item()
                    domain_pred_accuracy += ((train_domcl_output.max(dim=1)[1] == train_domain_label).float().mean().item() + (test_domcl_output.max(dim=1)[1] == test_domain_label).float().mean().item()) / 2

                    i += 1

                epa_mean = label_pred_accuracy / n_batches  # expected to increase
                dpa_mean = domain_pred_accuracy / n_batches  # expected to approximate 50%

                # dann_writer.add_scalar('loss/total_loss', loss, epoch)
                # dann_writer.add_scalar('loss/label_loss', label_loss, epoch)
                # dann_writer.add_scalar('loss/domain_loss', domain_loss, epoch)
                # dann_writer.add_scalar('accuracy/label_pred_accuracy', epa_mean, epoch)
                # dann_writer.add_scalar('accuracy/domain_pred_accuracy', dpa_mean, epoch)
                # dann_writer.flush()
                # dann_writer.close()

                if epoch==1 or epoch%10==0:
                    logger.debug('Epoch {:6d}: train_label_pred_accuracy {:.4f}, train_domain_pred_accuracy {:.4f}, domain_loss {:.4f}, label_loss {:.4f}, loss {:.4f}'.format(epoch, epa_mean, dpa_mean, domain_loss, label_loss, loss))
                    tqdm.write('Epoch {:6d}: train_label_pred_accuracy {:.4f}, train_domain_pred_accuracy {:.4f}, domain_loss {:.4f}, label_loss {:.4f}, loss {:.4f}'.format(epoch, epa_mean, dpa_mean, domain_loss, label_loss, loss))
                if epoch % 50 == 0:
                    test_acc, test_f1 = self.test(sgc, label_classifier, domain_classifier, test_dloader)
                    logger.debug('====== Test: test_label_pred_accuracy {:.4f}, test_f1 {:.4f}'.format(test_acc, test_f1))
                    tqdm.write('====== Test: test_label_pred_accuracy {:.4f}, test_f1 {:.4f}'.format(test_acc, test_f1))
                    if test_f1 > 0.9:
                        break
        
            end_outer = time.time()
            dur_outer = end_outer - start_outer
            logger.info('For this run, train time: {:4d}min {:2d}sec'.format(int(dur_outer // 60), int(dur_outer % 60)))
            
            # test
            validation_dloaders = [train_dloader, test_dloader]
            for ph, der in zip(phases, validation_dloaders):
                result[run][ph]['accuracy'] ,result[run][ph]['f1_macro'] = self.test(sgc, label_classifier, domain_classifier, der)
        
        best_run = runs[0]
        for run in runs:
            if(result[run]['test']['f1_macro'] > result[best_run]['test']['f1_macro']):
                best_run = run
        
        logger.info('====== Best hyper parameter: lr={} num_hidden={} K={} dropout={} lambda_={} alpha={} epsilon={}'.format(best_run.lr, best_run.num_hidden, best_run.K, best_run.dropout, best_run.lambda_, best_run.alpha, best_run.epsilon))
        return result[best_run]


class GNNAutoEncoderTrainApp:
    def __init__(self, eeg_dset, eye_dset, split_strategy, label_type):
        self.eeg_dset = eeg_dset
        self.eye_dset = eye_dset
        self.split_strategy = split_strategy
        self.label_type = label_type
        self.nworkers = os.cpu_count()
        self.maxepochs = args.maxepochs

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
    
    def get_runs(self):
        params = OrderedDict(lr_fe = eval(args.lr_fe),
                             lr_lc = eval(args.lr_lc),
                             num_hidden = eval(args.num_hidden),
                             K = eval(args.K),
                             dropout = eval(args.dropout),
                             epsilon = eval(args.epsilon),
                             reclosse = eval(args.reclosse),
                             lablosse = eval(args.lablosse))
        
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
    
    def getInitialEdgeWeightMatrix(self):
        adj = np.zeros((62, 62))
        xs, ys = np.tril_indices(62, -1)
        adj[xs, ys] = np.random.uniform(-1, 1, xs.shape[0])
        adj = adj + adj.T + np.identity(len(adj))
        return torch.tensor(adj, dtype=torch.float)

    def getModel(self, num_hidden, K, dropout):
        edge_weight = self.getInitialEdgeWeightMatrix()
        sgc = SGCFeatureExtractor(num_nodes=62,
                                  learn_edge_weight=True,
                                  edge_weight=edge_weight,
                                  num_features=self.eeg_dset.num_node_features,
                                  num_hidden=num_hidden,
                                  K=K)
        input_dim = 62 * self.eeg_dset.num_node_features + 115
        autoencoder = AutoEncoder(input_dim, dropout)
        lc = LabelClassifier(input_dim//4, dropout)
        
        if self.use_cuda:
            logger.info('Using cuda. Total {:d} devices.'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                sgc = DataParallel(sgc)
                autoencoder = nn.DataParallel(autoencoder)
                lc = nn.DataParallel(lc)
        else:
            logger.info('Using cpu.')

        return sgc.to(self.device), autoencoder.to(self.device), lc.to(self.device)

    def getSoftLabel(self, y, epsilon):
        """[summary]

        Args:
            y : (batch_size, )

        Returns:
            soft_y : (batch_size, 3)
        """
        batch_size = y.size(dim=0)
        soft_y = np.zeros((batch_size, 3))
        for i in range(batch_size):
            if y[i] == 0:
                # negative
                soft_y[i] = [1-2*epsilon/3, 2*epsilon/3, 0]
            elif y[i] == 1:
                # neutral
                soft_y[i] = [epsilon/3, 1-2*epsilon/3, epsilon/3]
            elif y[i] == 2:
                # positive
                soft_y[i] = [0, 2*epsilon/3, 1-2*epsilon/3]
        
        return torch.tensor(soft_y, dtype=torch.float, device=self.device)

    def test(self, sgc, autoencoder, lc, dloader):
        sgc.eval()
        autoencoder.eval()
        lc.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for graph_list, (eye_data, eye_label) in dloader:
                # (batch_size, 62*num_node_features)
                sgc_output = sgc(graph_list)
                # concate
                eye_data = eye_data.to(self.device)
                concatenated_feature = torch.cat((sgc_output, eye_data), dim=1).to(self.device)

                codes, _ = autoencoder(concatenated_feature)
                # (batch_size, 3)
                labcl_output = lc(codes)
                
                y_pred.append(torch.argmax(labcl_output, dim=1).detach().cpu().numpy())
                y = torch.cat([data.y for data in graph_list])
                y_true.append(y.detach().cpu().numpy())
            
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
        
        return acc, f1_macro

    def main(self):
        runs = self.get_runs()
        result = {}
        for run in runs:
            result[run] = {}
            for ph in phases:
                result[run][ph] = {}

        for run in runs:
            comment = ' GNNAutoEncoder lr_fe={} lr_lc={} num_hidden={} K={} dropout={} epsilon={} reclosse={}'.format(run.lr_fe, run.lr_lc, run.num_hidden, run.K, run.dropout, run.epsilon, run.reclosse)
            
            # dann_writer = SummaryWriter(comment=comment)

            logger.info('====== Hyper Parameter test:' + comment)
            start_outer = time.time()

            cross_validation_results = {}
            for ph in phases:
                cross_validation_results[ph] = {}
                for ind in indicators:
                    cross_validation_results[ph][ind] = []

            for train_idx, test_idx in self.split_strategy.split(self.eeg_dset, self.eeg_dset.data.y):
                train_eeg_dset = self.eeg_dset[train_idx]
                train_eeg_dset = train_eeg_dset.copy()
                test_eeg_dset = self.eeg_dset[test_idx]
                test_eeg_dset = test_eeg_dset.copy()

                train_eye_dset = Subset(eye_dset, train_idx)
                test_eye_dset = Subset(eye_dset, test_idx)

                train_eeg_dloader = DataListLoader(train_eeg_dset, batch_size=128, shuffle=False, drop_last=False)
                train_eye_dloader = DataLoader(train_eye_dset, batch_size=128, shuffle=False, drop_last=False)
                assert len(train_eeg_dloader) == len(train_eye_dloader)

                test_eeg_dloader = DataListLoader(test_eeg_dset, batch_size=128, shuffle=False, drop_last=False)
                test_eye_dloader = DataLoader(test_eye_dset, batch_size=128, shuffle=False, drop_last=False)
                assert len(test_eeg_dloader) == len(test_eye_dloader)

                sgc, autoencoder, lc = self.getModel(run.num_hidden, run.K, run.dropout)
                
                optim = torch.optim.Adam(
                    [{'params': list(sgc.parameters())+list(autoencoder.parameters()), 'lr': run.lr_fe},
                     {'params': lc.parameters(), 'lr': run.lr_lc}],
                    weight_decay=args.weight_decay)

                sgc.train()
                autoencoder.train()
                lc.train()
                for epoch in tqdm(range(1, self.maxepochs+1)):
                    loss, reconstruct_loss, label_pred_loss = 0, 0, 0
                    label_pred_accuracy = 0

                    n_batches = len(train_eeg_dloader)
                    for train_graph_list, (train_eye_data, train_eye_label) in tqdm(zip(train_eeg_dloader, train_eye_dloader), leave=False, total=n_batches):
                        # (batch_size, 62*num_node_features)
                        train_sgc_output = sgc(train_graph_list)
                        # concate
                        train_eye_data = train_eye_data.to(self.device)
                        concatenated_feature = torch.cat((train_sgc_output, train_eye_data), dim=1).to(self.device)

                        codes, decoded = autoencoder(concatenated_feature)
                        # (batch_size, 3)
                        train_labcl_output = lc(codes)

                        y = torch.cat([data.y for data in train_graph_list]).to(train_labcl_output.device)
                        if self.label_type == 'hard':
                            label_pred_loss_b = F.nll_loss(train_labcl_output, y)
                        elif self.label_type == 'soft':
                            soft_y = self.getSoftLabel(y, run.epsilon)
                            label_pred_loss_b = F.kl_div(train_labcl_output, soft_y, reduction='batchmean')
                        label_pred_loss += label_pred_loss_b

                        reconstruct_loss_b = F.mse_loss(decoded, concatenated_feature)
                        reconstruct_loss += reconstruct_loss_b

                        loss_b = run.lablosse * label_pred_loss_b + run.reclosse * reconstruct_loss_b
                        loss += loss_b

                        optim.zero_grad()
                        loss_b.backward()
                        optim.step()

                        label_pred_accuracy += (train_labcl_output.max(dim=1)[1] == y).float().mean().item()

                    epa_mean = label_pred_accuracy / n_batches  # expected to increase

                    # dann_writer.add_scalar('loss/total_loss', loss, epoch)
                    # dann_writer.add_scalar('accuracy/label_pred_accuracy', epa_mean, epoch)
                    # dann_writer.flush()
                    # dann_writer.close()

                    if epoch == 1 or epoch % 5 == 0:
                        logger.debug('Epoch {:6d}: train_label_pred_accuracy {:.4f}, loss {:.4f}, reconstruct_loss {:.4f}, label_pred_loss {:.4f}'.format(epoch, epa_mean, loss, reconstruct_loss, label_pred_loss))
                        tqdm.write('Epoch {:6d}: train_label_pred_accuracy {:.4f}, loss {:.4f}, reconstruct_loss {:.4f}, label_pred_loss {:.4f}'.format(epoch, epa_mean, loss, reconstruct_loss, label_pred_loss))
                    if epoch % 20 == 0:
                        # the iterator returned by zip() can not use twice
                        test_acc, test_f1 = self.test(sgc, autoencoder, lc, zip(test_eeg_dloader, test_eye_dloader))
                        logger.debug('====== Test: test_label_pred_accuracy {:.4f}, test_f1 {:.4f}'.format(test_acc, test_f1))
                        tqdm.write('====== Test: test_label_pred_accuracy {:.4f}, test_f1 {:.4f}'.format(test_acc, test_f1))
                        if test_f1 > 0.9:
                            break
                
                validation_dloaders = [zip(train_eeg_dloader, train_eye_dloader), zip(test_eeg_dloader, test_eye_dloader)]
                for ph, der in zip(phases, validation_dloaders):
                    acc, f1 = self.test(sgc, autoencoder, lc, der)
                    cross_validation_results[ph]['accuracy'].append(acc)
                    cross_validation_results[ph]['f1_macro'].append(f1)

            end_outer = time.time()
            dur_outer = end_outer - start_outer
            logger.info('For this run, train time: {:4d}min {:2d}sec'.format(int(dur_outer // 60), int(dur_outer % 60)))
            
            for ph in phases:
                for ind in indicators:
                    result[run][ph][ind] = np.array(cross_validation_results[ph][ind]).mean()
        
        best_run = runs[0]
        for run in runs:
            if(result[run]['test']['f1_macro'] > result[best_run]['test']['f1_macro']):
                best_run = run
        
        logger.info('====== Best hyper parameter: lr_fe={} lr_lc={} num_hidden={} K={} dropout={} epsilon={} reclosse={}'.format(best_run.lr_fe, best_run.lr_lc, best_run.num_hidden, best_run.K, best_run.dropout, best_run.epsilon, best_run.reclosse))
        return result[best_run]


class GNNTrainApp:
    def __init__(self, dset, split_strategy, label_type):
        self.dset = dset
        self.split_strategy = split_strategy
        self.label_type = label_type
        self.nworkers = os.cpu_count()
        self.maxepochs = args.maxepochs

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
    
    def get_runs(self):
        params = OrderedDict(lr = eval(args.lr),
                             num_hidden = eval(args.num_hidden),
                             K = eval(args.K),
                             dropout = eval(args.dropout),
                             epsilon = eval(args.epsilon))
        
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
    
    def getInitialEdgeWeightMatrix(self):
        adj = np.zeros((62, 62))
        xs, ys = np.tril_indices(62, -1)
        adj[xs, ys] = np.random.uniform(-1, 1, xs.shape[0])
        adj = adj + adj.T + np.identity(len(adj))
        return torch.tensor(adj, dtype=torch.float)

    def getModel(self, num_hidden, K, dropout):
        edge_weight = self.getInitialEdgeWeightMatrix()
        sgc = SGCFeatureExtractor(num_nodes=62,
                                  learn_edge_weight=True,
                                  edge_weight=edge_weight,
                                  num_features=self.dset.num_node_features,
                                  num_hidden=num_hidden,
                                  K=K)
        label_classifier = LabelClassifier(62*self.dset.num_node_features, dropout)
        
        if self.use_cuda:
            logger.info('Using cuda. Total {:d} devices.'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                sgc = DataParallel(sgc)
                label_classifier = nn.DataParallel(label_classifier)
        else:
            logger.info('Using cpu.')

        return sgc.to(self.device), label_classifier.to(self.device)

    def getSoftLabel(self, y, epsilon):
        """[summary]

        Args:
            y : (batch_size, )

        Returns:
            soft_y : (batch_size, 3)
        """
        batch_size = y.size(dim=0)
        soft_y = np.zeros((batch_size, 3))
        for i in range(batch_size):
            if y[i] == 0:
                # negative
                soft_y[i] = [1-2*epsilon/3, 2*epsilon/3, 0]
            elif y[i] == 1:
                # neutral
                soft_y[i] = [epsilon/3, 1-2*epsilon/3, epsilon/3]
            elif y[i] == 2:
                # positive
                soft_y[i] = [0, 2*epsilon/3, 1-2*epsilon/3]
        
        return torch.tensor(soft_y, dtype=torch.float, device=self.device)

    def test(self, sgc, label_classifier, dloader):
        sgc.eval()
        label_classifier.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for graph_list in dloader:
                # (batch_size, num_hidden)
                sgc_output = sgc(graph_list)
                # (batch_size, 3)
                labcl_output = label_classifier(sgc_output)
                
                y_pred.append(torch.argmax(labcl_output, dim=1).detach().cpu().numpy())
                y = torch.cat([data.y for data in graph_list]).to(labcl_output.device)
                y_true.append(y.detach().cpu().numpy())
            
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
        
        return acc, f1_macro

    def main(self):
        runs = self.get_runs()
        result = {}
        for run in runs:
            result[run] = {}
            for ph in phases:
                result[run][ph] = {}

        for run in runs:
            comment = ' RGNN lr={} num_hidden={} K={} dropout={} epsilon={}'.format(run.lr, run.num_hidden, run.K, run.dropout, run.epsilon)
            
            # dann_writer = SummaryWriter(comment=comment)

            logger.info('====== Hyper Parameter test: ' + comment)
            start_outer = time.time()

            cross_validation_results = {}
            for ph in phases:
                cross_validation_results[ph] = {}
                for ind in indicators:
                    cross_validation_results[ph][ind] = []

            for train_idx, test_idx in self.split_strategy.split(self.dset, self.dset.data.y):
                train_dset = self.dset[train_idx]
                train_dset = train_dset.copy()
                test_dset = self.dset[test_idx]
                test_dset = test_dset.copy()

                train_dloader = DataListLoader(train_dset, batch_size=16, shuffle=True, drop_last=True)
                test_dloader = DataListLoader(test_dset, batch_size=16, shuffle=True, drop_last=True)

                sgc, label_classifier = self.getModel(run.num_hidden, run.K, run.dropout)

                optim = torch.optim.Adam(list(sgc.parameters())+list(label_classifier.parameters()), lr=run.lr, weight_decay=args.weight_decay)

                sgc.train()
                label_classifier.train()
                for epoch in tqdm(range(1, self.maxepochs+1)):
                    loss = 0
                    label_pred_accuracy = 0

                    n_batches = len(train_dloader)
                    for train_graph_list in tqdm(train_dloader, leave=False, total=n_batches):
                        # (batch_size, num_hidden)
                        train_sgc_output = sgc(train_graph_list)
                        # (batch_size, 3)
                        train_labcl_output = label_classifier(train_sgc_output)

                        y = torch.cat([data.y for data in train_graph_list]).to(train_labcl_output.device)
                        if self.label_type == 'hard':
                            loss_b = F.nll_loss(train_labcl_output, y)
                        elif self.label_type == 'soft':
                            soft_y = self.getSoftLabel(y, run.epsilon)
                            loss_b = F.kl_div(train_labcl_output, soft_y, reduction='batchmean')
                        
                        loss += loss_b

                        optim.zero_grad()
                        loss_b.backward()
                        optim.step()

                        label_pred_accuracy += (train_labcl_output.max(dim=1)[1] == y).float().mean().item()

                    epa_mean = label_pred_accuracy / n_batches  # expected to increase

                    # dann_writer.add_scalar('loss/total_loss', loss, epoch)
                    # dann_writer.add_scalar('accuracy/label_pred_accuracy', epa_mean, epoch)
                    # dann_writer.flush()
                    # dann_writer.close()

                    if epoch == 1 or epoch % 10 == 0:
                        logger.debug('Epoch {:6d}: train_label_pred_accuracy {:.4f}, loss {:.4f}'.format(epoch, epa_mean, loss))
                        tqdm.write('Epoch {:6d}: train_label_pred_accuracy {:.4f}, loss {:.4f}'.format(epoch, epa_mean, loss))
                    if epoch % 50 == 0:
                        test_acc, test_f1 = self.test(sgc, label_classifier, test_dloader)
                        logger.debug('====== Test: test_label_pred_accuracy {:.4f}, test_f1 {:.4f}'.format(test_acc, test_f1))
                        tqdm.write('====== Test: test_label_pred_accuracy {:.4f}, test_f1 {:.4f}'.format(test_acc, test_f1))
                        if test_f1 > 0.9:
                            break
                        
                validation_dloaders = [train_dloader, test_dloader]
                for ph, der in zip(phases, validation_dloaders):
                    acc, f1 = self.test(sgc, label_classifier, der)
                    cross_validation_results[ph]['accuracy'].append(acc)
                    cross_validation_results[ph]['f1_macro'].append(f1)
            
            end_outer = time.time()
            dur_outer = end_outer - start_outer
            logger.info('For this run, train time: {:4d}min {:2d}sec'.format(int(dur_outer // 60), int(dur_outer % 60)))
            for ph in phases:
                for ind in indicators:
                    result[run][ph][ind] = np.array(cross_validation_results[ph][ind]).mean()
        
        best_run = runs[0]
        for run in runs:
            if(result[run]['test']['f1_macro'] > result[best_run]['test']['f1_macro']):
                best_run = run
        
        logger.info('====== Best hyper parameter: lr={} num_hidden={} K={} dropout={} epsilon={}'.format(best_run.lr, best_run.num_hidden, best_run.K, best_run.dropout, best_run.epsilon))
        return result[best_run]


class KNNTrainApp:
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

        hyperparams = [{'n_neighbors': [3, 5, 10, 15]}]
        # refit: after hp is determined, learn the best lp over the whole dataset, this is for prediction
        self.model = GridSearchCV(KNeighborsClassifier(),
                                  param_grid=hyperparams,
                                  scoring=['accuracy', 'f1_macro'],
                                  n_jobs=-1,
                                  refit='f1_macro',
                                  cv=self.split_strategy,
                                  verbose=1,
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
        
        # pic_result = []
        # for pp in range(90):
        #     pic_result.append(self.model.cv_results_['split{}_test_accuracy'.format(pp)][idx])

        # train_cm_display = plot_confusion_matrix(model, x_train, y_train, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # train_cm_display.figure_.suptitle('{}_{}_{}: Train set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # train_cm_display.figure_.savefig('./figs/{}_{}_{}_train.png'.format(self.feature, self.freq_band, model_name))

        # test_cm_display = plot_confusion_matrix(model, x_test, y_test, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # test_cm_display.figure_.suptitle('{}_{}_{}: Test set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # test_cm_display.figure_.savefig('./figs/{}_{}_{}_test.png'.format(self.feature, self.freq_band, model_name))
        # plt.close('all')

        return result
        

class GPTrainApp:
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

        hyperparams = [{'kernel': [1.0*RBF(1.0), 1.0*RBF(0.1), 1.0*RBF(10)]}]
        # refit: after hp is determined, learn the best lp over the whole dataset, this is for prediction
        self.model = GridSearchCV(GaussianProcessClassifier(),
                                  param_grid=hyperparams,
                                  scoring=['accuracy', 'f1_macro'],
                                  n_jobs=-1,
                                  refit='f1_macro',
                                  cv=self.split_strategy,
                                  verbose=1,
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
        
        # pic_result = []
        # for pp in range(90):
        #     pic_result.append(self.model.cv_results_['split{}_test_accuracy'.format(pp)][idx])

        # train_cm_display = plot_confusion_matrix(model, x_train, y_train, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # train_cm_display.figure_.suptitle('{}_{}_{}: Train set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # train_cm_display.figure_.savefig('./figs/{}_{}_{}_train.png'.format(self.feature, self.freq_band, model_name))

        # test_cm_display = plot_confusion_matrix(model, x_test, y_test, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # test_cm_display.figure_.suptitle('{}_{}_{}: Test set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # test_cm_display.figure_.savefig('./figs/{}_{}_{}_test.png'.format(self.feature, self.freq_band, model_name))
        # plt.close('all')

        return result
        

class DTTrainApp:
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

        hyperparams = [{'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 15, 20]}]
        # refit: after hp is determined, learn the best lp over the whole dataset, this is for prediction
        self.model = GridSearchCV(DecisionTreeClassifier(),
                                  param_grid=hyperparams,
                                  scoring=['accuracy', 'f1_macro'],
                                  n_jobs=-1,
                                  refit='f1_macro',
                                  cv=self.split_strategy,
                                  verbose=1,
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
        
        # pic_result = []
        # for pp in range(90):
        #     pic_result.append(self.model.cv_results_['split{}_test_accuracy'.format(pp)][idx])

        # train_cm_display = plot_confusion_matrix(model, x_train, y_train, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # train_cm_display.figure_.suptitle('{}_{}_{}: Train set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # train_cm_display.figure_.savefig('./figs/{}_{}_{}_train.png'.format(self.feature, self.freq_band, model_name))

        # test_cm_display = plot_confusion_matrix(model, x_test, y_test, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # test_cm_display.figure_.suptitle('{}_{}_{}: Test set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # test_cm_display.figure_.savefig('./figs/{}_{}_{}_test.png'.format(self.feature, self.freq_band, model_name))
        # plt.close('all')

        return result
        

class RFTrainApp:
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

        hyperparams = [{'n_estimators': [10, 100, 1000], 'criterion': ['gini', 'entropy'], 'max_depth':[5, 10, 15, 20]}]
        # refit: after hp is determined, learn the best lp over the whole dataset, this is for prediction
        self.model = GridSearchCV(RandomForestClassifier(),
                                  param_grid=hyperparams,
                                  scoring=['accuracy', 'f1_macro'],
                                  n_jobs=-1,
                                  refit='f1_macro',
                                  cv=self.split_strategy,
                                  verbose=1,
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
        
        # pic_result = []
        # for pp in range(90):
        #     pic_result.append(self.model.cv_results_['split{}_test_accuracy'.format(pp)][idx])

        # train_cm_display = plot_confusion_matrix(model, x_train, y_train, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # train_cm_display.figure_.suptitle('{}_{}_{}: Train set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # train_cm_display.figure_.savefig('./figs/{}_{}_{}_train.png'.format(self.feature, self.freq_band, model_name))

        # test_cm_display = plot_confusion_matrix(model, x_test, y_test, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # test_cm_display.figure_.suptitle('{}_{}_{}: Test set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # test_cm_display.figure_.savefig('./figs/{}_{}_{}_test.png'.format(self.feature, self.freq_band, model_name))
        # plt.close('all')

        return result
        

class ABTrainApp:
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

        hyperparams = [{'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 1, 10]}]
        # refit: after hp is determined, learn the best lp over the whole dataset, this is for prediction
        self.model = GridSearchCV(AdaBoostClassifier(),
                                  param_grid=hyperparams,
                                  scoring=['accuracy', 'f1_macro'],
                                  n_jobs=-1,
                                  refit='f1_macro',
                                  cv=self.split_strategy,
                                  verbose=1,
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
        
        # pic_result = []
        # for pp in range(90):
        #     pic_result.append(self.model.cv_results_['split{}_test_accuracy'.format(pp)][idx])

        # train_cm_display = plot_confusion_matrix(model, x_train, y_train, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # train_cm_display.figure_.suptitle('{}_{}_{}: Train set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # train_cm_display.figure_.savefig('./figs/{}_{}_{}_train.png'.format(self.feature, self.freq_band, model_name))

        # test_cm_display = plot_confusion_matrix(model, x_test, y_test, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # test_cm_display.figure_.suptitle('{}_{}_{}: Test set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # test_cm_display.figure_.savefig('./figs/{}_{}_{}_test.png'.format(self.feature, self.freq_band, model_name))
        # plt.close('all')

        return result
        

class QDATrainApp:
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

        hyperparams = [{'reg_param': [0.0, 0.1, 0.5]}]
        # refit: after hp is determined, learn the best lp over the whole dataset, this is for prediction
        self.model = GridSearchCV(QuadraticDiscriminantAnalysis(),
                                  param_grid=hyperparams,
                                  scoring=['accuracy', 'f1_macro'],
                                  n_jobs=-1,
                                  refit='f1_macro',
                                  cv=self.split_strategy,
                                  verbose=1,
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
        
        # pic_result = []
        # for pp in range(90):
        #     pic_result.append(self.model.cv_results_['split{}_test_accuracy'.format(pp)][idx])

        # train_cm_display = plot_confusion_matrix(model, x_train, y_train, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # train_cm_display.figure_.suptitle('{}_{}_{}: Train set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # train_cm_display.figure_.savefig('./figs/{}_{}_{}_train.png'.format(self.feature, self.freq_band, model_name))

        # test_cm_display = plot_confusion_matrix(model, x_test, y_test, normalize='true', display_labels=['Negative', 'Neutral', 'Positive'], cmap='Blues')
        # test_cm_display.figure_.suptitle('{}_{}_{}: Test set confusion matrix'.format(self.feature, self.freq_band, model_name))
        # test_cm_display.figure_.savefig('./figs/{}_{}_{}_test.png'.format(self.feature, self.freq_band, model_name))
        # plt.close('all')

        return result


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == '__main__':
    freq_bands = eval(args.freqbands)
    nslices_per_img = int(30 / args.slice_length)

    logger.info('#'*50 + 'Experiment: ' + str(args.exp))
    feature = args.feature
    logger.info('#'*30 + 'Feature: ' + feature)
    for freq in freq_bands:
        logger.info('#'*20 + 'Freq. Band: ' + freq)
        exp_result = {}

        if args.exp == 'subj_dep':
            for subject in subjects:
                logger.info('#'*10 + 'Train on ' + subject)

                if args.which == 0:
                    model_name = 'SVM'
                    logger.info('>>> Model: SVM')
                    dset = EEGEyeDataset(args.slice_length, feature, freq, [subject])
                    nncases = dset.data.size(dim=0)
                    nncases_per_class = int(nncases / 3)
                    nnpics_per_class = int(nncases_per_class / 6)
                    nnpics_per_validation = int(nnpics_per_class / 6)
                    nnpics_arr = [nnpics_per_validation] * 5 + [nnpics_per_class-nnpics_per_validation*5]
                    # (nncases, )
                    test_fold = []
                    for class_idx in range(3):
                        for val_idx, nnpics_a in enumerate(nnpics_arr):
                            test_fold += [val_idx] * nnpics_a * 6

                    # split_strategy = StratifiedKFold(n_splits=5)
                    split_strategy = PredefinedSplit(test_fold=test_fold)
                    result = SVMTrainApp(dset, split_strategy).main()
                elif args.which == 1:
                    model_name = 'GNN'
                    logger.info('>>> Model: GNN')
                    graph_data_dir = graph_data_root_dir + 'subjdep_{}s_{}_{}_{}_{}'.format(args.slice_length, feature, freq, subject, f"{datetime.datetime.now():%Y-%m-%d-%H-%M}")
                    if not os.path.exists(graph_data_dir):
                        os.makedirs(graph_data_dir)
                    eeg_dset = EEGEyeArtGraphDataset(args.slice_length, feature, freq, [subject], graph_data_dir)
                    eye_dset = EEGEyeArtDataset(args.slice_length, feature, freq, [subject])

                    nncases = eye_dset.data.size(dim=0)
                    nncases_per_class = int(nncases / 3)
                    nnpics_per_class = int(nncases_per_class / 6)
                    nnpics_per_validation = int(nnpics_per_class / 6)
                    nnpics_arr = [nnpics_per_validation] * 5 + [nnpics_per_class-nnpics_per_validation*5]
                    # (nncases, )
                    test_fold = []
                    for class_idx in range(3):
                        for val_idx, nnpics_a in enumerate(nnpics_arr):
                            test_fold += [val_idx] * nnpics_a * 6

                    # split_strategy = StratifiedKFold(n_splits=5)
                    split_strategy = PredefinedSplit(test_fold=test_fold)

                    result = GNNAutoEncoderTrainApp(eeg_dset, eye_dset, split_strategy, args.label_type).main()
                
                exp_result[subject] = result
        elif args.exp == 'subj_indep':
            for subject in subjects:
                logger.info('#'*10 + 'Target on ' + subject)
                
                if args.which == 0:
                    model_name = 'SVM'
                    logger.info('>>> Model: SVM')
                    dset = ArtDataset(args.slice_length, feature, freq, subjects, exclude_imgs, oversample=True)
                    subj_idx = subjects.index(subject)
                    nimgs_total = sum(nimgs_each_subject)
                    test_fold = np.empty(nimgs_total*nslices_per_img, dtype=np.int8)
                    test_fold.fill(-1)
                    test_fold[sum(nimgs_each_subject[:subj_idx])*nslices_per_img: sum(nimgs_each_subject[:subj_idx+1])*nslices_per_img] = 0
                    split_strategy = PredefinedSplit(test_fold)
                    result = SVMTrainApp(dset, split_strategy).main()
                elif args.which == 1:
                    model_name = 'GNN'
                    logger.info('>>> Model: GNN')
                    train_subjects = subjects.copy()
                    train_subjects.remove(subject)
                    test_subjects = [subject]
                    train_graph_data_dir = graph_data_root_dir + 'subjindep_train_{}s_{}_{}_{}_{}'.format(args.slice_length, feature, freq, subject, f"{datetime.datetime.now():%Y-%m-%d-%H-%M}")
                    test_graph_data_dir = graph_data_root_dir + 'subjindep_test_{}s_{}_{}_{}_{}'.format(args.slice_length, feature, freq, subject, f"{datetime.datetime.now():%Y-%m-%d-%H-%M}")
                    if not os.path.exists(train_graph_data_dir):
                        os.makedirs(train_graph_data_dir)
                    if not os.path.exists(test_graph_data_dir):
                        os.makedirs(test_graph_data_dir)
                    # train_dset = ArtGraphDataset(args.slice_length, feature, freq, train_subjects, exclude_imgs, train_graph_data_dir, oversample=True)
                    # test_dset = ArtGraphDataset(args.slice_length, feature, freq, test_subjects, exclude_imgs, test_graph_data_dir)
                    # result = GNNCrossSubjTrainApp(train_dset, test_dset, args.label_type).main()
                    train_eeg_dset = EEGEyeArtGraphDataset(args.slice_length, feature, freq, train_subjects, train_graph_data_dir)
                    train_eye_dset = EEGEyeArtDataset(args.slice_length, feature, freq, train_subjects)
                    test_eeg_dset = EEGEyeArtGraphDataset(args.slice_length, feature, freq, test_subjects, test_graph_data_dir)
                    test_eye_dset = EEGEyeArtDataset(args.slice_length, feature, freq, test_subjects)
                    result = GNNAutoEncoderTrainApp(train_eeg_dset, train_eye_dset, test_eeg_dset, test_eye_dset, args.label_type).main()
                
                exp_result[subject] = result

        logger.info('Result:')
        logger.info(str(exp_result))

        subj_train_accs = np.array([round(exp_result[subj]['train']['accuracy'], 4) for subj in subjects])
        subj_train_f1s = np.array([round(exp_result[subj]['train']['f1_macro'], 4) for subj in subjects])
        subj_test_accs = np.array([round(exp_result[subj]['test']['accuracy'], 4) for subj in subjects])
        subj_test_f1s = np.array([round(exp_result[subj]['test']['f1_macro'], 4) for subj in subjects])

        plt.style.use('seaborn')
        x = np.arange(0, (len(subjects)-1)*2.5+1, 2.5)  # the label locations
        width = 1.0  # the width of the bars
        fig, ax = plt.subplots(figsize=(35, 7.8))
        acc_train_rect = ax.bar(x - width/2, subj_train_accs, width, label='Train/Acc', fill=False, ls='--')
        acc_test_rect = ax.bar(x - width/2, subj_test_accs, width, label='Test/Acc')
        f1_train_rect = ax.bar(x + width/2, subj_train_f1s, width, label='Train/F1', fill=False, ls='--')
        f1_test_rect = ax.bar(x + width/2, subj_test_f1s, width, label='Test/F1')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Subjects')
        ax.set_title('{}_{}_{}_{}'.format(args.exp, feature, freq, model_name), pad=36)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects)
        ax.set_ylim(0.0, 1.0)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.legend([acc_train_rect, acc_test_rect, f1_test_rect], ['Train', 'Test/Acc.', 'Test/F1.'], loc='center left', bbox_to_anchor=(1, 0.5))
        ax.bar_label(acc_train_rect, padding=3)
        ax.bar_label(acc_test_rect, padding=3)
        ax.bar_label(f1_train_rect, padding=3)
        ax.bar_label(f1_test_rect, padding=3)
        fig.savefig('./figs20220621/{}_{}_{}s_{}_{}.png'.format(args.exp, feature, args.slice_length, freq, model_name))
        plt.close('all')

        logger.info('====Train:\nacc: {:.4f}/{:.4f}\nf1: {:.4f}/{:.4f}'.format(subj_train_accs.mean(), subj_train_accs.std(), subj_train_f1s.mean(), subj_train_f1s.std()))
        logger.info('====Test:\nacc: {:.4f}/{:.4f}\nf1: {:.4f}/{:.4f}'.format(subj_test_accs.mean(), subj_test_accs.std(), subj_test_f1s.mean(), subj_test_f1s.std()))

        # plt.style.use('default')
        # dta = np.array([pic_results[sj] for sj in subjects])
        # imgs = list(range(90))
        # fig2, ax2 = plt.subplots(figsize=(30.8, 4.8))
        # im, cbar = heatmap(dta, subjects, imgs, ax=ax2, cmap="YlGn", cbarlabel="acc")
        # # texts = annotate_heatmap(im, valfmt="{x:.4f}")
        # fig2.savefig('./figs_cao/{}_{}_{}_{}_picresult.png'.format(args.exp, feature, freq, model_name))
        # plt.close('all')
