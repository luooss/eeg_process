## Result

```
################################################## Experiment:  subj_dependent
############################## Feature:  psd
#################### Freq. Band:  all
########## Train on  chenyi
>>> Model: SVM
The best hyper parameters: {'C': 100000.0, 'gamma': 0.0001, 'kernel': 'rbf'}
########## Train on  huangwenjing
>>> Model: SVM
The best hyper parameters: {'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}
########## Train on  huangxingbao
>>> Model: SVM
The best hyper parameters: {'C': 100.0, 'gamma': 0.001, 'kernel': 'rbf'}
########## Train on  huatong
>>> Model: SVM
The best hyper parameters: {'C': 10.0, 'gamma': 0.001, 'kernel': 'rbf'}
########## Train on  wuwenrui
>>> Model: SVM
The best hyper parameters: {'C': 10000000.0, 'gamma': 1e-06, 'kernel': 'rbf'}
########## Train on  yinhao
>>> Model: SVM
The best hyper parameters: {'C': 1000.0, 'gamma': 0.001, 'kernel': 'rbf'}
Result:
{'chenyi': {'test': {'accuracy': {'mean': 0.45999999999999996,       
                                  'std': 0.06110100926607786},       
                     'f1_macro': {'mean': 0.4516314715843758,        
                                  'std': 0.06726801253014318}},      
            'train': {'accuracy': {'mean': 0.9979166666666666,       
                                   'std': 0.0017677669529663747},    
                      'f1_macro': {'mean': 0.998158899730002,        
                                   'std': 0.0015621961959533298}}},  
 'huangwenjing': {'test': {'accuracy': {'mean': 0.4833333333333333,  
                                        'std': 0.04966554808583782}, 
                           'f1_macro': {'mean': 0.4248635810692923,  
                                        'std': 0.0507569784044742}}, 
                  'train': {'accuracy': {'mean': 1.0, 'std': 0.0},   
                            'f1_macro': {'mean': 1.0, 'std': 0.0}}}, 
 'huangxingbao': {'test': {'accuracy': {'mean': 0.5422222222222223,  
                                        'std': 0.040490815907307985},
                           'f1_macro': {'mean': 0.5347388302172852,  
                                        'std': 0.04143830211896707}},
                  'train': {'accuracy': {'mean': 1.0, 'std': 0.0},   
                            'f1_macro': {'mean': 1.0, 'std': 0.0}}}, 
 'huatong': {'test': {'accuracy': {'mean': 0.41111111111111115,
                                   'std': 0.04605418172458379},
                      'f1_macro': {'mean': 0.40601655302181827,
                                   'std': 0.04433167578831999}},
             'train': {'accuracy': {'mean': 0.8125,
                                    'std': 0.007046472718869897},
                       'f1_macro': {'mean': 0.8111382888197185,
                                    'std': 0.00740559385952236}}},
 'wuwenrui': {'test': {'accuracy': {'mean': 0.4311111111111111,
                                    'std': 0.052587375849774354},
                       'f1_macro': {'mean': 0.4240539719844986,
                                    'std': 0.053216048112240996}},
              'train': {'accuracy': {'mean': 0.8080555555555555,
                                     'std': 0.00911178859269818},
                        'f1_macro': {'mean': 0.8081025803225358,
                                     'std': 0.009199564841310381}}},
 'yinhao': {'test': {'accuracy': {'mean': 0.46222222222222226,
                                  'std': 0.02779999111821511},
                     'f1_macro': {'mean': 0.439698498170755,
                                  'std': 0.02861136526258109}},
            'train': {'accuracy': {'mean': 1.0, 'std': 0.0},
                      'f1_macro': {'mean': 1.0, 'std': 0.0}}}}
====Train:
acc: 0.9364/0.0892
f1: 0.9362/0.0895
====Test:
acc: 0.4650/0.0416
f1: 0.4468/0.0418
#################### Freq. Band:  delta
########## Train on  chenyi
>>> Model: SVM
The best hyper parameters: {'C': 100.0, 'gamma': 0.01, 'kernel': 'rbf'}
########## Train on  huangwenjing
>>> Model: SVM
The best hyper parameters: {'C': 1000.0, 'gamma': 0.001, 'kernel': 'rbf'}
########## Train on  huangxingbao
>>> Model: SVM
The best hyper parameters: {'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}
########## Train on  huatong
>>> Model: SVM
The best hyper parameters: {'C': 10000.0, 'gamma': 0.0001, 'kernel': 'rbf'}
########## Train on  wuwenrui
>>> Model: SVM
The best hyper parameters: {'C': 10000.0, 'gamma': 1e-05, 'kernel': 'rbf'}
########## Train on  yinhao
>>> Model: SVM
The best hyper parameters: {'C': 10.0, 'gamma': 0.001, 'kernel': 'rbf'}
Result:
{'chenyi': {'test': {'accuracy': {'mean': 0.4933333333333333,
                                  'std': 0.03559026084010436},
                     'f1_macro': {'mean': 0.4654840329725828,
                                  'std': 0.03291968995670233}},
            'train': {'accuracy': {'mean': 1.0, 'std': 0.0},
                      'f1_macro': {'mean': 1.0, 'std': 0.0}}},
 'huangwenjing': {'test': {'accuracy': {'mean': 0.5122222222222222,
                                        'std': 0.046134532204503746},
                           'f1_macro': {'mean': 0.4767001283290806,
                                        'std': 0.049463810697905695}},
                  'train': {'accuracy': {'mean': 1.0, 'std': 0.0},
                            'f1_macro': {'mean': 1.0, 'std': 0.0}}},
 'huangxingbao': {'test': {'accuracy': {'mean': 0.6011111111111112,
                                        'std': 0.03603838831161523},
                           'f1_macro': {'mean': 0.5984298010074619,
                                        'std': 0.03634145086440037}},
                  'train': {'accuracy': {'mean': 1.0, 'std': 0.0},
                            'f1_macro': {'mean': 1.0, 'std': 0.0}}},
 'huatong': {'test': {'accuracy': {'mean': 0.44999999999999996,
                                   'std': 0.06815016100086956},
                      'f1_macro': {'mean': 0.44478303524073975,
                                   'std': 0.06949723629547999}},
             'train': {'accuracy': {'mean': 1.0, 'std': 0.0},
                       'f1_macro': {'mean': 1.0, 'std': 0.0}}},
 'wuwenrui': {'test': {'accuracy': {'mean': 0.45222222222222225,
                                    'std': 0.04103596736137643},
                       'f1_macro': {'mean': 0.4468072834595461,
                                    'std': 0.04121392424995519}},
              'train': {'accuracy': {'mean': 0.8291666666666667,
                                     'std': 0.004370036867375641},
                        'f1_macro': {'mean': 0.8286276853459087,
                                     'std': 0.004376759684517969}}},
 'yinhao': {'test': {'accuracy': {'mean': 0.508888888888889,
                                  'std': 0.04433319409613336},
                     'f1_macro': {'mean': 0.4880330435818738,
                                  'std': 0.045118509525838735}},
            'train': {'accuracy': {'mean': 0.8970833333333333,
                                   'std': 0.007312470322826772},
                      'f1_macro': {'mean': 0.8919343337416,
                                   'std': 0.0073139585015860225}}}}
====Train:
acc: 0.9544/0.0674
f1: 0.9534/0.0684
====Test:
acc: 0.5030/0.0504
f1: 0.4867/0.0522
#################### Freq. Band:  theta
########## Train on  chenyi
>>> Model: SVM
The best hyper parameters: {'C': 10000000000.0, 'gamma': 1e-07, 'kernel': 'rbf'}
########## Train on  huangwenjing
>>> Model: SVM
The best hyper parameters: {'C': 100000.0, 'gamma': 0.0001, 'kernel': 'rbf'}
########## Train on  huangxingbao
>>> Model: SVM
The best hyper parameters: {'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}
########## Train on  huatong
>>> Model: SVM
The best hyper parameters: {'C': 1000.0, 'gamma': 0.001, 'kernel': 'rbf'}
########## Train on  wuwenrui
>>> Model: SVM
The best hyper parameters: {'C': 10000000.0, 'gamma': 1e-07, 'kernel': 'rbf'}
########## Train on  yinhao
>>> Model: SVM
The best hyper parameters: {'C': 1000000.0, 'gamma': 1e-08, 'kernel': 'rbf'}
Result:
{'chenyi': {'test': {'accuracy': {'mean': 0.4322222222222222,
                                  'std': 0.04340449996583246},
                     'f1_macro': {'mean': 0.42037545938084403,
                                  'std': 0.03853253174215101}},
            'train': {'accuracy': {'mean': 0.67125,
                                   'std': 0.011740835669671129},
                      'f1_macro': {'mean': 0.6688895691725012,
                                   'std': 0.014277854915040493}}},
 'huangwenjing': {'test': {'accuracy': {'mean': 0.4855555555555556,
                                        'std': 0.039189315752329784},
                           'f1_macro': {'mean': 0.46096944090340103,
                                        'std': 0.04386611792491434}},
                  'train': {'accuracy': {'mean': 1.0, 'std': 0.0},
                            'f1_macro': {'mean': 1.0, 'std': 0.0}}},
 'huangxingbao': {'test': {'accuracy': {'mean': 0.5744444444444443,
                                        'std': 0.045973690812393775},
                           'f1_macro': {'mean': 0.5709216594632216,
                                        'std': 0.046914735981849476}},
                  'train': {'accuracy': {'mean': 1.0, 'std': 0.0},
                            'f1_macro': {'mean': 1.0, 'std': 0.0}}},
 'huatong': {'test': {'accuracy': {'mean': 0.4222222222222222,
                                   'std': 0.04732342097565967},
                      'f1_macro': {'mean': 0.41769091401078323,
                                   'std': 0.04712251407836751}},
             'train': {'accuracy': {'mean': 1.0, 'std': 0.0},
                       'f1_macro': {'mean': 1.0, 'std': 0.0}}},
 'wuwenrui': {'test': {'accuracy': {'mean': 0.42888888888888893,
                                    'std': 0.03446236093171295},
                       'f1_macro': {'mean': 0.4205349087171268,
                                    'std': 0.04009150376934624}},
              'train': {'accuracy': {'mean': 0.7704166666666666,
                                     'std': 0.013629522694830096},
                        'f1_macro': {'mean': 0.770436518496479,
                                     'std': 0.013733454871087725}}},
 'yinhao': {'test': {'accuracy': {'mean': 0.49, 'std': 0.04618802153517008},
                     'f1_macro': {'mean': 0.47385805244263335,
                                  'std': 0.05093828081948937}},
            'train': {'accuracy': {'mean': 0.7109722222222222,
                                   'std': 0.0073781249918296885},
                      'f1_macro': {'mean': 0.7010450650634016,
                                   'std': 0.00743757127796391}}}}
====Train:
acc: 0.8588/0.1441
f1: 0.8567/0.1464
====Test:
acc: 0.4722/0.0531
f1: 0.4607/0.0539
#################### Freq. Band:  alpha
########## Train on  chenyi
>>> Model: SVM
The best hyper parameters: {'C': 0.1, 'kernel': 'linear'}
########## Train on  huangwenjing
>>> Model: SVM
The best hyper parameters: {'C': 1000000.0, 'gamma': 1e-07, 'kernel': 'rbf'}
########## Train on  huangxingbao
>>> Model: SVM
The best hyper parameters: {'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf'}
########## Train on  huatong
>>> Model: SVM
The best hyper parameters: {'C': 1000000.0, 'gamma': 1e-08, 'kernel': 'rbf'}
########## Train on  wuwenrui
>>> Model: SVM
The best hyper parameters: {'C': 10000.0, 'gamma': 1e-06, 'kernel': 'rbf'}
########## Train on  yinhao
```