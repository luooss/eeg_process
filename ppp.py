import numpy as np
import matplotlib
import matplotlib.pyplot as plt


subjects = ['zuoyaxi', 'zhangliuxin', 'zhaosijia', 'zhuyangxiangru', 'pujizhou', 'chenbingliang', 'mengdong', 'zhengxucen',
            'hexingtao', 'wanghuiling', 'panshuyi', 'wangsifan', 'zhaochangquan', 'wuxiangyu', 'xiajingtao', 'liujiaxin',
            'wangyanchu', 'liyizhou', 'weifenfen', 'chengyuting', 'chenjiajing', 'matianfang', 'liuledian', 'zuogangao', 'feicheng', 'xuyutong']

plt.style.use('seaborn')
x = np.arange(0, (26-1)*2.5+1, 2.5)  # the label locations
width = 1.0  # the width of the bars
fig, ax = plt.subplots(figsize=(35, 7.8))
subj_train_accs = np.linspace(0, 1, 26)
subj_test_accs = np.linspace(0, 1, 26)
subj_train_f1s = np.linspace(0, 1, 26)
subj_test_f1s = np.linspace(0, 1, 26)
acc_train_rect = ax.bar(x - width/2, subj_train_accs, width, label='Train/Acc', fill=False, ls='--')
acc_test_rect = ax.bar(x - width/2, subj_test_accs, width, label='Test/Acc')
f1_train_rect = ax.bar(x + width/2, subj_train_f1s, width, label='Train/F1', fill=False, ls='--')
f1_test_rect = ax.bar(x + width/2, subj_test_f1s, width, label='Test/F1')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Subjects')
ax.set_title('test', pad=36)
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
fig.savefig('./t.png')