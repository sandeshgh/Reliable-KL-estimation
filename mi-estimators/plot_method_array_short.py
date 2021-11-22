import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import mi_schedule

def find_name(name):
    if 'smile_' in name:
        clip = name.split('_')[-1]
        return f'SMILE ($\\tau = {clip}$)'
    else:
        return {
            'infonce': 'CPC',
            'js': 'JS',
            'nwj': 'NWJ',
            'flow': 'GM (Flow)',
            'smile': 'SMILE ($\\tau = \\infty$)'
        }[name]

def find_legend(label):
    return {'concat': 'Joint critic', 'separable': 'Separable critic'}[label]

# file ='Lip_RKHS_mi_gauss.pkl'
# filename = 'Lip_features_mi_lam_0.001_lip_g2_lip_5_D_500_gamma_5_gauss'#'Lip_RKHS_mi_lip_5_cubic'
filename = 'Smile_results_gauss_10k'
estimator = 'lip_features' #'lip_rkhs'
critic = 'concat_lip_features'#'concat_lip_rkhs'
x_limit = 10000
y_limit = 14

file =filename +'.pkl'
with open(file, 'rb') as f:
    mi_numpys = pickle.load(f)

ncols = 5
nrows = 2
EMA_SPAN = 200
fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axs = np.ravel(axs)

mi_true = mi_schedule(x_limit)

for i, estimator in enumerate(['infonce', 'nwj']):
    key = f'{estimator}'
    plt.sca(axs[i])
    plt.title(find_name(key), fontsize=18)
    for net in ['concat']:
        mis = mi_numpys[net][key]
        p1 = plt.plot(mis, alpha=0.3)[0]
        mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
        plt.plot(mis_smooth, c=p1.get_color(), label=find_legend(net))
    plt.ylim(0, y_limit)
    plt.xlim(0, x_limit)
    axs[i].set_xticks(np.linspace(0, x_limit, 6))
    plt.plot(mi_true, color='k', label='True MI')
    if i == 0:
        plt.ylabel('MI (nats)')
        plt.xlabel('Steps')
        plt.axhline(np.log(64), color='k', ls='--', label='log(bs)')
        plt.legend()

estimator = 'smile'
for i, clip in enumerate([1.0, 5.0, None]):
    if clip is None:
        key = estimator
    else:
        key = f'{estimator}_{clip}'

    plt.sca(axs[i+2])
    plt.title(find_name(key), fontsize=18)
    for net in ['concat']:
        mis = mi_numpys[net][key]
        EMA_SPAN = 200
        p1 = plt.plot(mis, alpha=0.3)[0]
        mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
        plt.plot(mis_smooth, c=p1.get_color(), label=find_legend(net))
    plt.plot(mi_true, color='k', label='True MI')
    axs[i+2].set_xticks(np.linspace(0, x_limit, 6))
    plt.ylim(0, y_limit)
    plt.xlim(0, x_limit)

# our method
lambd_array =[0.0, 1e-05, 0.001, 0.1, 1.0]
for j in range(len(lambd_array)):
    lambd =lambd_array[j]
    filename_ours  = 'Lip_features_mi_lam_'+str(lambd)+'_lip_g5_lip_5_D_500_gamma_5_gauss10k'#'Lip_RKHS_mi_lip_5_cubic'
    estimator = 'lip_features' #'lip_rkhs'
    critic = 'concat_lip_features'#'concat_lip_rkhs'

    file =filename_ours +'.pkl'
    with open(file, 'rb') as f:
        data = pickle.load(f)

    plt.sca(axs[j+5])
    plt.title('Ours(lambda = {lam})'.format(lam = lambd), fontsize=18)
    kl = data[critic][estimator]
    EMA_SPAN = 200
    p1 = plt.plot(kl, alpha=0.3)[0]
    mis_smooth = pd.Series(kl).ewm(span=EMA_SPAN).mean()
    plt.plot(mis_smooth, c=p1.get_color(), label='Ours')
    plt.plot(mi_true, color='k', label='True MI')
    if j ==0:
        plt.ylabel('Lips =5, Lip-g = 5')
    axs[j+5].set_xticks(np.linspace(0,x_limit,6))
    plt.ylim(0, y_limit)
    plt.xlim(0, x_limit)



plt.gcf().tight_layout()
plt.savefig('smile_g_5_results_gauss_all_row_5k.png')
plt.close()