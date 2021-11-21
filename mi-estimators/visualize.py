import pickle
import matplotlib.pyplot as plt

# file ='Lip_RKHS_mi_gauss.pkl'
filename = 'Lip_features_mi_lam_0.001_lip_g2_lip_5_D_500_gamma_5_gauss'#'Lip_RKHS_mi_lip_5_cubic'
estimator = 'lip_features' #'lip_rkhs'
critic = 'concat_lip_features'#'concat_lip_rkhs'

file =filename +'.pkl'
with open(file, 'rb') as f:
    data = pickle.load(f)

kl = data[critic][estimator]
plt.plot(kl)
plt.ylim([0,12])
plt.savefig(filename+ '.png')
plt.close()

