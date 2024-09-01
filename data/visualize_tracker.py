from cProfile import label
import sys
sys.path.append('/media/max/TOSHIBA EXT/ONERA_SONDRA/algos/contrastive_learning')
from pipeline.storage.state import StateStorageFile, StateStorageBase
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

def plot_all(dic):
    for key in dic._state:
        l = dic.get_value(key)
        if 'step' not in locals():
            step = list(range(1,len(l)+1)) 
        plt.plot(step,l,label=key)
        plt.legend()
        plt.show()

def plot_param(dic:StateStorageBase,key:str or list,epoch=None,step_per_epoch=10):

    if not dic.has_key(key):
        print('Parameter {} not in dic'.format(key))
        return False
    assert dic.has_key('n_step')
    n_step = dic.get_value('n_step')


    l = dic.get_value(key)
    step = np.array(list(range(len(l))))/n_step
    step_sz = int(n_step/step_per_epoch)
    if epoch == None:
            l = l[::step_sz]
            step = step[::step_sz]
    else :
        l = l[epoch[0]:epoch[1]:step_sz]
        step = step[epoch[0]:epoch[1]:step_sz]

    plt.plot(step,l,label=key)
    plt.legend()
    plt.show()

if __name__ == "__main__":

    param = ['loss mean regularization','loss cross entropy','lr','wd']
    dic = StateStorageFile('./pipeline/out/contrastive_proto/tracker')
    
    for k in param:
        plot_param(dic,k)

