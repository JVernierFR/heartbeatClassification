import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import os 
from scipy import signal, interpolate
from sklearn.utils import shuffle


# a recoder: serait plus pratique avec une classe
# charge les données brute
def load(file='../data/raw/mitbih_train.csv'):
    if not os.path.isfile(file):
        raise ValueError('file not found')
    df = pd.read_csv(file,header=None)
    target = df.pop(df.columns[-1])
    X = np.array(df)
    y = np.array(target)

    return X,y

# charge les données et sort un dataset equilibré
def load_balanced_data(file='../data/raw/mitbih_train.csv', 
                       method='under',
                       n_normal=6500,
                       smote_perturb_smote_ratio=0.5,
                       plow=0.85,phigh=1.15,slopelow=-0.1,slopehigh=0.1,strechlow=-5,strechhigh=5):
    if method not in ['under','smote','perturb','smote-perturb']:
        raise ValueError('mode must be "under","smote","perturb" or "smote-perturb"')
                         
    X,y = load(file)

    X_rs,y_rs = sampling_strategy(X,y,method=method,n_normal=n_normal,smote_perturb_smote_ratio=smote_perturb_smote_ratio,
                                 plow=plow,phigh=phigh,slopelow=slopelow,slopehigh=slopehigh,strechlow=strechlow,strechhigh=strechhigh)
    return X_rs,y_rs 
    
    
# radom selection of N classe 0 signals 
def random_subsample_classe(y,n_normal,classe=0):
    """
    # inputs
    ------------
    y = a 1D array defining classification
    # outputs
    ------------
    selected_indices : indices within y of size n_normal:  n_normal random indices of y = classe
    # example
    ------------
    I want 10 signals randomly selected within classe 4, this will give the 10 indices to apply to y
    random_subsample_classe(y,10,classe=4)
    """
    bool_0 = y == classe
    ii_0 = [k for k,x in enumerate(bool_0) if x]
    ii_0 = shuffle(ii_0,random_state=123)
    selected_indices = ii_0[0:n_normal]
    return selected_indices

def sampling_strategy(X,y,method='under',n_normal=6500,smote_perturb_smote_ratio=0.5,
                     plow=0.85,phigh=1.15,slopelow=-0.1,slopehigh=0.1,strechlow=-5,strechhigh=5):
    
    """
    Balance the dataset
    # inputs
    ------------
    X, y : data and target
    method :
        - under : apply undersampling: the number of samples per class is the number of the minority class
        - smote : apply oversamplng : the number of samples per class is n_normal
        - perturb : apply low perturbation on the signals so that all the number of samples per class is n_normal. append the classes that have less than  n_normal sample, if the class has more than n_normal samples, then random undersample
        - smote-perturb : combine smote and perturb with smote_perturb_smote_ratio using smote and 1-smote_perturb_smote_ratio using perturb
    # outputs
    ------------
    """
    
    if method not in ['under','smote','perturb','smote-perturb']:
        raise ValueError('mode must be "under","smote","perturb" or "smote-perturb"')
    if method == 'under':
        ru = RandomUnderSampler(random_state=123)
        X_rs, y_rs = ru.fit_resample(X,y)
    elif method == 'smote':
        ii_0 = random_subsample_classe(y,n_normal,classe=0)
        X1 = np.concatenate((X[ii_0,:],X[y != 0,:]), axis=0)
        y1 = np.concatenate((y[ii_0],y[y != 0]))
    
        sm = SMOTE(random_state=123,sampling_strategy='not majority') 
        X_rs, y_rs = sm.fit_resample(X1 ,y1)
    elif method == 'perturb':
        ii_0 = random_subsample_classe(y,n_normal,classe=0)
        X1 = np.concatenate((X[ii_0,:],X[y != 0,:]), axis=0)
        y1 = np.concatenate((y[ii_0],y[y != 0]))
    
        X_rs, y_rs = augmentbase_shape(X1,y1,n_normal)
    elif method == 'smote-perturb':
        nsmote = int(n_normal*smote_perturb_smote_ratio)
        npeturb = n_normal - nsmote
        
        c = pd.Series(y).value_counts()
        too_many = c.index[c>nsmote]
        too_many_bool = np.zeros(X.shape[0], dtype=bool)
        selected_indices_smote = []
        selected_indices_perturb = []
        for cl in too_many:
            ii_0 = random_subsample_classe(y,n_normal,classe=cl)
            print('nsmote: ',nsmote,' , nperturb: ',npeturb)
            selected_indices_smote += ii_0[0:nsmote]
            selected_indices_perturb += ii_0[nsmote:n_normal]
            too_many_bool = np.logical_or(too_many_bool,y == cl)
            
        
        other_classes = np.logical_not(too_many_bool)
        X1 = np.concatenate((X[selected_indices_smote,:],X[other_classes,:]), axis=0)
        y1 = np.concatenate((y[selected_indices_smote],y[other_classes]))

        X2 = np.concatenate((X[selected_indices_perturb,:],X[other_classes,:]), axis=0)
        y2 = np.concatenate((y[selected_indices_perturb],y[other_classes]))
        
        X_sm, y_sm = sampling_strategy(X1,y1,'smote',nsmote)
        print('after smote')
        print(pd.Series(y_sm).value_counts())
        X_pe, y_pe = sampling_strategy(X2,y2,'perturb',npeturb)

        X_rs = np.concatenate((X_sm,X_pe), axis=0)
        y_rs = np.concatenate((y_sm,y_pe))
        
    return X_rs, y_rs
    
def signal_temporal_trend(x,pente,no_change_index):
    """
    # inputs
    -------------
        - x : signal np.array(npts)
        - pente : float : slope to apply to the data
        - no_change_index : int, value at index no_change_index is not modified :  x_out[no_change_index] = x[no_change_index]
    """
    # applique une trend lineaire au signal . au point no_change_index, le coef vaut 1 
    # la pente de la trend est m
    coef = pente/no_change_index*np.arange(x.shape[0])+1+pente
    return coef*x


def augment(x,plow=0.85,phigh=1.15,slopelow=-0.1,slopehigh=0.1,strechlow=-5,strechhigh=5):
    """
    # inputs
    -------------
        - x : signal np.array(npts)
        - plow,phigh  : float :min and max power to apply (signal_temporal_trend)
        - slopelow,slopehigh  : float :min and max slope to apply (signal_temporal_trend)
    apply a perturbation including randomly between the bounds:
        - streching
        - add linear slope
        - signal to the power 
         """
    np.random.seed(123)
    # p : mise a la puissance p
    p = np.random.uniform(low=plow, high=phigh)
    # c : application d'une trend lineaire, au point c, la trend vaut 1
    # c = int(np.random.uniform(low=20,high=160))
    # je crois que c ne sert à rien ca apres les data sont renormalisée, donc je le met au milieu
    c= x.shape[0] //2 
    # pente de la trend
    m = np.random.uniform(low=slopelow,high=slopehigh)
    # applique une puissance au signal
    a = signal_temporal_trend(x,m,c)**p
    a = a/(max(a)-min(a))
    
    # streching
    Nadd = np.random.randint(low=strechlow,high=strechhigh)
    if Nadd !=0:
        f = interpolate.interp1d(np.arange(0, x.shape[0]), a)
        a = f(np.linspace(0.0, x.shape[0]-1, x.shape[0]+Nadd))
    if Nadd<0:
        b = np.zeros(shape=(x.shape[0], ))
        b[0:x.shape[0]+Nadd] = a
        a = b
    else:        
        a = a[0:x.shape[0]]
    return a


def augmentbase_shape(X_data,y_data,n_per_class,plow=0.85,phigh=1.15,slopelow=-0.1,slopehigh=0.1,strechlow=-5,strechhigh=5):
    X2 = None
    y2 = None
    npts = X_data.shape[1]
    print(np.unique(y_data))
    np.random.seed(123)
    for classe in np.unique(y_data):
        # indices des signaux de classe "classe"
        index_of_class = [k for k,x in enumerate(y_data) if x==classe]
        if len(index_of_class) > n_per_class:
            print('class ',classe,' undersampling')
            index_of_class = shuffle(index_of_class,random_state=123)
            DataAdd = X_data[index_of_class[0:n_per_class],:]
            Yadd = y_data[index_of_class[0:n_per_class]]       
        else:
            Nmissing = n_per_class-len(index_of_class)
            print('class ',classe,' oversampling, missing ',Nmissing)
            ii = np.random.choice(index_of_class,Nmissing)
            data_temp = np.zeros((Nmissing,npts))
            for kk,jj in enumerate(ii):
                data_temp[kk,:] = augment(X_data[jj,:],plow=plow,phigh=phigh,slopelow=slopelow,slopehigh=slopehigh,strechlow=strechlow,strechhigh=strechhigh)
                
            DataAdd = np.concatenate((X_data[index_of_class,:],data_temp),axis=0)
            Yadd =  np.ones(DataAdd.shape[0])*classe
            #print('class ',classe,' add ', data_temp.shape[0],'new shape',DataAdd.shape[0],',',Yadd.shape[0])
       
        if X2 is None:
            X2 = DataAdd
            y2 = Yadd
        else:
            X2 = np.concatenate((X2,DataAdd),axis=0)
            y2 =np.concatenate((y2,Yadd))
    return X2,y2