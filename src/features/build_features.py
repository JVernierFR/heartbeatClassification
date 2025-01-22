import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy import signal
from scipy.ndimage import gaussian_filter
import pycatch22 as c22


def transform_inputs(files=['../data/raw/mitbih_train.csv','../data/raw/mitbih_test.csv','../data/raw/ptbdb_abnormal.csv','../data/raw/ptbdb_abnormal.csv'],
                 outdir = '../data/processed',
                applyfilter='gaussian',
                  paramfilter=1,
                  Fs=125,
                  nech=6):
    """
         load data, compute features and save database
         input:
             - files : list : files to process
         outputs
             - csv file saved on disk with features computed
    """
    for file in files:
        print('reading file ' + file)
        df = pd.read_csv(file,header=None)
        # last colum is the class
        target = df.pop(df.columns[-1])
        
        # work with array
        X = np.array(df)
        y =  np.array(target)
        
        # apply feature computations
        print('compute features on ' + file)
        XX = all_features(X,applyfilter=applyfilter,paramfilter=paramfilter,Fs=Fs,nech=nech)
        XX['target'] = y

        # save features
        rep,filename = os.path.split(file)
        print('save  to ' + os.path.join(outdir,'features_'+filename))
        XX.to_csv(os.path.join(outdir,'features_'+filename),index=False)
        

# calcul de toutes les feaures
def all_features(X,
                 applyfilter='gaussian',
                  paramfilter=1,
                  Fs=125,
                  nech=6):
    """
        inputs
        --------------
        X = matrice des signaux : n_signals x n_samples
        applyfilter: 'gaussian', 'median','mean' type de smoothing a appliquer pour calculer les features sur les pics
        paramfilter: int : filtering parameter
        Fs = sampling frequency
        nech = nombre d'echantillons utilisé pour calculer les features sur le début du signal 

        outputs:
        XX : matrice des features : n_signal x n_features
    """
    # features sur les peaks
    X1 = peak_features(X,applyfilter=applyfilter,paramfilter=paramfilter,Fs=Fs,nech=nech)
    # features catch22
    X2 = catch24_features(X)
    # concat
    XX = pd.concat((X1,X2),axis=1)
    return XX


def catch24_features(X):
    """
      inputs
      --------------
      X = matric des signaux : n_signals x n_samples

       outputs:
      --------------
        XX : matrice des features : n_signal x n_features
    """

    # apply catch22 
    f24 = np.zeros((X.shape[0],24))
    for k in range(X.shape[0]):
        catch22_output = c22.catch22_all(X[k,:],catch24=True)
        f24[k,:] = catch22_output['values']

    X2 = pd.DataFrame(f24, columns = catch22_output['names'])

    return X2
    
def peak_features(X,
                  applyfilter='gaussian',
                  paramfilter=1,
                  Fs=125,
                  nech=6):
    
    Points = np.zeros((X.shape[0],13))
    check=[]
    # calcul des attributs des 3 pics principaux
    for k in range(X.shape[0]):
        Points[k,:] = get_waves(X[k,:],applyfilter=applyfilter,paramfilter=paramfilter)

    X1 = pd.DataFrame(data=Points,columns=
                      ['R','R_width','R_height','R_prominence',
                      'P1','P1_width','P1_height','P1_prominence',
                      'P2','P2_width','P2_height','P2_prominence',
                      'dur'])

    # samples vers secondes
    for col in ['R','P1','P2','R_width','P1_width','P2_width','dur']:
        X1.loc[:,col] /= Fs

    # ajout des features sur le début du signal
    X1['signal_start_median'] = np.median(X[:,0:nech],axis=1)
    X1['signal_start_std'] = np.std(X[:,0:nech],axis=1)
    X1['signal_start_range'] = np.max(X[:,0:nech],axis=1)-np.min(X[:,0:nech],axis=1)
    X1['signal_start_slope'] = np.mean(np.diff(X[:,0:nech],axis=1),axis=1)

    return X1
    
def get_waves(x,delay=10,applyfilter='gaussian',paramfilter=1):
    
    """

    function computing on 1 signal attributes on the 3 main peaks

        - look for the main peak that should be close to the end of the signal
        - if not found, all outputs are zeros
        - compute attributes on the two main peaks before the previous peak 
        - attributes computed on a smoothed signal
            - position is sampe
            - width in sample
            - height = X[position]
            - prominence
        compute as well the signal duration because signal are zero padded = length of the non-zero padded data
        
    x : 1d array : signal
    delay : number of sample to skip at the beggining of the signal
    applyfilter: smoothing method
        - gaussian filter of kernel paramfilter
        - median : apply median filter : paramfilter = window size in samples
        - mean : appply mean filter: paramfilter = window size in samples
    outputs
    #############
        R,R_prominence : position et prominence du pic principal qui devrait etre l'onde R
        P1,P1_width,P1_height,P1_prominence, position, largeur, hauteur et prominence du 1er pic secondaire
        P2,P2_width,P2_height,P2_prominence, position, largeur, hauteur et prominence du 2e pic secondaire

    """
    n_zeros_pad = np.argmax( np.flip(x)!=0)
    signal_dur_sample = -n_zeros_pad + x.shape[0]

    if applyfilter=='gaussian':
        x1 = gaussian_filter(x,paramfilter)
    elif applyfilter=='median':
        x1 = signal.medfilt(x,paramfilter)
    elif applyfilter=='mean':
        x1 = signal.convolve(X[ii,:],np.ones(paramfilter)/paramfilter,mode='same')
    # trouve le pic principal et verifie qu'il est assez à la fin 
    # on  fait ca sur le signal non filtré pour le pic principal
    peaks, properties = signal.find_peaks(x[delay:signal_dur_sample+1],height=(None,None),plateau_size=(None,None),
                                          prominence=(0.2,None),width=(None,None),distance=10)
    
    if len(peaks):
        imax_pro = np.argmax(properties['prominences'])
        R = peaks[imax_pro]+delay
        R_prominence = properties['prominences'][imax_pro]
        R_height = properties['peak_heights'][imax_pro]
        R_width = properties['widths'][imax_pro]
        istop = R-5
        if R<signal_dur_sample/2:
            R = 0
            R_prominence = 0
            istop = signal_dur_sample
            R_height = 0 
            R_width = 0
    else:
        R = 0
        R_prominence = 0
        istop = signal_dur_sample
        R_height = 0
        R_width = 0 
    # on  fait ca sur le signal  filtré pour les pics secondaires
    peaks, properties = signal.find_peaks(x1[delay:istop],height=(None,None),plateau_size=(None,None),
                                          prominence=(None,None),width=(None,None),distance=10)
    if len(peaks)>0:
        imax_width = np.argmax(properties['widths']*properties['prominences'])
        P1 = peaks[imax_width]+delay
        P1_width = properties['widths'][imax_width]
        P1_height = properties['peak_heights'][imax_width]
        P1_prominence = properties['prominences'][imax_width]
        properties['widths'][imax_width] = 0

        imax_width = np.argmax(properties['widths']*properties['prominences'])
        P2 = peaks[imax_width]+delay
        P2_width = properties['widths'][imax_width]
        P2_height = properties['peak_heights'][imax_width]
        P2_prominence = properties['prominences'][imax_width]
    else:
        P1=0
        P1_width=0
        P1_height=0
        P1_prominence =0
        P2=0
        P2_width=0
        P2_height=0
        P2_prominence =0
        
    if P2<P1:
        P2,P1 = P1,P2
        P2_prominence,P1_prominence = P1_prominence,P2_prominence
        P2_width,P1_width = P1_width,P2_width
        P2_height,P1_height = P1_height,P2_height

    return R,R_width,R_height,R_prominence,P1,P1_width,P1_height,P1_prominence,P2,P2_width,P2_height,P2_prominence,signal_dur_sample