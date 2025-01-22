import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas as pd
st.title("Hearbeast Classification")
#st.write("Introduction")
#if st.checkbox("Afficher"):
#    st.write("Suite du Streamlit")
st.sidebar.title("Sommaire")
pages=["Introduction", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
from st_kaggle_connector import KaggleDatasetConnection
conn = st.connection("kaggle_datasets", type=KaggleDatasetConnection)
df1 = conn.get(path='shayanfazeli/heartbeat', filename='mitbih_test.csv', ttl=3600)
df1.columns = pd.RangeIndex(df1.shape[1])
st.write('rien')
if page == pages[0] : 
  st.write("### Introduction")
if page == pages[1] : 
    st.write("### DataVizualization")
  
    target1 = df1.pop(187)
    dict_target1 = {0: 'Normal',
                1:'Supraventricular premature beat',
                2:'Premature ventricular contraction',
                3:'Fusion of ventricular and normal beat',
                4: 'Unclassifiable beat'}
    target_str1 = target1.replace(dict_target1)
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(121)
    sns.countplot(x=target_str1,ax=ax,palette='PuRd_r')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title('Nombre de signaux par classe - dataset 1 ')
    ax.set_xlabel('')
    st.pyplot(fig)
if page == pages[2] : 
    st.write("### Modélisation")