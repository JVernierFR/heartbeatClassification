HEARTBEAT CLASSIFICATION
==============================

This project was created as part of the Data Scientits training  (Apr 2024 to Feb 2025). The presented project aims to assess heart disorders using electrocardiogram (ECG) recordings.

## Team
 - Jerome Vernier
 - Marouane Essougdali

## Introduction

 The analyzed cardiac signals come from both healthy patients and patients with arrhythmia or myocardial infarction. The primary goal of this project is to develop deep neural network architectures for the classification of cardiac signals

The dataset of ECG is composed of two collections of heartbeat signals derived from two famous datasets in heartbeat classification, the MIT-BIH Arrhythmia Dataset and The PTB Diagnostic ECG Database.
This dataset consists of a series of CSV files. Each of these CSV files contain a matrix, with each row representing an example in that portion of the dataset. The final element of each row denotes the class to which that example belongs.

Dataset 1 constains signals qualified as :
- normal: class 0
- Supraventricular premature beat: class 1
- Premature ventricular contraction: class 2
- Fusion of ventricular and normal beat: class 3
- Unclassifiable beat: class 4

Dataset 2 constains signals qualified as :
- normal
- abnormal

ECG is a measurement of the heart's electrical activity, represented by several peaks corresponding to the activity of different heart chambers. The main peak, known as the QRS complex, corresponds to the activation of the ventricles, the P wave represents the activation of the atria, and the T wave corresponds to the repolarization of the ventricles.

![ECG](reports/figures/ekg-ecg-interpretation-p-qrs-st-t-wave.jpg)

## Objective

The objective of this project is to predict the type of disorder present in an ECG signal. Two possible approaches were considered for processing the data:
- Using a single dataset for multiclass prediction
- Merging both datasets to perform binary classification

The first approach was selected as the primary method. However, the second approach was also tested at the end of the project in the notebook Modelling_4-AugmentOnthefly_MultiClasses.ipynb.

## Project-Structure

To run the notebooks, you will need to install the dependencies (in a dedicated environment)


`pip install -r requirements.txt`

All the code files are provided as notebooks. Those labeled with "Explo" focus on data exploration, visualization, and preprocessing. Notebooks labeled with "Modelling" cover various modeling techniques, including:
- SVM on features computed on the ECG signal with the module catc22
- ANN: neural network
- CNN: convolutional network

The modeling process is explored in several stages. Additionally, tests were conducted using recurrent layers and transfer learning techniques.


## Best model 

the best model is a CNN model from the Modelling_3-CNN.ipynb notebook.

## Retrain
the dataset must be downloaded and saved from kaggle https://www.kaggle.com/datasets/shayanfazeli/heartbeat in the folder /data/raw.

`python .\src\models\train.py`

this outputs a heartbeat_model.pkl under the models folder saved with joblib.

## Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
     │   ├── balance       <- Scripts to get a balanced dataset
    │   │   └── load_balanced.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
