
# pickle
try:
    import pickle
    print('pickle: {}'.format(pickle.compatible_formats))
except ImportError:
    print('pickle not present')
# scipy
try:
    import scipy
    print('scipy: %s' % scipy.__version__)
except ImportError:
    print('scipy not present')
# numpy
try:
    import numpy
    print('numpy: %s' % numpy.__version__)
except ImportError:
    print('numpy not present')
# matplotlib
try:
    import matplotlib
    print('matplotlib: %s' % matplotlib.__version__)
except ImportError:
    print('matplotlib not present')
# seaborn
try:
    import seaborn
    print('seaborn: %s' % seaborn.__version__)
except ImportError:
    print('seaborn not present')
# plotly
try:
    import plotly
    print('plotly: %s' % plotly.__version__)
except ImportError:
    print('plotly not present')
# lime
try:
    import lime
    print('lime: imported')
except ImportError:
    print('lime not present')
# pandas
try:
    import pandas
    print('pandas: %s' % pandas.__version__)
except ImportError:
    print('pandas not present')
# statsmodels
try:
    import statsmodels
    print('statsmodels: %s' % statsmodels.__version__)
except ImportError:
    print('statsmodels not present')
# scikit-learn
try:
    import sklearn
    print('sklearn: %s' % sklearn.__version__)
except ImportError:
    print('sklearn not present')
# tensorflow
try:
    import tensorflow as tf
    print('tensorflow: %s' % tf.__version__)
except ImportError:
    print('tensorflow not present')
# keras
try:
    import keras
    print('keras: %s' % keras.__version__)
except ImportError:
    print('keras not present')
# pytorch
try:
    import torch
    print('pytorch: %s' % torch.__version__)
except ImportError:
    print('pytorch not present')
# xgboost
try:
    import xgboost
    print('xgboost: %s' % xgboost.__version__)
except ImportError:
    print('xgboost not present')
# lightgbm
try:
    import lightgbm
    print('lightgbm: %s' % lightgbm.__version__)
except ImportError:
    print('lightgbm not present')
# dask
try:
    import dask
    print('dask: %s' % dask.__version__)
except ImportError:
    print('dask not present')
# pyarrow
try:
    import pyarrow
    print('pyarrow: %s' % pyarrow.__version__)
except ImportError:
    print('pyarrow not present')
# fastparquet
try:
    import fastparquet
    print('fastparquet: %s' % fastparquet.__version__)
except ImportError:
    print('fastparquet not present')
# kmodes
try:
    import kmodes
    print('kmodes: %s' % kmodes.__version__)
except ImportError:
    print('kmodes not present')
# fbprophet
try:
    import fbprophet
    print('facebook prophet: %s' % fbprophet.__version__)
except ImportError:
    print('fbprophet not present')
# gensim
try:
    import gensim
    print('gensim: %s' % gensim.__version__)
except ImportError:
    print('gensim not present')
# nltk
try:
    import nltk
    print('nltk: %s' % nltk.__version__)
except ImportError:
    print('nltk not present')
# spacy
try:
    import spacy
    print('spacy: %s' % spacy.__version__)
except ImportError:
    print('spacy not present')
# tweepy
try:
    import tweepy
    print('tweepy: %s' % tweepy.__version__)
except ImportError:
    print('tweepy not present')
# surprise
try:
    import surprise
    print('surprise: %s' % surprise.__version__)
except ImportError:
    print('surprise not present')
# hyperopt
try:
    import hyperopt
    print('hyperopt: %s' % hyperopt.__version__)
except ImportError:
    print('hyperopt not present')
# zipline
#try:
#    import zipline
#    print('quantopian zipline: %s' % zipline.__version__)
#except ImportError:
#    print('zipline not present')
