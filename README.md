# Sonar-Project
Pour que le code fonctionne, télécharger la database Urbansound8K : https://urbansounddataset.weebly.com/urbansound8k.html
et les librairies nécessaires : 
#import libraries 
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Project Specific Libraries
import librosa
import librosa.display
import IPython.display as ipd
# Libraries for Classification and building Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
