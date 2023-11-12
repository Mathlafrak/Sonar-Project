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


# Load data from the Database 
download_path = Path.cwd() / 'UrbanSound8K'
#path pour aller dans les tableaux csv
metadata_file = download_path / 'metadata' / 'UrbanSound8K.csv'
#path pour aller dans les fichiers audio   
audio_file = download_path / 'audio'
df = pd.read_csv(metadata_file)
print(df.head())

class_name = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

#boucle for aléatoire pour récupérer 3 fichiers 
for i in range(192, 197, 2):
    #path pour récupérer le fichier audio -- directory audio + foldX + nom du fichier audio spécifique pour ce i 
    audio_file_path = audio_file / f'fold{df["fold"][i]}' / df["slice_file_name"][i]
    #let's view the waveplot 
    plt.figure(figsize=(14,3))
    #charger le fichier audio (données stockées dans data et sample rate)
    data, sr = librosa.load(audio_file_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    
    plt.figure(figsize=(18, 3))
    
    plt.subplot(1, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(df["class"][i])
    
    plt.subplot(1, 2, 2)
    librosa.display.waveshow(data, sr = sr)
    plt.title(df["class"][i])

#plt.show()

# Example for one wav file
audio_file_path2 = audio_file / 'fold5' / '100032-3-0-0.wav'
# Extract the data
data, sr = librosa.load(audio_file_path2)
#mel_spectogram function of librosa to extract the spectogram data as a numpy array
arr = librosa.feature.melspectrogram(y = data, sr = sr)
print(arr.shape)

#Parser function

features = []
labels = []

def parser():
    # Function to load files and extract features
    for i in range(30):
        audio_file_path = audio_file / f'fold{df["fold"][i]}' / df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        data, sr = librosa.load(audio_file_path, res_type='kaiser_fast') 
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y = data, sr = sr), axis=1)       
        features.append(mels)
        labels.append(df["classID"][i])

    return features, labels

#Call parser function which fill the feature[] and label[]
x, y = parser()

#Final Data
X = np.array(x)
Y = np.array(y)

print(X.shape)
print(Y.shape)

# Check the datatype of X and Y
print(X.dtype)
print(Y.dtype)

# Split the data into Training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_test,y_test)


""""
#-------------------------------------------Random Forest Classifier----------------------------------------

forest = RandomForestClassifier()
# fit classifier to training set
print(forest.fit(X_train, y_train))
# make predictions on test set
forest_pred = forest.predict(X_test)
print(forest_pred)

print(classification_report(y_test, forest_pred))

# Confusion Matrix
confusion_matrix(y_test, forest_pred)

# Ploting Confusion Matrix
plt.figure(figsize = (12, 10))
sns.heatmap(confusion_matrix(y_test, forest_pred), 
            annot = True, linewidths = 2, fmt="d", 
            xticklabels = class_name,
            yticklabels = class_name)
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()


#---------------------------------- K_NearestNeighbors -----------------------------------------

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors (n_neighbors) as needed

# Train the KNN classifier on the training set
knn.fit(X_train, y_train)

# Make predictions on the test set
knn_pred = knn.predict(X_test)

# Print classification report for KNN
print("Classification Report for K-Nearest Neighbors:")
print(classification_report(y_test, knn_pred))

# Confusion Matrix for KNN
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_test, knn_pred),
            annot=True, linewidths=2, fmt="d",
            xticklabels=class_name, yticklabels=class_name)
plt.title("Confusion Matrix for K-Nearest Neighbors")
plt.show()



#---------------------------------- Naive Bayes  -----------------------------------------
# Initialize Naive Bayes classifier
naive_bayes = GaussianNB()

# Train the Naive Bayes classifier on the training set
naive_bayes.fit(X_train, y_train)

# Make predictions on the test set
naive_bayes_pred = naive_bayes.predict(X_test)

# Print classification report for Naive Bayes
print("\nClassification Report for Naive Bayes:")
print(classification_report(y_test, naive_bayes_pred))

# Confusion Matrix for Naive Bayes
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_test, naive_bayes_pred),
            annot=True, linewidths=2, fmt="d",
            xticklabels=class_name, yticklabels=class_name)
plt.title("Confusion Matrix for Naive Bayes")
plt.show()

#-----------------------------------------Support Vector Machine------------------------------------------
# Initialize SVM classifier
svm_classifier = SVC()

# Train the SVM classifier on the training set
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
svm_pred = svm_classifier.predict(X_test)

# Print classification report for SVM
print("\nClassification Report for Support Vector Machine:")
print(classification_report(y_test, svm_pred,zero_division=1))

# Confusion Matrix for SVM
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_test, svm_pred),
            annot=True, linewidths=2, fmt="d",
            xticklabels=class_name, yticklabels=class_name)
plt.title("Confusion Matrix for Support Vector Machine")
plt.show()

"""""

#-----------------------------------------Decision Trees------------------------------------------
# Initialize Decision Trees classifier
decision_tree_classifier = DecisionTreeClassifier()

# Train the Decision Trees classifier on the training set
decision_tree_classifier.fit(X_train, y_train)

# Make predictions on the test set
decision_tree_pred = decision_tree_classifier.predict(X_test)

# Print classification report for Decision Trees
print("\nClassification Report for Decision Trees:")
print(classification_report(y_test, decision_tree_pred))

# Confusion Matrix for Decision Trees
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_test, decision_tree_pred),
            annot=True, linewidths=2, fmt="d",
            xticklabels=class_name, yticklabels=class_name)
plt.title("Confusion Matrix for Decision Trees")
plt.show()
