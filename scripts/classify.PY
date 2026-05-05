#
#
#My name: Sevren Quijada
#Course: LING 539
#Date: 05/05/26
#Kaggle Competition 
#
#
#about:
#--->This is a python program that takes training and testing
#data and then trains a logistic regression classifier to make
#predictions in the form of a csv file. It consists of 3 parts: 
#preprocessing, training+testing, and final predictions+exporting

#-->imports pandas for initial data manipulation. numpy will
#be used to format vectors for the model. we will be using the
#tfidf vectorizer, logRegression, and pipline from scikitlearn; 
#as well as metrics

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


	#PART A: Preprocessing

#1.read in the data obtained from Kaggle
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#2.This step fills in any missing data that is in the data frame
train['TEXT'] = train['TEXT'].fillna("")
test['TEXT'] = test['TEXT'].fillna("")
test['ID'] = test['ID'].fillna("")

#3.turn the training data into numpy arrays for evenual vectorization 
x_np = train['TEXT'].to_numpy()
y_np = train['LABEL'].to_numpy()

#4.This step slices off a part of the training data into a development
#set for testing, I used the first 100 reviews for the dev set
x_dev = x_np[:100]
y_dev = y_np[:100]

x_train = x_np[100:]
y_train = y_np[100:]

#5.Here I process the real testing data for the competition
test_Text = test['TEXT'].to_numpy()
test_ID = test['ID'].to_numpy()

#6.Sanity check: there should be 32,289, 19139, and 18889 reviews 
#for labels 0, 1, and 2 respectively
#print(train['LABEL'].value_counts())


	#PART B: Training and Testing

#1.This step creates the pipeline of vectorizer to model
model_lm = make_pipeline(TfidfVectorizer(ngram_range=(1,3), min_df=4, max_df=.8,max_features=50000), LogisticRegression(penalty='l2',solver='saga'))

#2.This is the training step. It uses the Text of the data and the labels 
#to fit a model 
model_lm.fit(x_np, y_np)

#3.Below is the training step for development
#model_lm.fit(x_train, y_train)

#4.this predicts the developement set for training
val_pred = model_lm.predict(x_dev)

#optional print for testing and development 
#print(classification_report(y_dev,val_pred))


	#Part C: Final Prediction and Exporting

#1.This step creates predictions based on the testing data
test_predict = model_lm.predict(test_Text)

#2.Turning the labels and ids from np.arrays into lists
labels = list(test_predict)
ids = list(test_ID)

#3.Create a pandas datafram for exporting
df = pd.DataFrame({
    "ID": ids,
    "LABEL": labels
})

#4.export to the kaggle site, naming the file submission2.csv
# 
# |
# v
#
#uncomment this to export it and input the name of the export file you need
df.to_csv("/grad-level-term-project-kaggle-competition-Sevren-Quijada/submission-1.csv",index=False)
print("End of program")