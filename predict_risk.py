from pymongo import MongoClient
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
import joblib
from joblib import dump, load
import numpy
import argparse
import requests
import pymysql
import pandas
import csv
## 1 - Extraction and save of data from MongoDB
## connection's details
host = "208.87.130.253"
port = 27017
database_name = "mag1_project"
username = "mag1_student"
password = "Gogo1gogo2"
authentication_database = "mag1_project"
collection = "project"
## Connection to database MongoDB
client = MongoClient(host, port, username=username, password=password, authSource=authentication_database)
db = client[database_name]

##  collection 'project'
collection = db[collection]

## extract the files in the project (collection)
files = collection.find()

## Save in dataframe (df1)
mongo_data = list(collection.find({})) 
df1=pandas.DataFrame(mongo_data)
#print (df1)

## 2 - Extract and save of data from MySql database

## Connection detail 
host = "144.24.194.65"
port = 3999
database_name = "mag1_project"
username = "mag1_student"
password = "Gogo1gogo2"
table = "project"
db_url = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database_name}'

## Creating of engine's connection
engine = create_engine(db_url)

## Recuperate data in dataframe (df2)
query = "SELECT * FROM project ;"
df2 = pandas.read_sql_query(query, engine)
engine.dispose()
#print (df2)

## 3 - Extract and save data from HTML base

url = "https://h.chifu.eu/data.html"
response = requests.get(url)
html_content = response.content
soup = BeautifulSoup(html_content, "html.parser")

## Find the table HTML using the balises 'table' & 'tr' 
table = soup.find("table")
rows = table.find_all("tr")
data_html=[]
## Printing the data of table
for row in rows:
    cols = row.find_all(['th', 'td'])  # columns (header & data)
    cols = [col.get_text(strip=True) for col in cols]
    data_html.append(cols)

## Recuperate data in dataframe (df3)
header = data_html[0]
data = data_html[1:]
df3=pandas.DataFrame(data, columns=header)
#print (df3)

## 4 - Merging of databses

## Merging of dataframes on  "Company ID" variable
merged_df = pandas.merge(df1, df2, on="Company ID", how='inner')  # two files firstly
## To merge the previous dataframe with df3, we need to convert the type of the variable of df3 to int64.
df3['Company ID'] = df3['Company ID'].astype('int64')

merged_df = pandas.merge(merged_df, df3, on="Company ID", how='inner')  
#print (merged_df)

## 5 - Training of prediction model
## We use a library scikit-learn for training the model
## Proprocessing of data (for Credit Rating, we have to use the ordinal encoding because, this variable is categoric or not numeric)
## For the encoding, we started from rating 1 to 9, where 1 == AAA meaning 'Very good repayment capacity' and 9 ==C meaning 'Very poor repayment capacity'.
## Creating of mapping dictionnary
mapping_dict = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4, 'BB': 5, 'B': 6, 'CCC': 7, 'CC': 8, 'C': 9}
merged_df['credit_rating_encoded'] = merged_df['Credit Rating'].map(mapping_dict)
#print (merged_df)

## Division of data into features (X) and target variable (y)
X = merged_df[["Revenue", "Expenses","Profit", "Employee Count", "Debt-to-Equity Ratio", "Price-to-Earnings Ratio", "Research and Development Spend", "Market Capitalization"]] 
y = merged_df["Risk"] # target variable

## Data split into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Initialising of the random forest model
model_pred = RandomForestClassifier()

## Training the model
model_pred.fit(X_train, y_train)

## Getting relevance of the variables
Relevance = model_pred.feature_importances_

## Sorting variable indices by importance
sorted_indices = Relevance.argsort()[::-1]

## Printing of relevance
#for index in sorted_indices:
    #print(f"Variable : {X.columns[index]}, Relevance : {Relevance[index]}")

## Making predictions on the test data
predictions = model_pred.predict(X_test)

## Evaluating of model performance
accuracy = accuracy_score(y_test, predictions)

## Displaying a classification report
print(classification_report(y_test, predictions))
## For variable selection, after training the model on the training data, we checked the importance of each variable. Most variables have an estimated importance of 0.1. 
## Only the variable 'credit_rating_encoded' has an importance of around 0.05. We therefore removed it from the independent variables to improve the model's performance.
## Saving the train model
joblib.dump(model_pred, 'train_model.pkl')

## Uploading of train model du modèle sauvegardé
model_loaded = joblib.load('train_model.pkl')

## Upload the new data
new_data = pandas.read_csv("https://h.chifu.eu/final_test.csv")
new_data_m=new_data.drop(['Company ID','Credit Rating'], axis=1)

## Initializing the argument analyzer
parser = argparse.ArgumentParser(description='Script to save predicted labels for test dataset along with Company ID')

## Add argument for output file
parser.add_argument('--output', type=str, help='Output file to save the predicted labels with Company ID')

## Parsing of argument
args = parser.parse_args()

## Retrieve output file name from arguments
output_file = args.output
## Making predictions based on new data 
predicted_risk=model_loaded.predict(new_data_m)
predicted_risk=numpy.array(predicted_risk)
predictions= { 'Company ID' :new_data['Company ID'],
              'Predicted_Risk' : predicted_risk }
df4 = pandas.DataFrame(predictions)
#print (df4)
data_to_write = list(zip(df4['Company ID'], df4['Predicted_Risk']))
output_file='predictions.csv'
## Writing to CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Company ID', 'Predicted_Risk'])
    writer.writerows(data_to_write)

## We take the liberty of adding a python library to the script, enabling us to perform an exploratory analysis of the data in the 'df_to_analyse' dataframe.
import sweetviz as sv
df_to_analyse= merged_df.drop("_id", axis=1)
analysis_report= sv.analyze(source=df_to_analyse, target_feat="Risk")
analysis_report.show_html('analysis_report.html')

from art import *
text = "END OF EXERCICE. THANK YOU SIR."
print(text2art(text))