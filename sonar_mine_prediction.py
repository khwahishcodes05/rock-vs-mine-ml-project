from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#DATA COLLECTION AND PREPROCESSING
new_columns=[f"feature_{i}" for i in range(1,61)]+["Class"]
df = pd.read_csv('/content/drive/MyDrive/Copy of sonar data.csv',names=new_columns)
df.head()
df.columns
df.info()
df.isnull().sum()
df.duplicated().sum()
df["Class"].unique()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['Class'] = encoder.fit_transform(df['Class'])
df["Class"]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_scaled=scaler.fit_transform(X)
df.head(5)
#TRAIN_TEST_SPLIT
X_train, X_test, y_train, y_test = train_test_split(
X_scaled,
y,
test_size=0.2,
random_state=42
)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
print("Accuracy:",round(accuracy*100,2),"%")
from sklearn.metrics import classification_report

print(classification_report(y_test,
    			
    			
    			
    		
   