# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE

### Data.csv
```python
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```

### Data.csv Output:

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/df7b0755-3aa1-4639-b68c-7d6d920ae7a0)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/ae51e80b-c199-45ab-9934-8daa001ac97e)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/904dbf06-4c8f-4db7-afe6-4842ffc6b801)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/cae9ab47-8267-43d2-918d-40742ffc42a0)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/5c2cb315-705d-4d83-86f4-7604ab2d27be)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/df913ea5-2db7-4ab4-9af4-255a291276ae)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/d4ef8b4d-ecd9-4a53-a517-083dd914c73d)

### Encoding.csv Code:

```python
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```
### Encoding.csv Output:

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/668dc881-d653-48d7-bf8d-e520045e1e24)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/020f4c6a-9c91-4c12-99de-b7211981acba)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/8fbdf35b-705d-41ea-ba77-ad2b52fbcce2)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/ca28dbf0-4f20-4ec8-b29f-203a2d63218f)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/597cd03f-d6e7-4832-b4b1-8d03f131ce4d)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/f4c9d821-515f-49f7-8ff4-9a648641d13b)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/d8266483-1fe6-4a66-8cd8-69d904a127a6)

### Titanic.csv Code:

```python
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```

### Titanic.csv Output:

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/f93c7440-8bec-428c-aca1-5f8fdc31a9a2)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/ced73b3e-6357-42df-acd7-e9e1c4ad71c2)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/891bcd61-6b4c-40d8-977d-221f039676ce)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/f8735661-92ce-4536-b08c-3abebfde3c7c)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/684c5b7e-2549-4d3c-9a4d-6a6031d84525)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/9432cd21-2782-4b81-8458-d1502593fa4f)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/02d6ae02-1f82-4892-8bd6-55e967f07cf1)

![image](https://github.com/Sachin-vlr/EX-05-Feature-Generation/assets/113497666/4f698c86-7f80-4d44-a169-b8a535d2d1f1)

# RESULT:
Hence Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.

