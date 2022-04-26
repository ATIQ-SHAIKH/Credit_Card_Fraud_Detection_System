import pandas as pd
import operator
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample
import joblib

df = pd.read_csv('C:\\Users\\ATIK SHAIKH\\Desktop\\Model Based Web App\\creditcard.csv')

x = df.drop('Class', axis=1)
y = df.Class.values

# Dropping unnecessary columns
x.drop(['Time','Amount'], axis=1, inplace=True)

# Since dataset is highly unbalanced we can use under sampling or mix of under and over sampling to increase number of samples
leg_df = df[df.Class == 0]
fraud_df = df[df.Class == 1]

no_of_samples = round(leg_df.shape[0] * 0.05)
no_of_samples

leg_df_2 = resample(leg_df, n_samples=no_of_samples, random_state=15)
df_sampled = pd.concat([leg_df_2,fraud_df],axis=0)

x_sampled = df_sampled.drop('Class', axis=1)
y_sampled = df_sampled.Class

ros = RandomOverSampler(random_state=42)

x,y = ros.fit_resample(x_sampled,y_sampled)

# Splitting dataset for training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y, stratify=y, random_state=12)

columns = ['Model','accuracy score']
evaluation_df = pd.DataFrame(columns=columns)

model_acc = {"Logistic Regression Model" : [], "Bernoulli Naive Bayes Model" : [], "KNN Model" : [], "KMeans Model" : []}

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=200,random_state=12)
lr_model.fit(x_train,y_train)
pred1 = lr_model.predict(x_test)
accuracy_score  = metrics.accuracy_score(y_test,pred1) 
model_acc["Logistic Regression Model"].append(accuracy_score)

# Bernoulli Naive Bayes Model
gnb_model = BernoulliNB()
gnb_model.fit(x_train,y_train)
pred2 = gnb_model.predict(x_test)
accuracy_score  = metrics.accuracy_score(y_test,pred2) 
model_acc["Bernoulli Naive Bayes Model"].append(accuracy_score)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5,algorithm="kd_tree",n_jobs=-1)
knn.fit(x_train,y_train)
pred3 = knn.predict(x_test)
accuracy_score  = metrics.accuracy_score(y_test,pred3) 
model_acc["KNN Model"].append(accuracy_score)

# KMeans Model
kmeans = KMeans(n_clusters=2,random_state=0,algorithm="elkan",max_iter=10000)
kmeans.fit(x_train)
pred4 = kmeans.predict(x_test)
accuracy_score  = metrics.accuracy_score(y_test,pred4) 
model_acc["KMeans Model"].append(accuracy_score)

model_max_acc = max(model_acc.items(), key=operator.itemgetter(1))[0]

print(model_max_acc)
print(model_acc[model_max_acc])

filename3 = "KNN Model.sav"
joblib.dump(knn, filename3)

# filename4 = "KMeans Model.sav"
# joblib.dump(kmeans, filename4)