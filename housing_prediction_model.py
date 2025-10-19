import pandas as pd

df = pd.read_csv('Housing.csv')
df

"""# Data Visualization and Exploration"""

selected = df.iloc[:,:-1]
selected.hist(figsize=(14,14))

import matplotlib.pyplot as plt

# Identify the columns with "Yes" and "No" values
yes_no_columns = ['guestroom', 'mainroad', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Prepare data for visualization
data = {col: df[col].value_counts() for col in yes_no_columns}
visualization_df = pd.DataFrame(data)

# Plot grouped bar chart
visualization_df.T.plot(kind='bar', figsize=(12, 8), color=['skyblue', 'salmon'], width=0.8)

plt.title('Distribution of Yes and No Across Features', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Category', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

selected = df.iloc[:,:-1]
selected.plot(kind='density', subplots=True, layout=(3,5), sharex=False,figsize=(15,15))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Setting up the plot
num_features = 5
# Limit to num_features columns
limited_features = selected.iloc[:, :num_features]

fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(10, 5 * num_features))

for i, column in enumerate(limited_features.columns):
    df[column].plot(kind='density', ax=axes[i], color='blue', alpha=0.5, label='PDF')

    # Calculate mean, median, and mode
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()[0]

    # Marking mean, median, and mode
    axes[i].axvline(x=mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    axes[i].axvline(x=median, color='green', linestyle='--', label=f'Median: {median:.2f}')
    axes[i].axvline(x=mode, color='purple', linestyle='--', label=f'Mode: {mode:.2f}')

    # Adding titles and labels
    axes[i].set_title(f'Probability Density Function for {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Density')
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Calculate skewness
skewness = numeric_df.skew()
print(skewness)

selected.plot(kind='box', subplots=True, sharex=False, sharey=False, layout=(5,3), figsize=(15,30))

df.describe()

"""# checking for NaN values"""

df.isnull().sum()

"""no missing values

## **removing the outliers**
"""

Q1 = np.percentile(df['price'], 25)
Q3 = np.percentile(df['price'], 75)
print(Q3)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

Q1 = np.percentile(df['area'], 25)
Q3 = np.percentile(df['area'], 75)
print(Q3)
IQR = Q3 - Q1
df = df[(df['area'] >= Q1 - 1.5 * IQR) & (df['area'] <= Q3 + 1.5 * IQR)]

Q1 = np.percentile(df['bedrooms'], 25)
Q3 = np.percentile(df['bedrooms'], 75)
print(Q3)
IQR = Q3 - Q1
df = df[(df['bedrooms'] >= Q1 - 1.5 * IQR) & (df['bedrooms'] <= Q3 + 1.5 * IQR)]

Q1 = np.percentile(df['bathrooms'], 25)
Q3 = np.percentile(df['bathrooms'], 100)
print(Q3)
IQR = Q3 - Q1
df = df[(df['bathrooms'] >= Q1 - 1.5 * IQR) & (df['bathrooms'] <= Q3 + 1.5 * IQR)]

Q1 = np.percentile(df['stories'], 25)
Q3 = np.percentile(df['stories'], 75)
print(Q3)
IQR = Q3 - Q1
df = df[(df['stories'] >= Q1 - 1.5 * IQR) & (df['stories'] <= Q3 + 1.5 * IQR)]

Q1 = np.percentile(df['parking'], 25)
Q3 = np.percentile(df['parking'], 75)
print(Q3)
IQR = Q3 - Q1
df = df[(df['parking'] >= Q1 - 1.5 * IQR) & (df['parking'] <= Q3 + 1.5 * IQR)]

df

"""## Dropping Duplicates"""

df = df.drop_duplicates()
df

# Drop rows where 'furnishingstatus' is NaN
df= df.dropna(subset=['furnishingstatus'])

print("\nDataFrame after dropping rows with NaN in 'furnishingstatus':")
df

furnishingstatus_count = df.furnishingstatus.value_counts()
furnishingstatus_count

"""# Encode"""

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoding_col = ['furnishingstatus','prefarea','airconditioning','hotwaterheating','basement','guestroom','mainroad']
for col in encoding_col:
    df[col]=encoder.fit_transform(df[col])

df

"""# Correlation"""

correlation_matrix = df.corr() #calculating correlation coeffecients between different features
correlation_matrix

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", linewidths=0.5, cbar=True)
plt.show()

sns.pairplot(df)
plt.show()

df.plot(kind='scatter', x='price',y='area')

"""# Classification - LOGICAL REGRESSION"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Separate the target from the features
target = df['furnishingstatus']
data = df.drop(['furnishingstatus'], axis=1)

# remove the ID column as it is not relevant
# data = data.drop(['price'], axis=1)
# data = data.drop(['area'], axis=1)
# data = data.drop(['bedrooms'], axis=1)
data = data.drop(['bathrooms'], axis=1)
# data = data.drop(['stories'], axis=1)
data = data.drop(['mainroad'], axis=1)
# akid drop mainroad
# data = data.drop(['guestroom'], axis=1)
#not droping guestroom by3le el acc
data = data.drop(['basement'], axis=1)
data = data.drop(['hotwaterheating'], axis=1)
data = data.drop(['airconditioning'], axis=1)
# data = data.drop(['parking'], axis=1)
# data = data.drop(['prefarea'], axis=1)
# akid not drop parking and prefarea

# Normalize the data
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)


# load the data
X_train, X_test, y_train, y_test =  train_test_split(df, target, test_size=0.3, random_state=42)

# Show the first 5 rows of the dataframe
X_train.head()

# Do logistic regression on the iris dataset
from sklearn.linear_model import LogisticRegression


# Create a logistic regression model
log_model = LogisticRegression()

# Train the model
log_model.fit(X_train, y_train)

# Make predictions on the test data
predictions = log_model.predict(X_test)

predictions

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# calculate the confusion matrix
conf_mat = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()

from sklearn.metrics import accuracy_score

# Calculate the accuracy
lr_accuracy = accuracy_score(y_test, predictions)

print('Accuracy:', lr_accuracy*100, '%')

from sklearn.metrics import precision_score

# Calculate the percision
lr_precision = precision_score(y_test, predictions, average='weighted')

print('Precision:', lr_precision*100, '%')

from sklearn.metrics import recall_score

# Calculate the recall
lr_recall = recall_score(y_test, predictions, average='weighted')
print('Recall:', lr_recall*100, '%')

from sklearn.metrics import f1_score

# Calculate the F1 score
lr_f1 = f1_score(y_test, predictions, average='weighted')

print('F1 score:', lr_f1*100, '%')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Binarize the y_test for two classes (0 and 1)
y_test_bin = (y_test == 1).astype(int)  # Adjust the class of interest (1 here)

# Get the predicted probabilities for the positive class
y_score = log_model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')  # Updated label to clarify
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

"""# Decision Tree Classification"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size = 0.25, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.predict([[9100000,6000,	4,	2,0, 2, 0]]))

"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)

# Convert y_test to a NumPy array before reshaping
y_test_array = y_test.values if hasattr(y_test, 'values') else y_test  # Handles Series or array-like
y_test_array = y_test_array.reshape(len(y_test_array), 1)

# Reshape y_pred as well
y_pred_array = y_pred.reshape(len(y_pred), 1)

# Concatenate predictions and actual values
np.concatenate((y_pred_array, y_test_array), axis=1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
accuracy_score(y_test, y_pred)

from sklearn.metrics import precision_score

# Calculate the percision
lr_precision = precision_score(y_test, y_pred, average='weighted')

print('Precision:', lr_precision*100, '%')

from sklearn.metrics import recall_score

# Calculate the recall
lr_recall = recall_score(y_test, y_pred, average='weighted')
print('Recall:', lr_recall*100, '%')

from sklearn.metrics import f1_score

# Calculate the F1 score
lr_f1 = f1_score(y_test, y_pred, average='weighted')

print('F1 score:', lr_f1*100, '%')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Binarize the y_test for two classes (0 and 1)
y_test_bin = (y_test == 1).astype(int)  # Adjust the class of interest (1 here)

# Get the predicted probabilities for the positive class
y_score = log_model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')  # Updated label to clarify
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

"""# Creating a random forest"""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.predict([[9100000,6000,	4,	2,0, 2, 0]]))

y_pred = classifier.predict(X_test)

# Convert y_test to a NumPy array before reshaping
y_test_array = y_test.values if hasattr(y_test, 'values') else y_test  # Handles Series or array-like
y_test_array = y_test_array.reshape(len(y_test_array), 1)

# Reshape y_pred as well
y_pred_array = y_pred.reshape(len(y_pred), 1)

# Concatenate predictions and actual values
np.concatenate((y_pred_array, y_test_array), axis=1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
accuracy_score(y_test, y_pred)

from sklearn.metrics import precision_score

# Calculate the percision
lr_precision = precision_score(y_test, y_pred, average='weighted')

print('Precision:', lr_precision*100, '%')

from sklearn.metrics import recall_score

# Calculate the recall
lr_recall = recall_score(y_test, y_pred, average='weighted')
print('Recall:', lr_recall*100, '%')

from sklearn.metrics import f1_score

# Calculate the F1 score
lr_f1 = f1_score(y_test, y_pred, average='weighted')

print('F1 score:', lr_f1*100, '%')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Binarize the y_test for two classes (0 and 1)
y_test_bin = (y_test == 1).astype(int)  # Adjust the class of interest (1 here)

# Get the predicted probabilities for the positive class
y_score = log_model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')  # Updated label to clarify
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

"""# retry every thing with PCA

split data
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size = 0.2, random_state = 0)

"""scale"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""apply PCA"""

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_trainPCA = pca.fit_transform(X_train)
X_testPCA = pca.transform(X_test)

plt.scatter(x=X_trainPCA[:,0],y=X_trainPCA[:,1])
plt.scatter(x=X_testPCA[:,0],y=X_testPCA[:,1])

"""# UnSupervised"""

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_trainPCA)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_trainPCA)

import numpy as np
# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_trainPCA[y_kmeans == 0, 0], X_trainPCA[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_trainPCA[y_kmeans == 1, 0], X_trainPCA[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_trainPCA[y_kmeans == 2, 0], X_trainPCA[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='yellow', edgecolor='black', label='Centroids')
plt.title('Clusters of PCA-transformed data')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid()
plt.show()

"""# Hierarchical Clustering"""

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X_trainPCA, method = 'ward'))
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
y_hc = hc.fit_predict(X_trainPCA)

plt.scatter(X_trainPCA[y_hc == 0, 0], X_trainPCA[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_trainPCA[y_hc == 1, 0], X_trainPCA[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_trainPCA[y_hc == 2, 0], X_trainPCA[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.legend()
plt.show()

from sklearn.metrics import silhouette_score
print(silhouette_score(X_trainPCA, y_hc))

"""# Comparison With Ground Truth"""

def plot_clusters_with_ground_truth(X, y_pred, y_true, title, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='plasma', label='Ground Truth')
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='plasma', marker='x', label='Predicted Clusters')
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.colorbar(label='Furnishing Status')
    plt.show()

plot_clusters_with_ground_truth(X_trainPCA, y_kmeans, y_train, "KMeans Clustering vs. Ground Truth", labels=np.unique(y_train))
plot_clusters_with_ground_truth(X_trainPCA, y_hc, y_train, "Agglomerative Clustering vs. Ground Truth", labels=np.unique(y_train))

"""# PCA & LOGICAL REGRESSION"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_trainPCA, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_testPCA)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

from sklearn.metrics import precision_score

# Calculate the percision
lr_precision = precision_score(y_test, y_pred, average='weighted')

print('Precision:', lr_precision*100, '%')

from sklearn.metrics import recall_score

# Calculate the recall
lr_recall = recall_score(y_test, y_pred, average='weighted')
print('Recall:', lr_recall*100, '%')

from sklearn.metrics import f1_score

# Calculate the F1 score
lr_f1 = f1_score(y_test, y_pred, average='weighted')

print('F1 score:', lr_f1*100, '%')

"""Decision tree"""

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_trainPCA, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_testPCA)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

from sklearn.metrics import precision_score

# Calculate the percision
lr_precision = precision_score(y_test, y_pred, average='weighted')

print('Precision:', lr_precision*100, '%')

from sklearn.metrics import recall_score

# Calculate the recall
lr_recall = recall_score(y_test, y_pred, average='weighted')
print('Recall:', lr_recall*100, '%')

from sklearn.metrics import f1_score

# Calculate the F1 score
lr_f1 = f1_score(y_test, y_pred, average='weighted')

print('F1 score:', lr_f1*100, '%')

"""Random Forest"""

from sklearn.ensemble import RandomForestClassifier

# Train the classifier on the training data
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_trainPCA, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_testPCA)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

from sklearn.metrics import precision_score

# Calculate the percision
lr_precision = precision_score(y_test, y_pred, average='weighted')

print('Precision:', lr_precision*100, '%')

from sklearn.metrics import recall_score

# Calculate the recall
lr_recall = recall_score(y_test, y_pred, average='weighted')
print('Recall:', lr_recall*100, '%')

from sklearn.metrics import f1_score

# Calculate the F1 score
lr_f1 = f1_score(y_test, y_pred, average='weighted')

print('F1 score:', lr_f1*100, '%')