# HW2- SVM & Random Forests 

<h3>Load data</h3>


```python
# import the required libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# read the csv data from local drive 
data = pd.read_csv("intrusion.csv")
print(data.shape)
data.head(5)
```

    (10100, 4)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attribute1</th>
      <th>Attribute2</th>
      <th>Attribute3</th>
      <th>Intrusion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.157322</td>
      <td>1.922947</td>
      <td>3.223735</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.711650</td>
      <td>5.534262</td>
      <td>1.519069</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.062710</td>
      <td>0.913824</td>
      <td>0.715046</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.100344</td>
      <td>6.153463</td>
      <td>2.250014</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.145073</td>
      <td>6.553025</td>
      <td>2.214019</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<h3>Scatter Plot</h3>


```python
# Draw scatter plot for Attribute 1 and Attribute 2
a1 = data["Attribute1"]
a2 = data["Attribute2"]
a3 = data["Attribute3"]
colors = data["Intrusion"]

fig = plt.figure()
ax = plt.axes()
scatter = ax.scatter(a1, a2, c = colors, cmap='viridis', alpha=0.3)
ax.set(xlabel = 'Attribute 1', ylabel = 'Attribute 2')
ax.set_title("Attribute 1 and Attribute 2 (colored by “Intrusion”)")
ax.legend(*scatter.legend_elements(), title = "Intrusion")
```




    <matplotlib.legend.Legend at 0x161b8746400>




    
![png](output_4_1.png)
    



```python
# Draw scatter plot for Attribute 2 and Attribute 3 
fig = plt.figure()
ax = plt.axes()
scatter = ax.scatter(a2, a3, c = colors, cmap='viridis', alpha=0.3)
ax.set(xlabel = 'Attribute 2', ylabel = 'Attribute 3')
ax.set_title("Attribute 2 and Attribute 3 (colored by “Intrusion”)")
ax.legend(*scatter.legend_elements(), title = "Intrusion")
```




    <matplotlib.legend.Legend at 0x161b8871100>




    
![png](output_5_1.png)
    


<h3>Split the data into train and test sets using sklearn</h3>


```python
from sklearn.model_selection import train_test_split

x = data.iloc[:, 0:3].to_numpy() # Select Column of Attribute 1, 2 and 3 
y = data[["Intrusion"]].to_numpy() # Select Intrusion column 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (6767, 3)
    (3333, 3)
    (6767, 1)
    (3333, 1)
    

<h3>Training a SVC</h3>


```python
# Training a SVC using the Poly kernels 
from sklearn.svm import SVC

model = SVC(kernel='poly', degree=3)
model.fit(x_train, y_train)
ypred = model.predict(x_test) 
```

    C:\Users\MEI-KUEI LU\anaconda3\lib\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    


```python
# Classification Report 
from sklearn import metrics
print(metrics.classification_report(y_test, ypred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      3301
               1       0.92      0.69      0.79        32
    
        accuracy                           1.00      3333
       macro avg       0.96      0.84      0.89      3333
    weighted avg       1.00      1.00      1.00      3333
    
    


```python
# Confusion matrix of SVC using the Poly kernels 
from sklearn.metrics import confusion_matrix

confused = confusion_matrix(y_test,ypred)
print(confused)

#### Confusion Matrix Interpretation 
# true positives: 3299 (Intrusion = 0 is identified as Intrusion = 0)
# true negatives: 22 (Intrusion = 1 is identified as Intrusion = 1)
# false positives: 2 (Intrusion = 0 is identified as Intrusion = 1)
# false negatives: 10 (Intrusion = 1 is identified as Intrusion = 0)
```

    [[3299    2]
     [  10   22]]
    


```python
# Training a SVC using the linear kernels 
model = SVC(kernel='linear', C=1E10)
model.fit(x_train, y_train) 
ypred = model.predict(x_test) 
```

    C:\Users\MEI-KUEI LU\anaconda3\lib\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    


```python
# Classification Report 
from sklearn import metrics
print(metrics.classification_report(y_test, ypred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      3301
               1       0.86      0.78      0.82        32
    
        accuracy                           1.00      3333
       macro avg       0.93      0.89      0.91      3333
    weighted avg       1.00      1.00      1.00      3333
    
    


```python
# Confusion matrix of SVC using the linear kernels 
from sklearn.metrics import confusion_matrix

confused = confusion_matrix(y_test,ypred)
print(confused)

#### Confusion Matrix Interpretation 
# true positives: 3297 (Intrusion = 0 is identified as Intrusion = 0)
# true negatives: 25 (Intrusion = 1 is identified as Intrusion = 1)
# false positives: 4 (Intrusion = 0 is identified as Intrusion = 1)
# false negatives: 7 (Intrusion = 1 is identified as Intrusion = 0)
```

    [[3297    4]
     [   7   25]]
    


```python
# Training a SVC using the RBF kernels 
model = SVC(kernel='rbf', random_state=0, gamma=.01, C=1)
model.fit(x_train, y_train)
ypred = model.predict(x_test)
```

    C:\Users\MEI-KUEI LU\anaconda3\lib\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    


```python
# Classification Report 
from sklearn import metrics
print(metrics.classification_report(y_test, ypred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      3301
               1       0.92      0.72      0.81        32
    
        accuracy                           1.00      3333
       macro avg       0.96      0.86      0.90      3333
    weighted avg       1.00      1.00      1.00      3333
    
    


```python
# Confusion matrix of SVC using the RBF kernels 
from sklearn.metrics import confusion_matrix

confused = confusion_matrix(y_test,ypred)
print(confused)

#### Confusion Matrix Interpretation 
# true positives: 3299 (Intrusion = 0 is identified as Intrusion = 0)
# true negatives: 23 (Intrusion = 1 is identified as Intrusion = 1)
# false positives: 2 (Intrusion = 0 is identified as Intrusion = 1)
# false negatives: 9 (Intrusion = 1 is identified as Intrusion = 0)
```

    [[3299    2]
     [   9   23]]
    

<h3>Train a Random Forest Classifier</h3>


```python
from sklearn.ensemble import RandomForestClassifier                                               

model = RandomForestClassifier(n_estimators=1000)
model.fit(x_train, y_train)
ypred = model.predict(x_test)
```

    <ipython-input-92-4cac915a3de8>:4: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      model.fit(x_train, y_train)
    


```python
# Classification Report 
from sklearn import metrics
print(metrics.classification_report(y_test, ypred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      3301
               1       0.90      0.81      0.85        32
    
        accuracy                           1.00      3333
       macro avg       0.95      0.91      0.93      3333
    weighted avg       1.00      1.00      1.00      3333
    
    

Confusion Matrix Interpretation 

true positives: 26 

true negatives: 3298 

false positives: 3 (Intrusion = 0 is identified as Intrusion = 1)

false negatives: 6 (Intrusion = 1 is identified as Intrusion = 0)


```python
# Confusion matrix of RFC
from sklearn.metrics import confusion_matrix

confused = confusion_matrix(y_test,ypred)
print(confused)
```

    [[3298    3]
     [   6   26]]
    


```python
# Get and reshape confusion matrix data
import seaborn as sns

matrix = confusion_matrix(y_test, ypred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10}, cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Intrusion = 0', 'Intrusion = 1']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks+0.5, class_names, rotation=0)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
```


    
![png](output_23_0.png)
    



```python
# View accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, ypred)
```




    0.9972997299729973


