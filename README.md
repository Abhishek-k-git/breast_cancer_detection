![Banner](https://images.unsplash.com/photo-1576669801343-117bb4054118?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTh8fGJyZWFzdCUyMGNhbmNlciUyMGRldGVjdGlvbnxlbnwwfHwwfHw%3D&auto=format&fit=crop&w=1920&h=400&q=60)

# Breast cancer detection
### Breast cancer detection using KNN and Support Vector Machine Algorithm

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/navendu-pottekkat/awesome-readme)
![GitHub issues](https://img.shields.io/github/issues-raw/navendu-pottekkat/awesome-readme)
![GitHub pull requests](https://img.shields.io/github/issues-pr/navendu-pottekkat/awesome-readme)
![GitHub](https://img.shields.io/github/license/navendu-pottekkat/awesome-readme)

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

> **Features are classified into ten different real-valued classes for each nucleus:**
1. radius (mean of distances from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter^2 / area - 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension ("coastline approximation" - 1)
 

| Data Set Characteristics | Multivariate | Number of Instances | 569 | Area | Life |
| -- | -- | -- | -- | -- | -- | 
| Attribute Characteristics | Real | Number of Attributes | 32 | Date Donated | 1995-11-01 |
| Associated Tasks | Classification | Missing Values | No | Number of Web Hits | 1771156 |

> Problem statement

Breast cancer is a widely occurring cancer in women worldwide and is related to high mortality. The objective of this review is to present several approaches to investigate the application of multiple algorithms based on machine learning (ML) approach and for early detection of breast cancer.

> **Implemented Algorithms:**
1. K-Nearest Neighbor(KNN)
2. Support Vector Machine (SVM)

**K-Nearest Neighbor(KNN) -**
It is a type of supervised machine learning. K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.

**Support Vector Machine (SVM) -**
It is also a type of supervised machine learning, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning. The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane. SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine. Consider the below diagram in which there are two different categories that are classified using a decision boundary or hyperplane:

<img src="https://github.com/Abhishek-k-git/breast_cancer_detection/blob/main/images/svm.png" height="200" alt="svm" />

Download dataset from [here](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))

> Data preparation

The data we have, may contain some impurities like null value or notations. These doesn't add any value, instead it causes problem in data visualization or model building. We need to first find that noises, and remove it before we build our model.

``` pd.read_csv('data.csv') ``` - convert to pandas dataframe

``` data.info() ``` - returns data info like, data type
.Here we can easily see that ``` bare_nuclei - 699 non-null - object ``` which means *bare_nuclei* has some impurities/data which is not int
```
bare_nuclei = pd.DataFrame(data['bare_nuclei'].str.isdigit())
bare_nuclei.value_counts()
```
| bare_nuclei | -- |
| -- | -- |
| True | 683 |
| False | 16 |
| dtype: int64 |

```
df = data.replace('?', np.nan)
df['bare_nuclei'].value_counts()
```
```
df.describe().T
```
![data](https://github.com/Abhishek-k-git/breast_cancer_detection/blob/main/images/data.png)

> **Data visualization:**

After dataprocessing or cleaning, it is very crucial to visualize dataset, there are many datavisualization tool out there. But here we use [seaborn](https://seaborn.pydata.org/), which is a python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

> **Build model:**

Now data is divided into two sets one is *training dataset* which is used to train the model (just like a new born child learns by sensing things around him), the other dataset is *testing dataset* which is used to evaluate or predict the accuracy of model. The machine uses its model, apply to testing dataset to give out predicted results. The predicted output then compared to final result in actual dataset (In this case it is labeled as *class*). That's why it is necessary to first drop that column named class, before we train our model.

```
X = df.drop('class', axis = 1)
y = df['class']
X.shape, y.shape

# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 32)
```

> K-neighbors classification
```
# from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
KNN.fit(x_train, y_train)
```
```
knn_predict = KNN.predict(x_test)
print('predicted class value: ')
knn_predict
```
```
predicted class value: 
array([4, 4, 4, 2, 4, 2, 4, 4, 2, 2, 2, 2, 4, 2, 2, 4, 4, 2, 4, 2, 2, 4,
       4, 2, 4, 2, 4, 2, 4, 4, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 4, 2, 2, 2,
       4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 4, 4, 4, 2, 2, 4, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2,
       4, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2,
       4, 4, 2, 2, 4, 2, 4, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 4, 4, 4, 4,
       4, 2, 2, 4, 2, 4, 4, 4, 2, 2, 2, 4, 2, 2, 4, 2, 4, 2, 2, 4, 2, 2,
       2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 4, 4, 2, 2, 2, 2,
       2, 2, 4, 4, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 4, 4, 2, 2, 2, 2, 2, 4,
       4, 4, 2, 2, 4, 2, 4, 4, 4, 4, 4, 2], dtype=int64)
```
```
actual class value: 
array([4, 4, 4, 2, 4, 2, 4, 4, 2, 2, 2, 2, 4, 2, 2, 4, 4, 2, 4, 2, 2, 4,
       4, 2, 4, 2, 4, 2, 4, 4, 2, 4, 2, 2, 2, 4, 2, 2, 4, 2, 4, 2, 2, 2,
       4, 2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 4, 4, 4, 2, 2, 4, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2, 4, 4, 2, 2,
       4, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2,
       4, 4, 2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 4, 4, 4, 4,
       4, 2, 2, 4, 2, 4, 4, 4, 4, 2, 2, 4, 2, 2, 4, 2, 4, 2, 2, 4, 2, 2,
       2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 4, 2, 2, 2, 2, 2,
       2, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 4, 4, 2, 2, 2, 2, 2, 4,
       4, 4, 2, 4, 4, 2, 4, 4, 4, 4, 4, 2], dtype=int64)
```
```
from scipy.stats import zscore
print('KNN predion score :{0: 2g}%'.format(KNN.score(x_test, y_test)*100))
```
**KNN predion score :** ```94.7619%```

> K-neighbors classification

```
#from sklearn.svm import SVC
SVC = SVC()
SVC.fit(x_train, y_train)
```
```
print('SVM prediction')
svm_predict = SVC.predict(x_test)
svm_predict
```
```
SVM prediction
array([4, 4, 4, 2, 4, 2, 4, 4, 2, 2, 2, 2, 4, 2, 2, 4, 4, 2, 4, 2, 2, 4,
       4, 2, 4, 2, 4, 2, 4, 4, 2, 4, 2, 4, 2, 4, 2, 2, 4, 2, 4, 2, 2, 2,
       4, 2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 4, 4, 4, 2, 2, 4, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2,
       4, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2,
       4, 4, 2, 2, 4, 2, 4, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 4, 4, 4, 4, 4,
       4, 2, 2, 4, 2, 4, 4, 4, 4, 2, 2, 4, 4, 2, 4, 2, 4, 2, 2, 4, 2, 2,
       2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 4, 4, 2, 2, 2, 2,
       2, 2, 4, 4, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 4, 4, 2, 2, 2, 2, 2, 4,
       4, 4, 2, 4, 4, 2, 4, 4, 4, 4, 4, 2], dtype=int64)
```
**SVM predion score :** ```95.7143%```

![matrix](https://github.com/Abhishek-k-git/breast_cancer_detection/blob/main/images/matrix.png)
