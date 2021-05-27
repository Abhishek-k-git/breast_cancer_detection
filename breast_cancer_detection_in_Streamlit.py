# pip install streamlit

# streamlit run breast_cancer.py

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
matplotlib.use('Agg')
from PIL import Image

st.title('Breast cancer study in Streamlit')
image = Image.open('breast_cancer.png')
st.image(image, use_column_width = True)

def main():
	activities = ['Introduction', 'EDA', 'Visualisation', 'model']
	option = st.sidebar.selectbox('Selection options:', activities)

	if option == 'Introduction':
		st.balloons()
		st.markdown('This Breast cancer detection in machine learning, the chances of cancer tratment always always depends on its early detection, as it can not be prevented this should need to be early detected. Machine learning models with good dataset and algorithm can predict satisfing results thats why it plays vital role in the field of cancer detection. Here we can predict different models with different accuracies and deside which shoots best.')

	elif option == 'EDA':
		st.subheader('Exploratory data analysis')

		data = st.file_uploader('Upload dataset file', type = ['csv', 'xlsx', 'txt', 'json'])
		
		if data is not None:

			st.success('data loaded successfully')
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox('Display shape'):
				st.write(df.shape)

			if st.checkbox('Display columns'):
				st.write(df.columns)

			if st.checkbox('Select multiple columns'):
				selected_cols = st.multiselect('Select multiple columns which are most prefered', df.columns)
				new_df = df[selected_cols]
				st.dataframe(new_df)

			if st.checkbox('Display summary'):
				st.write(new_df.describe().T)

			if st.checkbox('Display null values'):
				st.write(df.isnull().sum())

			if st.checkbox('Display the data type'):
				st.write(df.dtypes)

			if st.checkbox('Display correlation with each columns'):
				st.write(df.corr())

			

	elif option == 'Visualisation':
		st.subheader('Visualisation')

		data = st.file_uploader('Upload dataset file', type = ['csv', 'xlsx', 'txt', 'json'])
		
		if data is not None:

			st.success('data loaded successfully')
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox('Display heatmap'):
				st.set_option('deprecation.showPyplotGlobalUse', False)
				st.write(sns.heatmap(df.corr(), vmax = 1, square = True, annot = True, cmap = 'viridis'))
				st.pyplot()

			if st.checkbox('Display pairplot'):
				st.set_option('deprecation.showPyplotGlobalUse', False)
				st.write(sns.pairplot(df, diag_kind = 'kde'))
				st.pyplot()

			if st.checkbox('Display piechart'):
				all_columns = df.columns.tolist()
				pie_columns = st.selectbox('Select columns to display', all_columns)
				st.set_option('deprecation.showPyplotGlobalUse', False)
				piechart = df[pie_columns].value_counts().plot.pie(autopct = '%1.1f%%')
				st.write(piechart)
				st.pyplot()
			

	elif option == 'model':
		st.subheader('Model building')

		data = st.file_uploader('Upload dataset file', type = ['csv', 'xlsx', 'txt', 'json'])
		
		if data is not None:

			st.success('data loaded successfully')
			df = pd.read_csv(data)
			st.write('Your dataset')
			st.dataframe(df.head())

			X = df.iloc[:, 1: -5]
			y = df.iloc[:, -1]
			st.info('X data')
			st.write(X.head())
			st.write(X.shape)
			st.write('')
			st.info('y data')
			st.write(y.head())
			st.write(y.shape)

			if st.checkbox('Select multiple columns'):
				selected_cols = st.multiselect('Select multiple columns which are most prefered', df.columns)
				new_df = df[selected_cols]
				st.dataframe(new_df)

				X = new_df.iloc[:, 0:-1]
				y = new_df.iloc[:, -1]

				st.write(X.head())
				st.write(X.shape)
				st.write(y.head())
				st.write(y.shape)

			seed = st.sidebar.slider('seed', 1, 200)
			classifier_name = st.sidebar.selectbox('Select your prefered classifier', ('KNN', 'SVM', 'LR', 'naive_bayes', 'decision tree'))

			def add_parameter(name_of_clf):
				params = dict()

				if name_of_clf == 'SVM':
					c = st.sidebar.slider('c', 1, 15)
					params['c'] = c
					return params

				elif name_of_clf == 'KNN':
					k = st.sidebar.slider('k', 1, 15)
					params['k'] = k
					return params

			params = add_parameter(classifier_name)
	
			def get_classifier(name_of_clf, params):
				clf = None
				if name_of_clf == 'SVM':
					clf = svm.SVC(C = params['c'])
				elif name_of_clf == 'KNN':
					clf = KNeighborsClassifier(n_neighbors = params['k'])
				elif name_of_clf == 'LR':
					clf = LogisticRegression()
				elif name_of_clf == 'naive_bayes':
					clf = GaussianNB()
				elif name_of_clf == 'decision tree':
					clf = DecisionTreeClassifier()
				else:
					st.warning('Select your choice of algorithm')
				return clf

			clf = get_classifier(classifier_name, params)
			x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)
			clf.fit(x_train, y_train)
			y_pred = clf.predict(x_test)
			st.write('Predictions : ', y_pred)
			accuracy = accuracy_score(y_test, y_pred)
			st.write('Name of classifier :', classifier_name)
			st.write('Accuracy', accuracy)


if __name__ == '__main__':
	main()	