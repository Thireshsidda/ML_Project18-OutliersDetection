# ML_Project18-OutliersDetection

### Outlier Detection in the Titanic Dataset
This project explores outlier detection and its impact on machine learning models using the Titanic dataset. It analyzes the "age" and "fare" features to identify potential outliers and demonstrates how handling outliers can influence model performance.

### Machine Learning Algorithms and Outlier Sensitivity

The provided code includes a table summarizing the sensitivity of various machine learning algorithms to outliers. This serves as a reference for choosing appropriate algorithms based on the presence of outliers in your data.

### Data Exploration and Outlier Identification

##### Data Loading and Inspection:

The code loads the "titanic.csv" dataset using pandas and displays the first few rows.

The code checks for missing values using df.isna().sum().

##### Visualization:

The code utilizes seaborn's distplot (deprecated) function to visualize the distribution of the "age" feature, initially with missing values and then after filling them with a placeholder value (100). This helps assess the normality of the distribution.

A histogram (df.age.hist(bins=50)) is created to visualize the distribution of "age".

A boxplot (df.boxplot(column='age')) is generated to visually identify potential outliers.

##### Outlier Detection for Normally Distributed Data:

Assuming a normal distribution for "age", the code calculates upper and lower bounds using the mean, standard deviation, and a threshold of 3 standard deviations. However, the results indicate that the "age" feature is not normally distributed.

##### Outlier Detection for Skewed Data:

The code analyzes the "fare" feature using similar techniques as with "age". It identifies a right-skewed distribution through visualization (histogram and boxplot).

Interquartile Range (IQR) is used to calculate upper and lower bounds for outliers. Additionally, extreme outlier bounds are determined using a threshold of 3 times the IQR.

### Outlier Treatment (Capping)

The code creates a copy of the data (data = df.copy()) to avoid modifying the original dataset.

It replaces values in "age" exceeding the upper bound (73) with the upper bound value.

Similarly, values in "fare" exceeding the upper bound (100) are replaced with the upper bound.
\
Histograms are created to visualize the distributions of "age" and "fare" after capping outliers.

### Machine Learning Model Training

##### Data Splitting:

The data is split into training and testing sets using train_test_split from scikit-learn for model evaluation.

##### Logistic Regression:

A Logistic Regression model is trained on the features "age" and "fare" (after filling missing values with 0) to predict passenger survival ("survived") in the Titanic disaster.

The model's performance is evaluated using the confusion matrix, accuracy score, and ROC AUC score.

##### Random Forest Classifier:

A Random Forest Classifier model is trained on the same data and evaluated similarly to the Logistic Regression model.

### Results

The results show that handling outliers by capping extreme values can improve the accuracy of the Logistic Regression model (slightly) in this example. This highlights the importance of outlier detection and treatment for certain machine learning algorithms.

### Note:

The code uses seaborn.distplot which is deprecated. Consider using seaborn.displot or plt.hist for future projects.

### Further Exploration

Explore different outlier detection methods like z-scores, anomaly detection algorithms, and isolation forests.

Experiment with various outlier treatment techniques beyond capping, such as winsorization or removal.

Evaluate the impact of outlier handling on different machine learning algorithms and datasets.
