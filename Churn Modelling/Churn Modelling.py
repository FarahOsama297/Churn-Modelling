import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#load data

data = pd.read_csv('Churn_Modelling.csv')
data.head(10)

#preprocessing

#drop nulls

data = data.dropna()

data['Age'].fillna(data['Age'].median(), inplace=True)

Encoding Categorical Variables:

gender_mapping = {'Male': 1, 'Female': 0}
data['Gender'] = data['Gender'].map(gender_mapping)

label_encoder = LabelEncoder()
data['Geography'] = label_encoder.fit_transform(data['Geography'])

X = data.drop(['Exited','Surname'], axis=1)
y = data['Exited']

X.head(10)

Feature Scaling:

#scaler = StandardScaler()
#numerical_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
#X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
#X.head(10)

#Splitting the dataset into train and test sets:

X = data.drop(['Exited','Surname'], axis=1)
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Training set - Features:", X_train.shape, "Target:", y_train.shape)
print("Testing set - Features:", X_test.shape, "Target:", y_test.shape)

#decitision tree model

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy for Decision Tree: {accuracy_train:.2f}")

y_pred_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy for Decision Tree: {accuracy_test:.2f}")

report = classification_report(y_test, y_pred_test)
print("Classification Report:")
print(report)

#to improve accuracy

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [None, 5, 10, 15],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at a leaf node
    # You can add more parameters to explore
}

model = DecisionTreeClassifier()

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Best Model Training Accuracy: {accuracy_train:.2f}")

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)


#SVM model


from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {accuracy_train:.2f}")

y_pred = model.predict(X_test)

accuracy2 = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy2:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


#to improve accuracy of SVM by using GridSearchCV

# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'C': [0.1, 1, 10, 100],  # Regularization parameter values
#     'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],  # Kernel coefficient for 'rbf'
#     # You can add more parameters to explore
# }


# model = SVC(kernel='rbf')

# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_

# y_pred_train = model.predict(X_train)
# accuracy_train = accuracy_score(y_train, y_pred_train)
# print(f"Training Accuracy: {accuracy_train:.2f}")

# y_pred = best_model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Best Model Accuracy: {accuracy:.2f}")

# report = classification_report(y_test, y_pred)
# print("Classification Report:")
# print(report)

# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

to improve accuracy of SVM by using RandomizedSearchCV

# from sklearn.model_selection import RandomizedSearchCV

# param_dist = {
#     'C': [0.1, 1, 10],  # Reducing the number of values for 'C'
#     'gamma': ['scale', 'auto', 0.1, 0.01],  # Reducing the number of values for 'gamma'
#     # You can add more parameters to explore
# }

# model = SVC(kernel='rbf')

# random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=0)
# random_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_

# y_pred_train = model.predict(X_train)
# accuracy_train = accuracy_score(y_train, y_pred_train)
# print(f"Training Accuracy: {accuracy_train:.2f}")

# y_pred = best_model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Best Model Accuracy: {accuracy:.2f}")

# report = classification_report(y_test, y_pred)
# print("Classification Report:")
# print(report)

# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

#Random Forest Classifier model

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {accuracy_train:.2f}")

y_pred = rf_model.predict(X_test)

accuracy4 = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy4:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

#improve random forest

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],  # Adjust the number of trees
    'max_depth': [None, 5, 10, 20],  # Maximum depth of trees
    'min_samples_leaf': [1, 2, 4]  # Minimum samples per leaf
    # You can add more parameters to explore
}

rf_model = RandomForestClassifier(random_state=0)

grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {accuracy_train:.2f}")

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

#XGBClassifier model

from xgboost import XGBClassifier

model = XGBClassifier(max_depth=2, min_child_weight=15, gamma=1.0, subsample=0.4, colsample_bytree=0.4)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {accuracy_train:.2f}")

y_pred = model.predict(X_test)

accuracy5 = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy5:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

#improve XGBClassifier model

from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],  # Adjust the learning rate
    'max_depth': [3, 6, 9],  # Maximum depth of trees
    'subsample': [0.6, 0.8, 1.0],  # Subsample ratio
    # You can add more parameters to explore
}

model = XGBClassifier()

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {accuracy_train:.2f}")

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

#K-Nearest Neighbours model

#from sklearn.neighbors import KNeighborsClassifier

#model = KNeighborsClassifier(n_neighbors=500)  # Set the number of neighbors, you can adjust this parameter
#model.fit(X_train, y_train)

#y_pred_train = model.predict(X_train)

#import numpy as np
#y_pred_train_altered = np.random.permutation(y_pred_train)

#y_pred_train = best_model.predict(X_train)
#accuracy_train = accuracy_score(y_train, y_pred_train)
#print(f"Training Accuracy: {accuracy_train:.2f}")

#y_pred = model.predict(X_test)

#accuracy6 = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy6:.2f}")

#report = classification_report(y_test, y_pred)
#print("Classification Report:")
#print(report)

#Stochastic Gradient Descent model

from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='hinge', max_iter=1000, random_state=0)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {accuracy_test:.2f}")

y_pred = model.predict(X_test)

accuracy7 = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy7:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

improve Stochastic Gradient Descent model

from sklearn.model_selection import GridSearchCV

param_grid = {
    'loss': ['squared_hinge'],  # Different loss functions
    'alpha':[ 0.0001],  # Regularization parameter values
    # You can add more parameters to explore
}

model = SGDClassifier(max_iter=1000, random_state=0)

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy (Best Decision Tree Model): {accuracy_train:.2f}")

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

#build model from scratch

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_shuffled, y_shuffled = shuffle(X_train_scaled, y_train, random_state=0)

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(X_shuffled.shape[1],)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

history = model.fit(X_shuffled, y_shuffled, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

y_pred_train = model.predict(X_train_scaled)

y_pred_train_classes = (y_pred_train > 0.5).astype("int32")  # Convert

train_accuracy = accuracy_score(y_train, y_pred_train_classes)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

y_pred_test_probs = model.predict(X_test_scaled)
y_pred_test_classes = (y_pred_test_probs > 0.5).astype("int32")

report = classification_report(y_test, y_pred_test_classes)
print("Classification Report for Test Data:")
print(report)