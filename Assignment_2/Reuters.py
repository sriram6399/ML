import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler as SC
import time
	
def get_train_test_accuracy_NB(X_train, Y_train,X_test,Y_test):
    start = time.time()
    model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True).fit(X_train, Y_train.values.ravel())
    end = time.time() 
    train_accuracy = accuracy_score(model.predict(X_train), Y_train)
    test_accuracy = accuracy_score(model.predict(X_test), Y_test)
    return train_accuracy,test_accuracy,end-start
  
# reading csv files train data
  
X_train = SC().fit_transform(pd.read_csv('HW2_data/train.csv'))

Y_train = pd.read_csv('HW2_data/train_labels.txt')

# reading csv files test data

X_test = SC().fit_transform(pd.read_csv('HW2_data/test.csv')) 
Y_test = pd.read_csv('HW2_data/test_labels.txt')

print(X_train)


# train and evaluate a k-depth tree
train_accuracy,test_accuracy,train_time = get_train_test_accuracy_NB(X_train, Y_train,X_test,Y_test)
print('\n')
print('Training Accuracy Obtained for NB is ',train_accuracy)
print('Test Accuracy Obtained for NB is ',test_accuracy)
print('Training time required for NB is',train_time)
print('\n')