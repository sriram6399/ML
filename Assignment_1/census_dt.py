import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

#function to get accuracies
def get_accuracies(X, y, tree_depths,cv,scoring):
    mean=[]
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth,criterion='entropy',random_state=0)
        tree_model.fit(X, y)
        scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        mean.append(scores.mean())
        accuracy_scores.append(accuracy_score(tree_model.predict(X),y))
    return mean, accuracy_scores
    
  
# function for plotting accuracy results
def plot_accuracies(depths, mean, accuracy_scores):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, mean, label='mean of cross-validation accuracy')
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores,  label='train accuracy')
    ax.set_title("Accuracy at various Depths", fontsize=16)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()
    return fig


def get_train_test_accuracy(X_train, Y_train,X_test,Y_test,depth):
    model = DecisionTreeClassifier(max_depth=depth,criterion='entropy',random_state=0).fit(X_train, Y_train)
    train_accuracy = model.score(X_train, Y_train)
    test_accuracy = accuracy_score(model.predict(X_test), Y_test)
    return train_accuracy,test_accuracy
  

# reading csv files train data
data =  pd.read_csv('census/census-income.data', sep=",")
cat_columns = data.select_dtypes(['object']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: pd.factorize(x)[0])
X_train = data.iloc[:,:-1]
Y_train = data.iloc[:,-1]



# reading csv files test data
test_data =  pd.read_csv('census/census-income.test', sep=",")
test_cat_columns = test_data.select_dtypes(['object']).columns
test_data[cat_columns] = test_data[test_cat_columns].apply(lambda x: pd.factorize(x)[0])
X_test = data.iloc[:,:-1]
Y_test = data.iloc[:,-1]


# fitting trees of depth 2 to 10
tree_depths = range(2,20)
mean,accuracy = get_accuracies(X_train, Y_train, tree_depths,5,'accuracy')
for i in range(0,len(accuracy)):
    print('Training Accuracy for depth',i+2,'is :',accuracy[i])
print('\n')
# plotting accuracy
fig = plot_accuracies(tree_depths,mean, accuracy)

idx = mean.index(max(mean))
best_tree_depth = tree_depths[idx]
best_tree_mean_accuracy = mean[idx]
print('The depth-{} tree achieves the best mean accuracy {} training dataset'.format(best_tree_depth, round(best_tree_mean_accuracy*100,5)))

# train and evaluate a k-depth tree
best_tree_train_accuracy,best_tree_test_accuracy = get_train_test_accuracy(X_train, Y_train,X_test,Y_test ,best_tree_depth)

print('Best Training Accuracy Obtained is ',best_tree_train_accuracy)
print('Best Test Accuracy Obtained is ',best_tree_test_accuracy)
fig.savefig("test.png")