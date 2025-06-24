from sklearn.datasets import load_iris 
iris=load_iris()

x=iris.data 
y=iris.target 
print(x[:5],y[:5])

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest =train_test_split(x,y,test_size=0.4,random_state=1) 
print(iris.data.shape)
print(len(xtrain))
print(len(ytest))

from sklearn.neighbors import KNeighborsClassifier 
knn=KNeighborsClassifier(n_neighbors=1) 
knn.fit(xtrain,ytrain)
pred=knn.predict(xtest)

from sklearn import metrics 
print("Accuracy",metrics.accuracy_score(ytest,pred)) 
print(iris.target_names[2]) 
ytestn=[iris.target_names[i] for i in ytest] 
predn=[iris.target_names[i] for i in pred]

print("predicted Actual")
for i in range(len(pred)):
    print(i," ",predn[i]," ",ytestn[i])

# OUTPUT
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]] [0 0 0 0 0]
# (150, 4)
# 90
# 60
# Accuracy 0.9666666666666667
# virginica
# predicted Actual
# 0   setosa   setosa
# 1   versicolor   versicolor
# 2   versicolor   versicolor
# 3   setosa   setosa
# 4   virginica   virginica
# 5   virginica   versicolor
# 6   virginica   virginica
# 7   setosa   setosa
# 8   setosa   setosa
# 9   virginica   virginica
# 10   versicolor   versicolor
