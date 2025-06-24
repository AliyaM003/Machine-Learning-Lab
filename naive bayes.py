import csv 
import math 
import random 
import statistics

def cal_probability(x,mean,stdev):
    exponent=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return(1/(math.sqrt(2*math.pi)*stdev))*exponent

dataset=[] 
dataset_size=0

with open('lab5.csv') as csvfile:
    lines=csv.reader(csvfile)
    for row in lines:
        dataset.append([float(attr) for attr in row]) 


dataset_size=len(dataset)
print("Size of dataset is: ",dataset_size)
train_size=int(0.7*dataset_size) 
print(train_size)
X_train=[] 
X_test=dataset.copy()

training_indexes=random.sample(range(dataset_size),train_size)
for i in training_indexes: 
    X_train.append(dataset[i]) 
    X_test.remove(dataset[i])

classes={}
for samples in X_train: 
    last=int(samples[-1]) 
    if last not in classes:
        classes[last]=[] 
    classes[last].append(samples)
print(classes) 
summaries={}

for classValue,training_data in classes.items(): 
    summary=[(statistics.mean(attribute),statistics.stdev(attribute)) for attribute in zip(*training_data)] 
    del summary[-1]
    summaries[classValue]=summary
print(summaries) 

X_prediction=[]
for i in X_test: 
    probabilities={}
    for classValue,classSummary in summaries.items(): 
        probabilities[classValue]=1 
        for index,attr in enumerate(classSummary): 
            probabilities[classValue]*=cal_probability(i[index],attr[0],attr[1])
	
    best_label,best_prob=None,-1
    for classValue,probability in probabilities.items(): 
        if best_label is None or probability>best_prob:
            best_prob=probability 
            best_label=classValue 
    X_prediction.append(best_label)

correct=0
for index,key in enumerate(X_test):
   if X_test[index][-1]==X_prediction[index]: 
        correct+=1
print("Accuracy: ",correct/(float(len(X_test)))*100)

#OUTPUT
# Size of dataset is:  768
# 537
# {0: [(3.353623188405797, 3.0513330917272925), (110.32463768115942, 25.700965990561418), (68.2695652173913, 18.26197683098768), 
# (19.8, 14.966746084974282), (65.90144927536232, 97.20729242902442), (30.477391304347826, 7.4580812366897495), (0.4189623188405797, 0.2810822756223273), 
# (31.533333333333335, 11.927819088880174)], 1: [(5.015625, 3.7729771633807525), (142.93229166666666, 32.082527726111046), (71.77604166666667, 19.85910710568086), 
# (22.770833333333332, 16.973326266160143), (115.43229166666667, 147.4972556278842), (34.934895833333336, 7.2599295418373915), 
# (0.549921875, 0.37385870884421685), (37.911458333333336, 11.244892916799905)]}
# Accuracy:  71.42857142857143
