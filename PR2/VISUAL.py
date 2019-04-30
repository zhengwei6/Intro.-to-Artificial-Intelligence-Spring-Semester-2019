import pandas as pd
import matplotlib.pyplot as plt
trainDataframe       = pd.read_csv('training.csv')
validationDataframe  = pd.read_csv('validation.csv')
train_mean      = trainDataframe.loc[:,'mean_score']/100
train_std       = trainDataframe.loc[:,'std_score']/100
validation_mean = validationDataframe.loc[:,'mean_score']/100
validation_std  = validationDataframe.loc[:,'std_score']/100
train_size      = trainDataframe.loc[:,'sample_num']
plt.figure(figsize=(15,10))
plt.plot(train_size, train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_size,train_mean+train_std,train_mean - train_std,alpha = 0.15)
plt.plot(train_size, validation_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_size, validation_mean + validation_std, validation_mean - validation_std,alpha = 0.15,color ='green')
plt.grid()
plt.title("Change tree number of breast")
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')
plt.ylim([0,1.0])
plt.show()