# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader
import logging
import random
import pandas as pd
import os
import numpy as np
from sklearn import datasets

# Constant
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler('my.log', 'w', 'utf-8'),
        logging.StreamHandler()
    ])
logger   = logging.getLogger()

# Load a CSV file
def load_csv(filename):
	if 'txt' in filename:
		dataset = []
		file = open(filename,'r')
		for line in file.readlines():
			line = line.split()
			dataset.append(line)
	else:
		file = open(filename, "r")
		lines = reader(file,delimiter=' ')
		dataset = list(lines)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	logger.info('Divid dataset to '+ str(len(folds))    + ' folds')
	logger.info('Training samples have '  + str(len(folds[0])*(len(folds)-1)) + ' rows')
	validation_scores = list()
	training_scores = list()
	for fold in folds:
		# validation score
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		validation_scores.append(accuracy)
		# training score
		test_set = list()
		for row in train_set:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in train_set]
		training_accuracy = accuracy_metric(actual, predicted)
		training_scores.append(training_accuracy)
	return str(len(folds[0])*(len(folds)-1)),training_scores,validation_scores
 
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini
 
# Select the best split point for a dataset
def get_split(dataset, sample_feature):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		if not index in sample_feature:
			continue
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth, sample_feature):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, sample_feature)
		split(node['left'], max_depth, min_size, depth+1, sample_feature)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, sample_feature)
		split(node['right'], max_depth, min_size, depth+1, sample_feature)
 
# Build a decision tree
def build_tree(train, max_depth, min_size, sample_feature):
	root = get_split(train, sample_feature)
	root['sample_feature'] = sample_feature
	split(root, max_depth, min_size, 1, sample_feature)
	return root
 
# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 

# Random Forest of CART Algorithm
'''
@param train, test, max_depth, min_size, tree_num, bagging_ratio, bagging_feature_num
@return forest model
'''
def random_forest(train, test, max_depth, min_size, tree_num, bagging_ratio, bagging_feature_num):
    #train random_forest
	tree_list = []
	for i in range(tree_num):
		sub_sample_num = int(len(train)*bagging_ratio)
		bagging_list   = random.sample(train, k=sub_sample_num)
		sample_feature = random.sample(range(len(train[0]) - 1), k=bagging_feature_num)
		tree           = build_tree(bagging_list, max_depth, min_size, sample_feature)
		tree_list.append(tree)
	#boostrap aggregating
	predictions = list()
	for row in test:
		sub_predictions = list()
		for tree in tree_list:
			tree_predict   = predict(tree, row)
			sub_predictions.append(tree_predict)
		ans = max(set(sub_predictions), key=sub_predictions.count)
		predictions.append(ans)
	return (predictions)


def main():
	filename = 'cross200.txt'
	dataset = load_csv(filename)
	for i in range(len(dataset[0])):
			str_column_to_float(dataset, i)
	n_folds = 10
	max_depth = 5
	min_size = 10
	tree_num = 10
	bagging_ratio = 0.7
	bagging_feature_num = 2
	randomForestDataFrameTraining   = pd.DataFrame(columns=['max_depth','mean_score','std_score'])
	randomForestDataFrameValidation = pd.DataFrame(columns=['max_depth','mean_score','std_score'])
	wineDataList=[]
	wine = datasets.load_wine()
	for index in range(len(wine['target'])):
		wineDataList.append(np.append(wine['data'][index],wine['target'][index]).tolist())
	dataset = wineDataList
	for max_depth in range(1,11):
		sampleNum, training_scores,validation_scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size,tree_num,bagging_ratio,bagging_feature_num)
		randomForestDataFrameTraining   = randomForestDataFrameTraining.append(pd.Series([max_depth, np.mean(training_scores) , np.std(training_scores)], index=['sample_num','mean_score','std_score']), ignore_index=True)
		randomForestDataFrameValidation = randomForestDataFrameValidation.append(pd.Series([max_depth, np.mean(validation_scores) , np.std(validation_scores)], index=['sample_num','mean_score','std_score']), ignore_index=True)
		print('traing_scores')
		print(np.mean(training_scores))
		print('validation_scores')
		print(np.mean(validation_scores))
	
	trainingSavePath   = os.path.join('.','training' + '.csv')
	validationSavePath = os.path.join('.','validation' + '.csv')
	
	randomForestDataFrameTraining.to_csv(trainingSavePath,sep=',')
	randomForestDataFrameValidation.to_csv(validationSavePath,sep=',')
if __name__ == '__main__':
	main()
