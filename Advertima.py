import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import utils
import time
import copy
import numpy as np
import pprint
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt
import argparse
from update import *
from PIL import Image
from CustomVGG16_BN import *

parser = argparse.ArgumentParser(description='Argumets for Revolut Coding Challenge.')
# Hyperparameters for training and augmentations
parser.add_argument('--trainTestSplit', type=float, default=0.7, help='Ratio to split the training and validation data. This stands of percentage amout of training data')
parser.add_argument('--weightDecay', type=float, default=5e-4, help='l2 regularization')
parser.add_argument('--num_epochs_classifier', type=int, default=35, help='Number of epochs the classifier will run for')
parser.add_argument('--batch_size', type=int, default=8, help='batch size of single run')
# Models parameters
parser.add_argument('--useModel', type=str, default='CustomVGG16_BN', help='type of model to be used', choices=['vgg16_bn', 'CustomNet0', 'CustomNet1', 'CustomNet2', 'CustomVGG16_BN'])
parser.add_argument('--modelsPath', type=str, default='model/CustomVGG16_BN/', help='path where model will be stored')
parser.add_argument('--classifierModelTrained', type=str, default='model/CustomVGG16_BN/classifier_best.pth', help='path for loading a trained model')

# Main Function Arguments
parser.add_argument('--trainFlag', type=bool, default=False, help='indicates to train a model after augmentation')
parser.add_argument('--evaluateFlag', type=bool, default=False, help='indicates to evaluate on a trained model')
parser.add_argument('--demoSingleImagePath', type=str, help='indicates to predict thea class of a single image')
parser.add_argument('--demoVideoPath', type=str, help='indicates to predict thea class of a single image')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # Looks for GPU


class CustomNet0(nn.Module):
	"""Shallowest Network

	makes a Pytorch nn Module with two conv and two FC layers
	
	"""
	def __init__(self, num_classes=10, init_weights=True):
		super(CustomNet0, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 5)
		# self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
		self.conv2 = nn.Conv2d(32, 64, 3)
		# self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
		self.fc1 = nn.Linear(64 * 5 * 5, 4096)
		self.do1 = torch.nn.Dropout(0.3)
		self.fc2 = nn.Linear(4096, num_classes)
		# self.do2 = torch.nn.Dropout(0.2)
		# self.fc3 = nn.Linear(84, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		# x = self.bn1(x)
		x = F.max_pool2d(F.relu(x),2,2)
		x = self.conv2(x)
		# x = self.bn2(x)
		x = F.max_pool2d(F.relu(x),2,2)
		x = x.view(-1, 64 * 5 * 5)
		x = self.fc1(x)
		# x = self.do1(x)
		x = F.relu(x)
		x = self.fc2(x)
		# x = self.do2(x)
		# x = F.relu(x)
		# x = self.fc3(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


class CustomNet1(nn.Module):
	"""Neural Network

	makes a Pytorch nn Module with two conv and three FC layers
	
	"""
	def __init__(self, num_classes=10, init_weights=True):
		super(CustomNet1, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 5)
		self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
		self.conv2 = nn.Conv2d(32, 64, 5)
		self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
		self.fc1 = nn.Linear(64 * 4 * 4, 120)
		self.do1 = torch.nn.Dropout(0.4)
		self.fc2 = nn.Linear(120, 84)
		self.do2 = torch.nn.Dropout(0.2)
		self.fc3 = nn.Linear(84, num_classes)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))),2,2)
		x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))),2,2)
		x = x.view(-1, 64 * 4 * 4)
		x = F.relu(self.do1(self.fc1(x)))
		x = F.relu(self.do2(self.fc2(x)))
		x = self.fc3(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


class CustomNet2(nn.Module):
	"""Neural Network

	makes a Pytorch nn Module with three conv and three FC layers
	
	"""
	def __init__(self, num_classes=10, init_weights=True):
		super(CustomNet2, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 5)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.fcGAP = nn.Linear(64, num_classes)
		self.doGAP = torch.nn.Dropout(0.3)
		self.fc1 = nn.Linear(64 * 10 * 10, 2048)
		self.do1 = torch.nn.Dropout(0.3)
		self.fc2 = nn.Linear(2048, num_classes)
		
	def forward(self, x):
		x = self.conv1(x)
		x = F.max_pool2d(F.relu(x),2,2)
		x = self.conv2(x)
		x = F.relu(x)
		x1 = F.avg_pool2d(x, kernel_size=10)
		x1 = self.doGAP(x1)
		x1 = x1.view(x1.size(0), -1)
		x1 = self.fcGAP(x1)
		x2 = x.view(-1, 64*10*10)
		x2 = self.fc1(x2)
		x2 = self.do1(x2)
		x2 = self.fc2(x2)
		x = x1*x2
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

def objectClassifierArgs(numClasses, train=True):
	"""Prepares the Pytorch models, optimizers, learning rates and epochs
		Currently supported models are ['vgg11', 'vgg11_bn', 'vgg16', 'vgg16_bn',
										'vgg19', 'vgg19_bn', 'resnet18', 'resnet50', 
										'densenet121', 'inceptionv3', 'CustomNet1']

	Args:
	  numClasses: and Integer declaring Number of classes of for custom dataset
	  train: A tring declaring whether to prepare for training or validation/testing

	Returns:
	  model_ft: model with changed last layer based on useModel parameter
	  optimizer_ft: optimizer method for training
	  exp_lr_scheduler: learning rate scheduler only if phase is train
	  num_epochs: number of epochs to train 
	"""
	if train:
		if args.useModel == 'vgg16_bn':
			model_ft = models.vgg16_bn(pretrained=True)
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = torch.nn.Linear(num_ftrs, numClasses)
		elif args.useModel == 'CustomNet0':
			model_ft = CustomNet0()
		elif args.useModel == 'CustomNet1':
			model_ft = CustomNet1()
		elif args.useModel == 'CustomNet2':
			model_ft = CustomNet2()
		elif args.useModel == 'CustomVGG16_BN':
			model_ft = Cvgg16_bn(pretrained=True)
		model_ft = model_ft.to(device)
		optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay= args.weightDecay)
		exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
		num_epochs = args.num_epochs_classifier
		return model_ft, optimizer_ft, exp_lr_scheduler, num_epochs
	else:
		if args.useModel == 'vgg16_bn':
			model_ft = models.vgg16_bn(pretrained=False)
			num_ftrs = model_ft.classifier[6].in_features
			model_ft.classifier[6] = torch.nn.Linear(num_ftrs, numClasses)
		elif args.useModel == 'CustomNet0':
			model_ft = CustomNet0()
		elif args.useModel == 'CustomNet1':
			model_ft = CustomNet1()
		elif args.useModel == 'CustomNet2':
			model_ft = CustomNet2()
		elif args.useModel == 'CustomVGG16_BN':
			model_ft = Cvgg16_bn(pretrained=False)
		return model_ft

def savePlots(losses, accuracy, precisions, recalls, f1scores, classWiseAccuracies, intToLabels):
	"""saves the different metric curves

	Args:
	  losses: A dictionary containing lists of epoch wise losses for training and validation
	  accuracy: A dictionary containing lists of epoch wise accuracy for training and validation
	  precisions: A dictionary containing lists of epoch wise precisions for training and validation
	  recalls: A dictionary containing lists of epoch wise recalls for training and validation
	  f1scores: A dictionary containing lists of epoch wise f1scores for training and validation
	  classWiseAccuracies: A dictionary containing a dictionary of lists of epoch wise accuracies of each class for training and validation 
	  intToLabels: A dictionary containing integer label to class name mappings

	Returns:
	  None: saves the plots
	"""

	# Plot loss
	plt.plot(losses['train'], 'r',label='training loss') 
	plt.plot(losses['val'], 'b',label='validation loss')
	legend = plt.legend(loc='upper center', shadow=True, fontsize='x-small') 
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.savefig(args.modelsPath+'CurveLoss.png')
	plt.clf()

	# Plot accuracy
	plt.plot(accuracy['train'], 'r',label='training accuracy') 
	plt.plot(accuracy['val'], 'b',label='validation accuracy')
	legend = plt.legend(loc='upper center', shadow=True, fontsize='x-small') 
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.savefig(args.modelsPath+'CurveAccuracy.png')
	plt.clf()

	# Plot precision
	plt.plot(precisions['train'], 'r',label='training precision') 
	plt.plot(precisions['val'], 'b',label='validation precision')
	legend = plt.legend(loc='upper center', shadow=True, fontsize='x-small') 
	plt.xlabel('epoch')
	plt.ylabel('precision')
	plt.savefig(args.modelsPath+'CurvePrecision.png')
	plt.clf()

	# Plot recall
	plt.plot(recalls['train'], 'r',label='training recall') 
	plt.plot(recalls['val'], 'b',label='validation recall')
	legend = plt.legend(loc='upper center', shadow=True, fontsize='x-small') 
	plt.xlabel('epoch')
	plt.ylabel('recall')
	plt.savefig(args.modelsPath+'CurveRecall.png')
	plt.clf()

	# Plot f1score
	plt.plot(f1scores['train'], 'r',label='training f1score') 
	plt.plot(f1scores['val'], 'b',label='validation f1score')
	legend = plt.legend(loc='upper center', shadow=True, fontsize='x-small') 
	plt.xlabel('epoch')
	plt.ylabel('f1score')
	plt.savefig(args.modelsPath+'Curvef1scores.png')
	plt.clf()

	# Plot class wise accuracy for training
	# colorMap = {0:'r', 1:'b', 2:'g', 3:'c', 4:'', 5:'m', 6:'y', 7:'k', 8:'#fff6ee', 9:'#ffeefe'}
	colorMap = {0:'C0', 1:'C1', 2:'C2', 3:'C3', 4:'C4', 5:'C5', 6:'C6', 7:'C7', 8:'C8', 9:'C9'}
	for key in intToLabels.keys():
		plt.plot(classWiseAccuracies['train'][intToLabels[key]], colorMap[key],label=intToLabels[key]) 
	legend = plt.legend(loc='upper center', shadow=True, fontsize='x-small') 
	plt.xlabel('epoch')
	plt.ylabel('class wise training accuracy')
	plt.savefig(args.modelsPath+'CurveClassWiseTrainAcc.png')
	plt.clf()

	# Plot class wise accuracy for validation
	for key in intToLabels.keys():
		plt.plot(classWiseAccuracies['val'][intToLabels[key]], colorMap[key],label=intToLabels[key]) 
	legend = plt.legend(loc='upper center', shadow=True, fontsize='x-small') 
	plt.xlabel('epoch')
	plt.ylabel('class wise validation accuracy')
	plt.savefig(args.modelsPath+'CurveClassWiseValAcc.png')
	plt.clf()

def TrainClassifier(modelsPath):
	"""Does the training and returns the most optimal model path

	Args:
	  modelsPath: Path where the model & Plots are saved

	Returns:
	  model: trained pytorch model path
	"""
	# Prepare the dataLoader with train valid split
	if args.useModel == 'CustomNet0' or args.useModel == 'CustomNet1'  or args.useModel == 'CustomNet2':
		data_transforms = transforms.Compose([
		# transforms.RandomHorizontalFlip(p=0.5),
		# transforms.RandomVerticalFlip(p=0.5),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
		# transforms.RandomErasing(p=0.5, scale=(0.02,0.4), ratio=(0.3, 3.3), value=0.4914, inplace=True),
	])
	else:
		data_transforms = transforms.Compose([
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomVerticalFlip(p=0.5),
			transforms.Resize((224,224)),
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
			transforms.RandomErasing(p=0.5, scale=(0.02,0.4), ratio=(0.3, 3.3), value=0.4914, inplace=True),
		])

	
	dataset = torchvision.datasets.FashionMNIST('./data', train=True, transform=data_transforms, target_transform=None, download=False)
	indices = torch.randperm(len(dataset)).tolist()
	splitTill = int(len(indices)*(1-args.trainTestSplit))
	dataset_train = torch.utils.data.Subset(dataset, indices[:-splitTill])
	dataset_val = torch.utils.data.Subset(dataset, indices[-splitTill:])
	
	image_datasets = {'train':dataset_train, 'val':dataset_val}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

	# import pdb;pdb.set_trace()
	
	# Extract data 
	intToLabels = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
	LabelsToInt = dict((val, key) for key, val in intToLabels.items())
	no_of_classes = len(intToLabels)

	os.makedirs(modelsPath, mode = 0o777, exist_ok = True) 

	#define the parameters for training
	model, optimizer, scheduler, num_epochs= objectClassifierArgs(numClasses=no_of_classes, train=True)
	criterion = torch.nn.CrossEntropyLoss()
	# criterion = torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
	
	since = time.time() #Start timing

	best_model_wts = copy.deepcopy(model.state_dict())

	# Initialize variables for best model and plots
	best_f1 = 0.0
	losses = {'train': [], 'val': []}
	precisions = {'train': [], 'val': []}
	recalls = {'train': [], 'val': []}
	f1scores = {'train': [], 'val': []}
	accuracy = {'train': [], 'val': []}
	classWiseAccuracies = {'train': dict((k,[]) for k in intToLabels.values()), 'val': dict((k,[]) for k in intToLabels.values())}

	# Loop through the epochs
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		for phase in ['train', 'val']:
			predictions = [] # list containing running predictions
			groundTruths = [] # list containing running groundtruths

			# prepare model for task
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			# Iterating through data
			for inputs, labels in dataloaders[phase]:
				if not (args.useModel == 'CustomNet0' or args.useModel == 'CustomNet1'  or args.useModel == 'CustomNet2'):
					inputs = torch.squeeze(torch.stack((inputs,inputs,inputs),2))
				inputs = inputs.to(device)
				labels = labels.to(device)
				
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					samples_per_cls = [] # Evaluate the class distribution for using in class balanced loss function
					for key in intToLabels.keys():
						samples_per_cls.append(labels.data.cpu().numpy().tolist().count(key))
					
					# Run through the CNN model and find the predictions
					outputs = model(inputs) 
					loss = criterion(outputs, labels)
					_, preds = torch.max(outputs, 1)
					

					# Compute back propagation and converge using an optimezer
					if phase == 'train':
						loss.backward()
						optimizer.step()
				# Compute accuracy metrics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

				# store running predictions and groundtruths
				predictions.extend(preds.data.cpu().numpy().tolist())
				groundTruths.extend(labels.data.cpu().numpy().tolist())

			# Update learning rate
			if phase == 'train':
				scheduler.step()

			# Compute accuracy metrics over an epoch 
			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			clfReport = classification_report(np.array(groundTruths), np.array(predictions), target_names = [val for val in intToLabels.values()])
			epochf1score = f1_score(np.array(groundTruths), np.array(predictions), average='weighted')
			epochPrecision = precision_score(np.array(groundTruths), np.array(predictions), average='weighted') 
			epochRecall = recall_score(np.array(groundTruths), np.array(predictions), average='weighted') 
			classWiseAcc = {}
			for key in intToLabels.keys():
				classWiseAcc[intToLabels[key]] = (confusion_matrix(np.array(groundTruths), np.array(predictions))).diagonal()[key] / Counter(groundTruths)[key]
				classWiseAccuracies[phase][intToLabels[key]].append(classWiseAcc[intToLabels[key]])

			# Update accuracy metrics to running metrics logger for plots
			losses[phase].append(epoch_loss)
			accuracy[phase].append(epoch_acc)
			precisions[phase].append(epochPrecision)
			recalls[phase].append(epochRecall)
			f1scores[phase].append(epochf1score)

			# Print accuracy and loss over an epoch
			if phase == 'train':
				print ('Phase: ',phase+'ing')
			else:
				print ('Phase: ',phase+'idation')
			# print(clfReport)
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			print ('class wise accuracy:', classWiseAcc)

			# Update best model if there is an improvement in f1 score and store accuracy metrics for that f1 score
			if phase == 'val' and epochf1score >= best_f1:
				best_f1 = epochf1score
				bestclfReport = clfReport
				bestclassWiseAcc = classWiseAcc
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		# Save model after every 10 epochs. TODO: Will be useful when one wants to resume training. Will also require the state of the optimizer.
		if epoch%10==0:
			torch.save(model.state_dict(), modelsPath+'classifier_'+str(epoch)+'.pth')

		# Save model of the best weights
		torch.save(best_model_wts, modelsPath+'classifier_best.pth')
		print()

	time_elapsed = time.time() - since

	# Print best metrics achieved post training 
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	print('Best Classwise Accuracy: ',  bestclassWiseAcc)
	print('Best Classification Report: ' )
	print(bestclfReport)

	# Save the plots
	savePlots(losses, accuracy, precisions, recalls, f1scores, classWiseAccuracies, intToLabels)
	return modelsPath+'classifier_best.pth'

def EvaluateClassifier(classifierModelTrained):
	"""Does the Evaluation and prints the metrics upon evaluation

	Args:
	  classifierModelTrained: Path to trained model

	Returns:
	  None: Prints the classification report
	"""

	# Extract data from csv
	if args.useModel == 'CustomNet0' or args.useModel == 'CustomNet1'  or args.useModel == 'CustomNet2':
		data_transforms = transforms.Compose([
		# transforms.RandomHorizontalFlip(p=0.5),
		# transforms.RandomVerticalFlip(p=0.5),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
		# transforms.RandomErasing(p=0.5, scale=(0.02,0.4), ratio=(0.3, 3.3), value=0.4914, inplace=True),
	])
	else:
		data_transforms = transforms.Compose([
			# transforms.RandomHorizontalFlip(p=0.5),
			# transforms.RandomVerticalFlip(p=0.5),
			transforms.Resize((224,224)),
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
			# transforms.RandomErasing(p=0.5, scale=(0.02,0.4), ratio=(0.3, 3.3), value=0.4914, inplace=True),
		])

	
	dataset = torchvision.datasets.FashionMNIST('./data', train=False, transform=data_transforms, target_transform=None, download=False)
	image_datasets = {'test':dataset}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['test']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
	
	# Extract data from csv
	intToLabels = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
	LabelsToInt = dict((val, key) for key, val in intToLabels.items())
	no_of_classes = len(intToLabels)

	# Prepare the model with trained weights
	model = objectClassifierArgs(numClasses=no_of_classes, train=False)
	model.eval()

	checkpoint = torch.load(classifierModelTrained)
	model.load_state_dict(checkpoint)
	model = model.to(device)
	
	since = time.time()

	for phase in ['test']:
		predictions = [] # Running predictions for evaluation
		groundTruths = [] # Running groundtruths for evaluation

		model.eval() # set model mode

		running_corrects = 0 # initialize running correctly predicted data

		for inputs, labels in dataloaders[phase]:
			if not (args.useModel == 'CustomNet0' or args.useModel == 'CustomNet1'  or args.useModel == 'CustomNet2'):
				inputs = torch.squeeze(torch.stack((inputs,inputs,inputs),2))
			inputs = inputs.to(device)
			labels = labels.to(device)
			
			# pass inputs through models and predict the output			
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			
			# update the running predictions and groundtruts and the number of correctly predicted labels
			running_corrects += torch.sum(preds == labels.data)
			predictions.extend(preds.data.cpu().numpy().tolist())
			groundTruths.extend(labels.data.cpu().numpy().tolist())

	# calculate metrics
	phase = 'test'
	epoch_acc = running_corrects.double() / dataset_sizes[phase]
	clfReport = classification_report(np.array(groundTruths), np.array(predictions), target_names = [val for val in intToLabels.values()]) 
	classWiseAcc = {}
	for key in intToLabels.keys():
		classWiseAcc[intToLabels[key]] = (confusion_matrix(np.array(groundTruths), np.array(predictions))).diagonal()[key] / Counter(groundTruths)[key]
		
	time_elapsed = time.time() - since
	# Print report
	print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print(clfReport)
	print('{} Acc: {:.4f}'.format(
		phase, epoch_acc))
	print ('class wise accuracy:', classWiseAcc)

def demoSingleImage(imagePath, classifierModelTrained):
	"""Predicts the class of a single image

	Args:
	  imagePath: Path to image

	Returns:
	  None: Prints the class name
	"""
	net = objectClassifierArgs(numClasses=10, train=False)
	net.load_state_dict(torch.load(classifierModelTrained))
	net = net.to(device)
	net.eval()
	classes = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

	img = cv2.imread(imagePath)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = Image.fromarray(img)
	img = PIL.ImageOps.invert(img)
	if not args.useModel == 'vgg16_bn':
		preprocess = transforms.Compose([transforms.Resize((224,224), interpolation=2),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
		img_tensor = preprocess(img)
		img_variable = Variable(img_tensor.unsqueeze(0)).to(device)
		img_variable = torch.squeeze(torch.stack((img_variable,img_variable,img_variable),2),0)
	else:
		if useModel == 'CustomNet2':
			preprocess = transforms.Compose([transforms.Resize((28,28), interpolation=2),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
			img_tensor = preprocess(img_pil)
			img_variable = Variable(img_tensor.unsqueeze(0)).to(device)
		elif  useModel == 'CustomVGG16_BN':
			weight_softmax = np.squeeze(params[-8].data.cpu().numpy())
			preprocess = transforms.Compose([transforms.Resize((224,224), interpolation=2),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
			img_tensor = preprocess(img_pil)
			img_variable = Variable(img_tensor.unsqueeze(0)).to(device)
			img_variable = torch.squeeze(torch.stack((img_variable,img_variable,img_variable),2),0)
		

	
	logit = net(img_variable)
	probs, preds = torch.max(logit, 1)
	print('output for the top1 prediction: %s' % classes[preds.item()])
	

def demoVideo(path, classifierModelTrained):
	"""Shows the output of video feed

	Args:
	  path: Path to video file. -1 if webcam
	  classifierModelTrained: path to trained model

	Returns:
	  None: Prints the class name
	"""
	net = objectClassifierArgs(numClasses=10, train=False)
	net.load_state_dict(torch.load(classifierModelTrained))
	net = net.to(device)
	params = list(net.parameters())
	weight_softmax = np.squeeze(params[-6].data.cpu().numpy())
	net.eval()
	classes = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
	if not args.useModel == 'vgg16_bn':
		def hook_feature(module, input, output):
			features_blobs.append(output.data.cpu().numpy())
		
	if path == -1:
		cam = cv2.VideoCapture(0)
	else:
		cam = cv2.VideoCapture(path)
	while(True):
		ret, img = cam.read()
		img = cv2.resize(img, (640,480))
		imgCopy =copy.deepcopy(img)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = Image.fromarray(img)
		img = PIL.ImageOps.invert(img)
		if args.useModel == 'vgg16_bn':
			preprocess = transforms.Compose([transforms.Resize((224,224), interpolation=2),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
			img_tensor = preprocess(img)
			img_variable = Variable(img_tensor.unsqueeze(0)).to(device)
			img_variable = torch.squeeze(torch.stack((img_variable,img_variable,img_variable),2),0)
			logit = net(img_variable)
			probs, preds = torch.max(logit, 1)
			print('output for the top1 prediction: %s' % classes[preds.item()])
			cv2.imshow('frame',imgCopy)
		else:
			features_blobs = []
			if args.useModel == 'CustomNet2':
				net._modules.get('conv2').register_forward_hook(hook_feature)
			elif args.useModel == 'CustomVGG16_BN':
				net._modules.get('avgpool').register_forward_hook(hook_feature)
			result = get_cam(net, features_blobs, img, classes, imgCopy, args.useModel, device)
			cv2.imshow('frame',np.concatenate((imgCopy, result),1))
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	if args.trainFlag:
		args.classifierModelTrained = TrainClassifier(modelsPath=args.modelsPath)
	if args.evaluateFlag:
		EvaluateClassifier(classifierModelTrained=args.classifierModelTrained)
	if args.demoSingleImagePath:
		demoSingleImage(imagePath=args.demoSingleImagePath, classifierModelTrained=args.classifierModelTrained)
	if args.demoVideoPath:
		demoVideo(args.demoVideoPath, args.classifierModelTrained)
	if (not args.trainFlag) and (not args.evaluateFlag) and (not args.demoSingleImagePath) and (not args.demoVideoPath):
		print ('Please select a flag either to train or evaluate or process a single image/ Video/ Webcam')
	