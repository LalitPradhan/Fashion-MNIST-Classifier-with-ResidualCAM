# VGG classifier with Residual connection from CAM network outputs as Auxiliary task
Click on [Demo Youtube Link](https://www.youtube.com/watch?v=0Xbr2bHcrUo)
This library does object classification on Fashion MNIST dataset and creates Class Activation Maps to locate the discriminating pixels of an apparel.

## Requirements
- Python (>=3.5.2)
- Numpy (>=1.17.0)
- Pytorch (>=1.3.1+cu92)
- Torchvision (>=0.4.2+cu92)
- Pillow (>=5.3.0)
- scikit-learn (>=0.20.0)
- Matplotlib (>=3.0.3)

## Training Circumstances
- All models were trained on a Nvidia 1080 Ti 12 GB GPU with 16 GB CPU memory on Ubuntu 16.04 annd CUDA 9.2
- Used default Pytorch Dataloader for loading [Fashion-MNIST dataset](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.FashionMNIST)

## Description of Architecture
The proposed Network is inspired to visualize the discriminating feature on the pixels as proposed in the paper [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150). It uses Global Average Pooling (GAP) layer to generate Class Activation Maps (CAM) which indicate the discriminative regions in an image during classification. The base architecture use is VGG16 with Batch Normalization. The vgg16_bn network if used directly for classification reaches an overall accuracy of 94.63% on the 10000 test images. Implementing the GAP paper by itself doesn't reach a good accuracy for classification as the FC layers are replaced by GAP in the original vgg architecture which reduce the number of hidden units in FC layer. To this end a hybrid architecture is proposed that uses the qualities from both the paper. I chose to use a CAM network as the challenge required to make a video tool to classify the object as well with assistance from a object detector library. From CAMs, one could generate rough bounding boxes as done in the original paper [implementation](https://github.com/metalbubble/CAM/blob/master/generate_bbox.m) Further one can segment the pixel level data using IRNet as described in the [paper](https://arxiv.org/pdf/1904.05044.pdf). (I wasn't able to make the bounding box from IRNet owing to time constraints, but since I have implemented it in the past I can be sure it works). Here is how the proposed architecture looks like:
![alt text](https://github.com/LalitPradhan/Fashion-MNIST-Classifier-with-ResidualCAM/blob/master/misc/Architecture.png)

Proposed architecture learns the classifier and CAM simultaneously with a residual skip connection between the VGG final output layer and CAM's final output layer as shown above. Instead of adding the residual connection as propsed in Resnet, an element wise matrix multiplication is performed. This would ensure during backpropagation that the CAM network during backpropagation is heavily dependent on the classifier vgg network (differentiation of product of two functions rule) and forces the CAM network to learn as much as the classifier network. (Note: I haven't seen till date any paper using elementwise product instead of addition to my knowledge). The propsed architecture is inspired from VGG, CAM and Resnet with matrix multiplication instead of addition in residual connections.

The dot product between the GAP output and features extracted isn't a part of the training but is used for the generation of CAMs. 

## Other architectures evaluated during the challenge. All hyperparameters were tested against shallower network as it was faster to train.
 - Three shallower networks were trained. It was observed that with increase in depth of network the accuracy decreases with input image size 28x28. Also increasing the number of hidden units in FC increases the accuracy. The best accuracy for just the classifier without residual connection with GAP layer was 91.37%. The accuracy increases to 91.50% when combined with GAP layer because of the auxiliary learning task in the shallower network.
 	- a) With 2 Conv and 2 FC 
 	- b) With 2 Conv and 3 FC
 	- c) With 3 Conv and 3 FC
 - Random erasing and horizontal/ vertical flips were introduced into the training dataset. Contrary to claims in the paper and my previous experience with larger input images, the accuracy dropped on introducing the augmentations which could be explained due to small input size. Randomly erasing pixels in smaller input images reduced a greater percentage of pixels as compared to larger images which probably makes the input data very noisy. The reason behind selection of Random Erasing Augmentation in the VGG16 network is due to the proven accuracy in the [SOTA](https://arxiv.org/pdf/1708.04896v2.pdf) implementation of Fashion-MNIST which explains how it generalizes better with occlusions leading to better robust models during training which prevent over fitting.
 - Introducing batch norm improves the training speed. Batchnorm and dropout reduces the accuracy in shallower networks, the reasons for which as of now is unclear. In deeper networks like the VGG it improves the accuracy and generalizes better in accordance to the respective papers for [Batchnorm](https://arxiv.org/pdf/1502.03167.pdf) and [Dropout](https://dl.acm.org/doi/pdf/10.5555/2627435.2670313?download=true).
 - Cross entropy loss performs better than MSE loss. A variant of Class Balanced loss based on Focal loss and Multimargin loss performed poorly.
 - Introducing a standardization (subtracting mean and dividing by std) improved the accuracy from 90.01% to 90.97% for a network with 3 Conv and 3 FC layers. 
 - A low but greater than zero weight decay (l2 regularization) increases the accuracy marginally.
 - Further It was noticed that passing an Inverted image gives better result as it closely resembles the training datset in that scenario.
 - All Loss, Accuracy, Classwise accuracy, precision, recall and F1 scores are logged as plot in the model/<model name> folders.
 - The best classifier is chosen based on the best f1 score calculated weighted over classes.


The training summary of the architecture proposed above is as follows:
![alt text](https://github.com/LalitPradhan/Fashion-MNIST-Classifier-with-ResidualCAM/blob/master/misc/trainingSummary.png)

The different metrics for the proposed architecture are as follows:
![alt text](https://github.com/LalitPradhan/Fashion-MNIST-Classifier-with-ResidualCAM/blob/master/misc/Metrics.gif)

Detailed summaries of all networks can be found [here](https://github.com/LalitPradhan/Fashion-MNIST-Classifier-with-ResidualCAM/blob/master/misc/summary.txt)

## Code Description
Download trained [weights](https://1drv.ms/u/s!Au_917wA6i4mijgpDePPZOGW42pe?e=SQVrGF) to model/CustomVGG16_BN/ folder.
- **To Train**: `python3 Advertima.py --modelsPath model/CustomVGG16_BN/ --classifierModelTrained model/CustomVGG16_BN/classifier_best.pth --useModel CustomVGG16_BN --trainFlag True`
- **To Evaluate**: `python3 Advertima.py --classifierModelTrained model/CustomVGG16_BN/classifier_best.pth --useModel CustomVGG16_BN --evaluateFlag True`
- **Webcam demo**: `python3 Advertima.py --classifierModelTrained model/CustomVGG16_BN/classifier_best.pth --useModel CustomVGG16_BN --demoVideoPath -1`
- **VideoFile demo**: `python3 Advertima.py --classifierModelTrained model/CustomVGG16_BN/classifier_best.pth --useModel CustomVGG16_BN --demoVideoPath <Video File Path>`
- **Single Image demo**: `python3 Advertima.py --classifierModelTrained model/CustomVGG16_BN/classifier_best.pth --useModel CustomVGG16_BN --demoSingleImagePath <Image File Path>`

