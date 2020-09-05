# Documentation for Frame Classification Code Description and Research Findings

## Dependencies used:
* Python 3.7
* Cuda 10.1
* Nvidia gtx 1050ti 4GB GDDR5
* I7 8750H
* Spyder 

## Running the Codes:
* Train_test_generator.py returns in a Balanced Dataset and two folders Named “Train_Data” ( Contains training image and labels) and “Test_Data” (contains test images and labels). Images stored as a numpy array of (no of images X height X width X 3)  and labels are one-hot encoded. One can set the numbers of test samples per class parameter according to need.
* Next run the Train.py, which generates Model, trains it, plots training history of accuracy and loss, evaluates model performance on test data, plots confusion matrix and outputs classification results.

#### Note:
Here the height and width is kept 112, which is ideally 224 for VGG16. The main reason is by setting it 224, I cannot set batch size more than 10 while training. Otherwise it will show resource exhausted error (My poor 1050ti). Therefore I have trained the model with 112x112x3 size images, with a batch size of 50. Complied with better hardware and Setting image size to 224x224x3, batch size around 100 will give better results for sure.

## Model Selection:
VGG16 is one of the best performing Model for document classification. This implementation is inspired by [this paper](https://arxiv.org/pdf/1801.09321.pdf) . Here the authors took VGG16 with imagenet pretrained weight and trained the base model Then using this base model weights they trained holistic crop, header crop, footer crop, left crop, right crop models, combined all the models output in a MLP classifier to give the final softmax classifier predictions. This implementation is a lot more computation costly, and the database they used was mostly scanned documents. Thankfully, we have a better database. So, I chose a simplified version of this approach, training the model by changing the top 3 layers of VGG16 with weights loaded from imagenet pretrained VGG16. The model summary is as below:

#### Model: "model"
Layer (type)             |    Output Shape     |         Param #   
--------------|-------------|-------------
input_1 (InputLayer)     |    [(None, 112, 112, 3)] |    0         
block1_conv1 (Conv2D)     |   (None, 112, 112, 64)   |   1792      
block1_conv2 (Conv2D)     |   (None, 112, 112, 64)    |  36928     
block1_pool (MaxPooling2D)  | (None, 56, 56, 64)    |    0         
block2_conv1 (Conv2D)    |    (None, 56, 56, 128)   |    73856     
block2_conv2 (Conv2D)    |    (None, 56, 56, 128)   |    147584    
block2_pool (MaxPooling2D) |  (None, 28, 28, 128)   |    0         
block3_conv1 (Conv2D)     |   (None, 28, 28, 256)   |    295168    
block3_conv2 (Conv2D)     |   (None, 28, 28, 256)   |    590080    
block3_conv3 (Conv2D)     |   (None, 28, 28, 256)   |    590080    
block3_pool (MaxPooling2D)  | (None, 14, 14, 256)   |    0         
block4_conv1 (Conv2D)      |  (None, 14, 14, 512)   |    1180160   
block4_conv2 (Conv2D)      |  (None, 14, 14, 512)   |    2359808 
block4_conv3 (Conv2D)       | (None, 14, 14, 512)    |   2359808 
block4_pool (MaxPooling2D) |  (None, 7, 7, 512)   |      0   
block5_conv1 (Conv2D)      |  (None, 7, 7, 512)   |      2359808  
block5_conv2 (Conv2D)      |  (None, 7, 7, 512)   |      2359808   
block5_conv3 (Conv2D)      |  (None, 7, 7, 512)  |       2359808   
block5_pool (MaxPooling2D)  | (None, 3, 3, 512)  |       0         
flatten (Flatten)           | (None, 4608)       |       0         
dense (Dense)                |(None, 500)       |        2304500   
dense_1 (Dense)            |  (None, 50)       |         25050     
dense_2 (Dense)            |  (None, 4)       |          204   
===========================|======================|=========
Total params: 17,044,442
Trainable params: 17,044,442
Non-trainable params: 0
_________________________________________________________________

## Other approaches:
1. In [this paper](https://arxiv.org/pdf/1711.05862.pdf) they consider alexnet and changed the top 3 layers to Extreme Learning Machine (ELM), accuracy results obtained are not as good as VGG16.
2. [This paper](https://arxiv.org/pdf/1907.06370.pdf) is very interesting, in which they combine visual features from CNN layer and textual feature to obtain document classification, unfortunately for misc class in our problem statement OCR is not helpful. The computation cost also remains high.
3. [This review paper](https://arxiv.org/pdf/2004.03705.pdf) contains a lot of approaches regarding document classification.
4. In some papers Inception Resnet V2 is also used, accuracy is lower though.

## Configuration and Hyperparameters:
* Train Image number: 3964 per class resulting in 15856 images
* Test images number: 240 images per class resulting in 960 images
* Batch Size: 50
* Callbacks: If accuracy reaches 95%, stop training.
* Loss = Categorical Crossentropy
* Optimizer = Adam
* Validation data = 10% of train data

## Results:
* After 17 epochs the model returns: 
* 95% in Train accuracy
* 80.33% in Validation accuracy
* 78.85% Test Accuracy

The plots below show the training history, classification results and confusion matrix for test data.

###### Plot of Training History
![Train](/images/Train_history_plot.png "Plot of Training History")

###### Classification Results on Test Data
![Train](/images/Classification Report.png "Plot of Classification Results on Test Data")

###### Confusion Matrix for Test Data
![plot](/images/Confusion_Matrix.png "Confusion Matrix for Test Data")

## Future Improvements:
* The difference in train and validation accuracy clearly denotes there is an overfitting during training. To improve one can use more data or data augmentation. According to the problem statement one can try random cropping to generate more data.
* Secondly, with better hardware one can implement 224x224x3 sized images instead of 112x112x3. Which can improve accuracy. One can try Image Data Generator available from tensorflow.keras to avoid out of memory error (OOM).
* Thirdly, decaying learning rate and hyperparameter tuning (like dropout and batch normalization) can help in accuracy improvement.

## Dataset:
Unfortunately for copyright issue i cannot share the database.


