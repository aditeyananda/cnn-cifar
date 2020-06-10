# cnn-cifar
1. IMPORTING THE LIBRARIES
The torchvision package consists of popular datatsets, model architectures, and common image transformations for computer vision.

2. LOADING AND NORMALISING CIFAR
torch.utils.data.DataLoader is used to load the data set for training and testing. And torchvision.transforms is used to prepare the data for use with the CNN. It converts Python Image Library (PIL) format to PyTorch tensors and normalize the data by specifying a mean and standard deviation for each of the three channels.In order to prevent over-fitting of training, the size of the data set during training is usually increased by randomly flipping the image and randomly adjusting the brightness of the image on a small data set.

3. DEFINING THE NEURAL NETWORK
This Neural Network has 3 Convolutional layers and 2 Fully Connected layers. In the first Conv layer, there are three input channels, three filters to extract, a filter size of 3, and a stride of 1. And I padded one by one to prevent the size from shrinking. The input value is reduced by half each time it passes through each convolution because the MaxPooling size is 2,2.

4. DEFINING LOSS FUNCTION AND OPTIMISER
As images are being classified into more than two classes cross-entropy is used as loss function. To optimize the network stochastic gradient descent (SGD) is employed with momentum 0.9 and learning rate 0.001.

5. TRAINING THE NEURAL NETWORK
The network is trained using the training_loader data, by going over all the training data in batches of 4 images, and repeating the whole process 20 times, i.e., 20 epochs. Every 2000 batches training progress is reported by printing the current epoch and batch number along with the running loss value.


6. TESTING THE NETWORK
a) Predicting Category for Four Test Images
Four random images and their corresponding labels are loaded from the testing data set. The model predicts what the images are after it has been trained and the prediction is tested against the ground truth.

b) PREDICTING ACCURACY ON 1000 IMAGES

c) PREDICTING THE CATEGORY FOR INDIVIDUAL CLASSES
