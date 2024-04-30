# Garbage Gang

**Purpose of Project:**

In a world where the climate clock is steadily ticking down to our last minutes until climate change is irreversible, and where environmental sustainability is becoming a mainstream issue, it makes sense to recycle. Research has shown that 94% of Americans support recycling, but according to a Forbes article, only 35% actually recycle. (
https://www.forbes.com/sites/blakemorgan/2021/04/21/why-is-it-so-hard-to-recycle/) 
The large difference between these two statistics comes down to ignorance and inconvenience. People don’t know how to recycle, what to recycle, or where to recycle. Because people don’t understand what can be recycled, general waste is often put in the recyclable pile. And because the actual recyclables are contaminated by improperly sorted items, the entire load is ruined. Items are often burned or dumped into landfills. This is a common occurrence. 
Knowledge and convenience can solve this. 

**Summary of Project:** 

This is why we have created the Garbage Gang ANN. With a trash sorter that can identify whether one’s object is trash, paper, glass, cardboard, metal, or plastic, it can reduce the amount of contamination in the recyclables bin.

Faced with the issue of access to only 2527 data points between six classes we resolved to explore how data augmentation and transfer learning might bolster the accuracy of our model. Our secondary goal in this project has been to work out which combination of data augmentation and/or transfer learning would allow us to squeeze the most information out of our data and increase our base model’s accuracy from 69%.

**Data:**

We got our data from a Github called “Trashnet” (https://github.com/garythung/trashnet/tree/master). These images were personally photographed by Stanford University students themselves. The data consists of 6 types of trash and recyclables, which we made into classes. These classes are: paper, plastic, trash, cardboard, metal, and glass. There are 594 photos labeled paper, 482 plastic, 137 trash, 403 cardboard, 410 metal, and 501 glass.


<img width="570" alt="Screenshot 2024-04-29 at 5 34 29 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/049c8f93-c6e3-417a-b299-54abac1512cb">


As shown in the class dsitribution, there are a lot less photos of trash compared to the other classes. We attempted to fix this problem by reaching out to companies to get more photos, but unfortunately, none of them replied. To solve this problem, we decided to do data augmentation on the 137 data points of trash we already have. 


Non Augmented Dataset:


<img width="687" alt="Screenshot 2024-04-29 at 5 36 26 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/efc52267-a392-4d48-880f-ddd68d044ac0">


Augmented Dataset: 

<img width="687" alt="Screenshot 2024-04-29 at 7 32 41 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/8e8a5b3d-9e0f-42c7-aae6-06a3ea7f2f0a">

<img width="570" alt="Screenshot 2024-04-29 at 5 34 29 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/f0db4975-f9a3-468a-8a78-3ad494cfecf3">



**Methods:**

***Base Model*** - 
We based our model’s architecture off of the AlexNet architecture, a popular CNN architecture that won the ImageNet Large Scale Visual Recognition Challenge in 2021. We downscaled the model to fit our simplified needs, removing convolutional layers, removing fully connected layers, and increasing the stride within convolutional layers, which simultaneously reduces the training time and computing power necessary to run the model. 

Our Model has 3 convolutional layers. The first layer has 96 filters of size 11x11 with a stride of 5x5. We use ReLU activation and batch normalization to normalize the activations. The first conv layer is followed by a max-pooling layer with a pool size of 3x3 and a stride of 2x2 and a
dropout layer with a dropout rate of 0.25 to prevent overfitting. The second convolutional layer has 256 filters of size 5x5 with a stride of 1x1. We again use ReLU activation and batch normalization to normalize the activations. The second conv layer is followed by the same max-pooling layer and dropout layer as the first conv layer. The third and last convolutional layer has 256 filters of size 3x3 with a stride of 2x2, again using ReLU activation as well as max pooling and dropout. Then we use a flatten layer to squish the third convolutional layers output into a 1D array. This is followed by a fully connected layer with 2048 neurons, ReLU activation and dropout rate of 0.5. Lastly, the output layer is a dense layer with 6 neurons and the softmax activation function for multiclass classification. This base model has 11740486 trainable parameters, most of which come from the dense layer.  

***Transfer Learning  model*** - 
Transfer learning is a machine learning technique where a pre-trained model developed for a specific task is reused as the starting point for a new model on a related task. We used the DenseNet121 architecture, a convolutional neural network, for this.
We start by loading the DenseNet121 model pre-trained on the ImageNet dataset. We exclude the fully connected layers (include_top=False) since we're using the model as a feature extractor. To retain the pre-trained weights and prevent them from being updated during training, we freeze all layers in the base model. Next, we add custom layers on top of the base model to adapt it to our specific task. We use Global Average Pooling to reduce computational complexity and combat overfitting. Then, we add a fully connected layer with 512 units and ReLU activation, followed by Batch Normalization to stabilize training, and Dropout to regularize the network and prevent overfitting. The final layer consists of a Dense layer with softmax activation, which outputs probabilities for each of the six classes. 

In both cases, the model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function. During training, we use an ImageDataGenerator to preprocess the input images and in some cases perform data augmentation. We split the data into training and validation sets and specify the target image size and batch size. We train the model for 15 epochs, monitoring both training and validation accuracy to evaluate its performance.  

After training, we evaluate the model on the validation dataset to assess its loss and accuracy. Finally, we visualize the training history by plotting the training and validation loss over epochs using Matplotlib. This allows us to analyze the model's learning progress and identify any signs of overfitting or underfitting. 

***Base Model with Data Augmentation*** -

Data augmentation is the process by which a dataset is expanded by procedurally altering its files. For our images, we applied horizontal flips, vertical flips, brightness shifts and a slight zoom (between 0 and 20%). We avoided transformations such as shears and rotations because they create empty pixels that need to be filled, which can result in less realistic images. If the augmentations distort the original images too much, the network will be unable to classify images accurately, so we opted for more subtle transformations.

***Transfer Learning and Data Augmentation***

**Results:**

***Base Model*** -
<img width="1140" alt="Screenshot 2024-04-29 at 6 28 35 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/ba031ebe-9e48-4c9c-9733-dd0ebc242fcf">
The loss decreases and accuracy increases fairly consistantly throughout the 15 epochs
The fluctuation in validation accuracy, especially in the second half of training indicates an issue with overfitting where the model is having trouble generalizing to unseen data
<img width="544" alt="Screenshot 2024-04-29 at 6 27 44 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/fb46da53-1261-4cf8-9e12-4103dae14638">

 Post training evaluation -
 Test loss: 1.8880281448364258
 Test accuracy: 0.4881422817707062

 The confusion matrix indicates that the base model has trouble categorizing paper, often mistaking it for metal as well as trouble categorizing trash as metal. 
<img width="779" alt="Screenshot 2024-04-29 at 6 37 32 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/dd1c253b-36fa-4804-bd4a-95601ba1c711">

***Transfer Learning*** - 

<img width="1229" alt="Screenshot 2024-04-29 at 10 48 42 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/110687817/5c55169d-5f2b-4a66-a8c9-10c661d29210">

By the 15th epoch, we have a loss: 0.3457, accuracy: 0.8819, val_loss: 0.9096, and val_accuracy: 0.6978. The discrepancy between the accuracy (88%) and the validation accuracy (69%) shows that the model is overfitting. To fix overfitting, a couple strategies we could use is adding dropout layers, using data augmentation, using fewer layers or parameters, and/or increasing regularization. However, the transfer learning does take a while to run so there wasn't time to implement these strategies.


The Training Loss and Validation Loss is graphed here:

<img width="686" alt="Screenshot 2024-04-29 at 10 48 57 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/110687817/3df08747-8db6-47f9-b16e-6315fa6bac32">

We can see that it doesn't really converge and there's lots of fluctuations. This also shows overfitting.

The Confusion Matrix is below:

<img width="724" alt="Screenshot 2024-04-29 at 10 53 22 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/110687817/4b2b89f9-e851-4bdd-a1c0-fd31e92244fc">

We can see that it often confuses paper with other things, especially plastic.


***Base Model with Data Augmentation*** -

Test loss: 1.2529,
Test accuracy: 0.4970

Minimal improvement from base model, but smoother training - the validation loss goes down and stays down pointing to an improvement with the overfitting issue.

<img width="508" alt="Screenshot 2024-04-30 at 8 19 49 AM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/8935d8b6-1fdb-47d1-8254-88356bf60534">
<img width="561" alt="Screenshot 2024-04-30 at 8 20 10 AM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/795cb008-e697-49c6-83ca-8a9f6cbe4869">

<img width="1102" alt="Screenshot 2024-04-30 at 8 12 56 AM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/c056469b-9a33-42db-8a16-26acd0ab1540">

***Transfer Learning and Data Augmentation*** -
<img width="1166" alt="Screenshot 2024-04-30 at 8 23 49 AM" src="https://github.com/sgoffe/Garbage-Gang/assets/110687817/51b9cbff-bc53-4dbe-bd81-dda6b8a233c7">
The last epoch had a loss: 0.3901, accuracy: 0.8678, val_loss: 0.7899, and val_accuracy: 0.7192.
Here's the Training History.
<img width="608" alt="Screenshot 2024-04-30 at 8 23 58 AM" src="https://github.com/sgoffe/Garbage-Gang/assets/110687817/528e1c8e-8793-4889-86ea-4b4ab4abcbd2">

Here's the confusion matrix:

<img width="729" alt="Screenshot 2024-04-30 at 8 53 16 AM" src="https://github.com/sgoffe/Garbage-Gang/assets/110687817/e5cabf75-d7f5-43b1-9a22-095499f11c7d">



**Conclusion**
Out of all these models, we see that the transfer learning and data augmentation is the best result we got with a validation accuracy of 71%. 

**Issues We Ran Into**
- We only had 137 photos of trash, while other categories of recyclables are around 500 photos. To fix this discrepancy, we used data augmentation.
- No diversity in the background of the images and minimal data in general
**Moving forward**
- More data would be our first step
- Implementing other overfitting prevention tactics - early stopping
- Unfreeing more layers in transfer learning
- More specific categories 






