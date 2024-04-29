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


As shown in the class dsitribution, there are a lot less photos of trash compared to the other classes. We attempted to fix this problem by reaching out to companies to get more photos, but unfortunately, none of them replied. To solve this problem, we decided to do data augmentation on the 137 data points of trash we already have. By amalgamating the original photos with the new augmented ones, we are able to now have X photos of trash


Non Augmented Dataset:


<img width="687" alt="Screenshot 2024-04-29 at 5 36 26 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/98913773/efc52267-a392-4d48-880f-ddd68d044ac0">


Augmented Dataset: 

[insert preview of augmented image dataset]

**Methods:**

***Base Model*** - 
We based our model’s architecture off of the AlexNet architecture, a popular CNN architecture that won the ImageNet Large Scale Visual Recognition Challenge in 2021. We downscaled the model to fit our simplified needs, removing convolutional layers, removing fully connected layers, and increasing the stride within convolutional layers, which simultaneously reduces the training time and computing power necessary to run the model. 

Our Model has 3 convolutional layers. The first layer has 96 filters of size 11x11 with a stride of 5x5. We use ReLU activation and batch normalization to normalize the activations. The first conv layer is followed by a max-pooling layer with a pool size of 3x3 and a stride of 2x2 and a
dropout layer with a dropout rate of 0.25 to prevent overfitting. The second convolutional layer has 256 filters of size 5x5 with a stride of 1x1. We again use ReLU activation and batch normalization to normalize the activations. The second conv layer is followed by the same max-pooling layer and dropout layer as the first conv layer. The third and last convolutional layer has 256 filters of size 3x3 with a stride of 2x2, again using ReLU activation as well as max pooling and dropout. Then we use a flatten layer to squish the third convolutional layers output into a 1D array. This is followed by a fully connected layer with 2048 neurons, ReLU activation and dropout rate of 0.5. Lastly, the output layer is a dense layer with 6 neurons and the softmax activation function for multiclass classification. This base model has 11740486 trainable parameters, most of which come from the dense layer.  

***Transfer Learning  model*** - 
Transfer learning is a machine learning technique where a pre-trained model developed for a specific task is reused as the starting point for a new model on a related task. We used the DenseNet121 architecture, a convolutional neural network, for this.
We start by loading the DenseNet121 model pre-trained on the ImageNet dataset. We exclude the fully connected layers (`include_top=False`) since we're using the model as a feature extractor. To retain the pre-trained weights and prevent them from being updated during training, we freeze all layers in the base model. Next, we add custom layers on top of the base model to adapt it to our specific task. We use Global Average Pooling to reduce computational complexity and combat overfitting. Then, we add a fully connected layer with 512 units and ReLU activation, followed by Batch Normalization to stabilize training, and Dropout to regularize the network and prevent overfitting. The final layer consists of a Dense layer with softmax activation, which outputs probabilities for each of the six classes. 

In both cases, the model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function. During training, we use an ImageDataGenerator to preprocess the input images and in some cases perform data augmentation. We split the data into training and validation sets and specify the target image size and batch size. We train the model for 15 epochs, monitoring both training and validation accuracy to evaluate its performance. However, because the transfer learning takes a while and our computer shuts off often, the highest we made it for transfer learning was 13 out of 15 epochs with a of loss: 0.3433, accuracy: 0.8883, val_loss: 0.8454, and val_accuracy: 0.7237. 

After training, we evaluate the model on the validation dataset to assess its loss and accuracy. Finally, we visualize the training history by plotting the training and validation loss over epochs using Matplotlib. This allows us to analyze the model's learning progress and identify any signs of overfitting or underfitting. However, because the model stops and the computer times out at 13 out of 15 epochs, we have had to manually put the training loss and validation loss in a list and then graph it from there. And because the computer times out, we can’t run the confusion matrix on the transfer learning model.

***Base Model with Data Augmentation*** -

***Transfer Learning and Data Augmentation***

**Results:**

***Base Model*** -
 

***Transfer Learning*** - 
<img width="1225" alt="Screenshot 2024-04-26 at 9 32 13 AM" src="https://github.com/sgoffe/Garbage-Gang/assets/110687817/1094faf1-38b3-44dc-9da4-03c4af0a441d">
With Transfer Learning, we only managed to reach 13 out of 15 epochs. The last recorded data had a loss: 0.3433, accuracy: 0.8883, val_loss: 0.8454, and val_accuracy: 0.7237.
The Training Loss and Validation Loss is graphed below.
<img width="784" alt="Screenshot 2024-04-29 at 5 03 12 PM" src="https://github.com/sgoffe/Garbage-Gang/assets/110687817/47cf9aee-e822-4dd4-98af-86a4c909f6ff">



***Base Model with Data Augmentation*** -

***Transfer Learning and Data Augmentation*** -


**Issues We Ran Into**
- We only had 137 photos of trash, while other categories of recyclables are around 500 photos. To fix this discrepancy, we used data augmentation.
- Transfer Learning took a while to run and the computer would often shut down. The furthest we got was 13 out of 15 epochs as mentioned above. This issue was unable to be fixed since it was a hardware issue with our computer; sometimes we wouldn't have enough ram and sometimes our computer would just shut down. We did not include transfer learning in our final presentation as we ran into these issues and the accuracy was not as good as the other models either. -  


**Conclusion**





