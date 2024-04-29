# Garbage Gang

Purpose of Project:
In a world where the climate clock is steadily ticking down to our last minutes until climate change is irreversible, and where environmental sustainability is becoming a mainstream issue, it makes sense to recycle. Research has shown that 94% of Americans support recycling, but according to a Forbes article, only 35% actually recycle. (
https://www.forbes.com/sites/blakemorgan/2021/04/21/why-is-it-so-hard-to-recycle/) 
The large difference between these two statistics comes down to ignorance and inconvenience. People don’t know how to recycle, what to recycle, or where to recycle. Because people don’t understand what can be recycled, general waste is often put in the recyclable pile. And because the actual recyclables are contaminated by improperly sorted items, the entire load is ruined. Items are often burned or dumped into landfills. This is a common occurrence. 
Knowledge and convenience can solve this. 

Summary of Project: 
This is why we have created the Garbage Gang ANN. With a trash sorter that can identify whether one’s object is trash, paper, glass, cardboard, metal, or plastic, it can reduce the amount of contamination in the recyclables bin.

Faced with the issue of access to only 2527 data points between six classes we resolved to explore how data augmentation and transfer learning might bolster the accuracy of our model. Our secondary goal in this project has been to work out which combination of data augmentation and/or transfer learning would allow us to squeeze the most information out of our data and increase our base model’s accuracy from 69%.

Data:
We got our data from a Github called “Trashnet” (https://github.com/garythung/trashnet/tree/master). These images were personally photographed by Stanford University students themselves. The data consists of 6 types of trash and recyclables, which we made into classes. These classes are: paper, plastic, trash, cardboard, metal, and glass. There are 594 photos labeled paper, 482 plastic, 137 trash, 403 cardboard, 410 metal, and 501 glass.
There are a lot less photos of trash compared to the other classes. We attempted to fix this problem by reaching out to companies to get more photos, but unfortunately, none of them replied. To solve this problem, we decided to do data augmentation on the 137 data points of trash we already have. By amalgamating the original photos with the new augmented ones, we are able to now have X photos of trash

Method:
