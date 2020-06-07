---
layout: post
title: Multiple scene image classification
---

In this post I want to talk about multiple scene image classification and the problems tha I found along the way and the different ideas to overcome those problems.

In this problem setting each image $$I_{i}$$ has tags $$t_{i1}, t_{i2},..., t_{ik}$$ over $$T = \{t_{1}, t_{2}, ..., t_{N}\}$$ the set of all tags. 

Specifically speaking there are hotel images where you can see the guestroom, the bathroom, the living room. Another examples in the same image you can see the hotel front, the pool, and the beach.

## Examples

### bedroom, kitchen and livingroom
![noise](/images/multitag/guestroom-kitchen-living1.jpg){:class="img-responsive" width="500px"}

### bedroom and kitchen
![noise](/images/multitag/guestroom-kitchen.jpg){:class="img-responsive" width="500px"}

### Hotel exterior, poolview and naturalview
![noise](/images/multitag/hotel-pool-naturalview.jpg){:class="img-responsive" width="500px"}

### Poolview and beach
![noise](/images/multitag/pool-beach.jpg){:class="img-responsive" width="500px"}


## First approach (Binary classifiers by tag)

First we spent more or less two weeks tagging about 40000 images a lot of work. 

Then instead of build a classifier with all tags we decided to build a binary classifier for each tag. We used resnet-152 pre-trained neural network to get a image embedding of size 1048 and then we build a mlp with this embedding as input to train a binary classifier for each tag.

![transfer](/images/multitag/transfer-learning.png){:class="img-responsive" width="700px"}

When we made a binary classifier for one tag the positive and negative classes remains very unbalanced. To solve this problem we use convex combination of the positive class images embeddings to increase the number of positive samples. 

This approach gave us good performance on each tag, but for images for more than one tag the performance was terrible this models predict only one of the possible tags in general. To understand why this happened we made the following plot.

![embedding_problem](/images/multitag/embedding_problem.png){:class="img-responsive" width="700px"}

*In this plot we can see three classes bathroom , bedroom and bathroom_guestroom images*

The problem is the representation that give us the pre-trained network. The images with two tags (bathroom_bedroom) aren't distributed in the middle between the clusters bathroom and bedroom. They almost belong to bathroom cluster or bedroom cluster, with this representation it's impossible to predict bathroom and bedroom tags for those images.

## Second approach (Cropping + binary classifiers heuristic)

![crops](/images/multitag/crops.png){:class="img-responsive" width="500px"}

When we as humans classify an image we put attention on different parts of the image. We simulate this taking random crops of the image and we used the previous binary classifiers to predict over all the crops and if the amount of crops classified as positive is greatest than a threshold then we classify the image as a positive otherwise negative. With this heuristic approach we improved the multitag classification, but  worsened the false positive rate.


## Third approach(Cropping + LSTM binary classifiers)

![crops](/images/multitag/crops_lstm.png){:class="img-responsive" width="500px"}

Instead of use hard-code rules as in the previous approach we take the crops and feed a the crops embeddings to a neural network that learns the patters between the crops to classify a tag.
This approach improve the multitag classification a reduced a lot the false positive rate that suffer the previous approach.


## Some results of lstm approach
![crops](/images/multitag/results1.png){:class="img-responsive" width="2000px"}

![crops](/images/multitag/results2.png){:class="img-responsive" width="2000px"}

![crops](/images/multitag/results3.png){:class="img-responsive" width="2000px"}








