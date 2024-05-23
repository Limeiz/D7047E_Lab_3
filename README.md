# D7047E_Lab_3

## Overview
Image captioning is the task of generating textual descriptions for images automatically. This project aims to build an image captioning system using deep learning techniques.

## Pre-stuff
Download the dataset used [here](https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb). Then set the "images" folder and captions.txt inside a folder called "flickr8k".

Alternative dataset 'coco' [train](http://images.cocodataset.org/zips/train2014.zip) and [validate](http://images.cocodataset.org/zips/val2014.zip)

Alternative [captions](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) for both (used with coco)

## Good to know
If you want to save the trained model, be sure the set save_model to True in train.py.

## Results
The generated captions are in the results folder, using the pictures in the test_examples folder. 

### EISL results
```
Three stages [a,b,c]
Stage a weighted words, 70% ce & 30% EISL, when not improving =>
Stage b weighted words, 30% ce & 70% EISL, when not improving =>
Stage c no word weights, 0% ce and 100% EISL (true to paper)
```
```
Stage: a
Example 0 CORRECT: Dog on a beach by the ocean
           OUTPUT: <SOS> a dog that is laying down on the ground <EOS>
Example 1 CORRECT: Child holding red frisbee outdoors
           OUTPUT: <SOS> a little boy that is holding a bat <EOS>
Example 2 CORRECT: Bus driving by parked cars
           OUTPUT: <SOS> a double decker bus driving down the street <EOS>
Example 3 CORRECT: A small boat in the ocean
           OUTPUT: <SOS> a boat sailing across the water near a dock . <EOS>
Example 4 CORRECT: A cowboy riding a horse in the desert
           OUTPUT: <SOS> a group of people riding horses across a dirt road . <EOS>
Example 5 CORRECT: Man in black hat
           OUTPUT: <SOS> a woman sitting on a park bench next to a dog <EOS>
```
```
Stage: b
Example 0 CORRECT: Dog on a beach by the ocean
           OUTPUT: <SOS> a dog is running along the beach with a frisbee <EOS>
Example 1 CORRECT: Child holding red frisbee outdoors
           OUTPUT: <SOS> a little girl holding up a frisbee in her hand <EOS>
Example 2 CORRECT: Bus driving by parked cars
           OUTPUT: <SOS> a double decker bus driving down the street <EOS>
Example 3 CORRECT: A small boat in the ocean
           OUTPUT: <SOS> a boat that is floating in the water <EOS>
Example 4 CORRECT: A cowboy riding a horse in the desert
           OUTPUT: <SOS> a man riding on the back of an elephant <EOS>
Example 5 CORRECT: Man in black hat
           OUTPUT: <SOS> a man wearing glasses and a hat is sitting on a bench <EOS>
```
```
Stage: c
Example 0 CORRECT: Dog on a beach by the ocean
           OUTPUT: <SOS> a dog is standing on a beach with a frisbee in its mouth . <EOS>
Example 1 CORRECT: Child holding red frisbee outdoors
           OUTPUT: <SOS> a little boy holding a baseball bat next to a man . <EOS>
Example 2 CORRECT: Bus driving by parked cars
           OUTPUT: <SOS> a double decker bus driving down a city street . <EOS>
Example 3 CORRECT: A small boat in the ocean
           OUTPUT: <SOS> a boat is docked at the shore of a beach . <EOS>
Example 4 CORRECT: A cowboy riding a horse in the desert
           OUTPUT: <SOS> a man riding a horse in a dirt field . <EOS>
Example 5 CORRECT: Man in black hat
           OUTPUT: <SOS> a man sitting on a bench with a dog . <EOS>
```
