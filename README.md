# D7047E Lab 3: Image Captioning with Deep Learning

## Overview
This project builds an image captioning system using **deep learning**. It uses **CNNs** to extract image features and **LSTMs** for generating captions based on the extracted features.

## Dataset
Download the **Flickr8k** dataset [here](https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb). Organize: flickr8k/images/captions.txt

Alternatively dataset 'coco' [train](http://images.cocodataset.org/zips/train2014.zip) and [validate](http://images.cocodataset.org/zips/val2014.zip)

Alternative [captions](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) for both (used with coco)

## Running the Project

1. **Clone the repository** and navigate to the project folder.

2. **Download the dataset** (Flickr8k or COCO) and organize it as described in the folder structure.

3. **Preprocess the data** (e.g., resize images, tokenize captions).

4. **Train the model** by running the following command:
   ```bash
   python train.py --save_model True
    ```
5. Generate captions using the trained model on test images in the test_examples/ folder.
