# Content Based Image Retrieval System

It takes the output of a given layer from the neural network model. VGG19 and ResNet152_V2 has been used. This model takes into account the content of the image while extracting similar images from the dataset.

### Prerequisites

You must resize the image to (224,224) dimension.

### How to run


1.) for extracting features and saving it to a .csv file do 
'''
python full_model.py --dataset PATH_TO_DATASET
'''

2.) for testing any query image, put them into a folder and then run
'''
python query_image_retrieval.py --query_images QUERY_IMAGE_FOLDER --dataset PATH_TO_DATASET
'''
 
