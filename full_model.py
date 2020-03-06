# We find similar images in a database by using transfer learning via a pre-trained VGG-19 image classifier. We retreive the 10 most similar images for each image in the database.

import sys, os, argparse
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras import applications
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--dataset", type=str, help = "path to html files")
args = vars(ap.parse_args())

def save_to_csv(X, filename_heads, feature_length):
    df = pd.DataFrame(columns=['images', 'features'])
    for i in range(len(X)):
        df.loc[i] = [filename_heads[i], [X[i][j] for j in range(feature_length)]]
    with open('features_resnet.csv', 'w') as f:
        df.to_csv(f, header=f.tell()==0)
    del df

# Load pre-trained VGG-19 model and extract features from the deepest convolutional layer: fc1
ch = input("Press 1 to use VGG19 model\nPress 2 to use ResNet model")
if ch=='1':
    print("Loading VGG-19 pre-trained model...")
    base_model=applications.VGG19(weights='imagenet')
    print(base_model.summary())
    model = Model(input=base_model.input,output=base_model.get_layer('fc1').output) #try extracting from a different layer

else:
    print("Loading ResNet pre-trained model...")
    base_model=applications.resnet_v2.ResNet152V2(weights='imagenet')
    print(base_model.summary())
    model = Model(input=base_model.input,output=base_model.get_layer('post_bn').output) #try extracting from a different layer


# Read images and convert them to feature vectors
imgs, filename_heads, X, full_path = [], [], [], []
path = args['dataset'] 
print("Reading images from '{}' directory...\n".format(path))

for f in tqdm(os.listdir(path)):
    # Process filename
    filename = os.path.splitext(f)  # filename in directory
    filename_full = os.path.join(path,f)  # full path filename
    full_path.append(filename_full)
    head, ext = filename[0], filename[1]
    if ext.lower() not in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        continue

    # Read image file
    img = image.load_img(filename_full, target_size=(224,224))  # resize images as required by the pre-trained model
    filename_heads.append(head)  # filename head

    # Pre-process for model input
    img = image.img_to_array(img)  # convert to array
    img = np.expand_dims(img, axis=0)
    features = model.predict(img).flatten()  # features
    feature_length = len(features)
    X.append(features)  # append feature extractor

X = np.array(X)  # feature vectors

print("Saving to csv file")
save_to_csv(X, filename_heads, feature_length)

print("X_features.shape = {}\n".format(X.shape))
