import sys, os, argparse
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras import applications
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ast import literal_eval

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--query_images", type=str, help = "path to query image folder")
ap.add_argument("-d", "--dataset", type=str, help= "path to dataset folder")
args = vars(ap.parse_args())

def rescale_image(image):
    size = np.asarray(image.size)
    size = (size * IMAGE_SIZE / min(size)).astype(int)
    image = image.resize(size, resample=Image.LANCZOS)
    w, h = image.size
    image = image.crop((
        (w - IMAGE_SIZE) // 2,
        (h - IMAGE_SIZE) // 2,
        (w + IMAGE_SIZE) // 2,
        (h + IMAGE_SIZE) // 2)
    )
    return image

def read_csv():

    data = pd.read_csv('features_resnet.csv')
    X = data['features'].apply(literal_eval).values
    X = np.vstack(X[:])
    return X

X = read_csv()
print(X.shape)
print("Loading VGG-19 pre-trained model...")
i = input("Press 1 for VGG19 architecture\nPress 2 for Res-Net")
i='2'
if i=='1':
    base_model=applications.VGG19(weights='imagenet')
    model = Model(input=base_model.input,output=base_model.get_layer('fc1').output) # type the name of layer of which output you want to extract
else:
    base_model=applications.resnet_v2.ResNet152V2(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('post_bn').output) # type the name of layer of which output you want to extract


# loading images from query
imgs = []
query_img_folder = args['query_images']
query_image_codes, query_full_path = [], []
for i in os.listdir(query_img_folder):
    img = image.load_img(os.path.join(query_img_folder,i), target_size=(224,224))
    query_full_path.append(os.path.join(query_img_folder, i))
    img = image.img_to_array(img)  # convert to array
    img = np.expand_dims(img, axis=0)
    features = model.predict(img).flatten()  # features
    query_image_codes.append(features)

query_image_codes = np.array(query_image_codes)
print(query_image_codes.shape)


neighbors_count = 10
nearest_neighbors = NearestNeighbors(n_neighbors=neighbors_count, metric='cosine').fit(X)
_, indices = nearest_neighbors.kneighbors(query_image_codes)


IMAGE_SIZE = 224
space = 10
result_image_size = (
    (neighbors_count + 1) * (IMAGE_SIZE + space) - space,
    len(query_full_path) * (IMAGE_SIZE + space) - space
)

full_path = []
dataset = args['dataset']
for f in os.listdir(dataset):
    full_path.append(os.path.join(dataset, f))

result_image = Image.new('RGB', result_image_size, 'white')
for i, filename in enumerate(query_full_path):
    query_image = rescale_image(Image.open(filename))
    draw = ImageDraw.Draw(query_image)
    draw.line(
        (
            0, 0,
            query_image.width - 1, 0,
            query_image.width - 1, query_image.height - 1,
            0, query_image.height - 1,
            0, 0
        ),
        fill='red', width=1)
    result_image.paste(query_image, (0, i * (IMAGE_SIZE + space)))
    for j in range(neighbors_count):
        neighbor_image = Image.open(full_path[indices[i][j]])
        result_image.paste(neighbor_image, ((j + 1) * (IMAGE_SIZE + space), i * (IMAGE_SIZE + space)))

print("Saving result.jpg to current directory!")
result_image.save('result.jpg')
