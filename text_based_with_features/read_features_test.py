import json
import numpy as np # You may want to convert the list format into numpy

with open('img_features.json', 'r') as f:
    feat = json.load(f)
    print(feat['image1'])
    print(np.array(feat['image1']).shape) # You may want to convert the list format into numpy


