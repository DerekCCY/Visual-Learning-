import torch
from voc_dataset import VOCDataset
import utils
import random
from train_q2 import ResNet
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select 1000 data from test dataset randomly
pascal_test = VOCDataset(split='val', size=224, data_dir = '/home/ubuntu/hw1-main/q1_q2_classification/data/VOCdevkit/VOC2007')
#print(len(pascal_test))
#print(pascal_test[0])
random_indices = random.sample(range(len(pascal_test)), 1000)
random_images = [pascal_test[idx] for idx in random_indices]

# import trained model 
model = torch.load('/home/ubuntu/hw1-main/q1_q2_classification/checkpoint-model-epoch50.pth').to(device)
model.eval()

# No need to do transformation since they do it in VOCDataset
def extract_features(model, inputs):
    features = []
    with torch.no_grad():
        for image, label, weight in inputs:
            image = image.unsqueeze(0).to(device)
            output = model(image)
            features.append(output.squeeze().cpu().numpy())
    return np.array(features)

features = extract_features(model, random_images)
#print(features)

#xStart tsne
tsne  = TSNE(n_components=2,random_state=56)
features_2d = tsne.fit_transform(features)

num_classes = 20
CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
colors = plt.cm.get_cmap('tab20', num_classes)

# To record the classes indices [0,0,0,0,1,0,1] --> 4,6
def get_active_classes(label):
    active_classes = []
    for idx, label in enumerate(label):
        if label > 0:  # Assuming label > 0 indicates the class is active
            active_classes.append(idx)
    return active_classes

image_colors = []
for image, label, weight in random_images:
    active_classes = get_active_classes(label)
    if active_classes:
        mean_color = np.mean([colors(i) for i in active_classes], axis=0)
        image_colors.append(mean_color)
    else:
        image_colors.append([0, 0, 0, 1])  # Black for images with no active classes

# Convert colors to RGBA
image_colors = np.array(image_colors)

# Plot the 2D t-SNE features
plt.figure(figsize=(12, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=image_colors, s=10)

# Add legend mapping colors to classes
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{CLASS_NAMES[i]}',
                              markerfacecolor=colors(i), markersize=10) for i in range(num_classes)]
plt.legend(handles=legend_elements, loc='best', title="Classes")

plt.title('t-SNE Projection of PASCAL VOC Features')
plt.show()
plt.savefig("t-SNE Projection of PASCAL VOC Features.png")

