

import torchvision
image_path = './'
celeba_dataset = torchvision.datasets.CelebA(image_path, split='train', target_type='attr', download=True)


from torch.utils.data import Dataset
assert isinstance(celeba_dataset, Dataset)
example = next(iter(celeba_dataset))
print(example)

import matplotlib.pyplot as plt
from itertools import islice 
fig = plt.figure(figsize=(12, 8))
for i, (image, attributes) in islice(enumerate(celeba_dataset), 18):
    ax = fig.add_subplot(3, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image)
    ax.set_title(f'{attributes[31]}', size=15)

plt.show()