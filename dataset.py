import os
import json

from skimage import io

import torch
from torch import randperm
from torch.utils.data.dataset import Dataset


class BeardDataset(Dataset):
    def __init__(self, images_path, label_path, transform=None):
        self.images_path = images_path
        self.labels_path = label_path
        self.transform = transform

        self.images = os.listdir(self.images_path)

        with open(self.labels_path) as f_labels:
            self.labels = json.load(f_labels)

        self.classes = self.__class_to_index()

    def __class_to_index(self):
        return {label: idx for idx, label in enumerate(sorted(list(set(self.labels.values()))))}

    def read_image(self, image_name):
        img_path = os.path.join(self.images_path, image_name)
        img = io.imread(img_path)
        return img

    def get_label(self, image_name):
        return self.classes[self.labels[image_name]]

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img = self.read_image(image_name)
        label = self.get_label(image_name)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)


class SubsetWithDistribution(Dataset):
    def __init__(self, dataset: BeardDataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.samples_weights = self.calculate_samples_weights()

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_class_distribution(self):
        class_distribution = {}

        for idx in self.indices:
            image_name = self.dataset.images[idx]
            label = self.dataset.get_label(image_name)

            if label not in class_distribution.keys():
                class_distribution[label] = 1
            else:
                class_distribution[label] += 1

        return class_distribution

    def get_class_weights(self):
        class_dist = self.get_class_distribution()
        return [len(self.dataset) / item[1] for item in sorted(list(class_dist.items()))]

    def calculate_samples_weights(self):
        class_weights = self.get_class_weights()
        samples_weights = []
        for idx in self.indices:
            image_name = self.dataset.images[idx]
            label = self.dataset.get_label(image_name)
            samples_weights.append(class_weights[label])
        return samples_weights


def train_test_split(dataset, test_fraction=0.2):
    if not 0 < test_fraction < 1:
        raise ValueError("Test set fraction should be in (0, 1) interval")

    train_len = int((1 - test_fraction) * len(dataset))
    indices = randperm(len(dataset)).tolist()

    return SubsetWithDistribution(dataset, indices[:train_len]), SubsetWithDistribution(dataset, indices[train_len:])

# TODO: Add mean and std calculation for centering
# def calculate_mean(self):
#     temp = self.__getitem__(0)[0]
#     for idx in range(1, self.__len__()):
#         temp += self.__getitem__(idx)[0]
#
#     return torch.sum(temp, (1, 2)) / self.__len__()
#
# def calculate_std(self):
#     pass


if __name__ == "__main__":
    from torchvision import transforms
    import matplotlib.pyplot as plt

    dataset_path = 'dataset'
    imgs_path = os.path.join(dataset_path, "data_for_test")
    lbls_path = os.path.join(dataset_path, "labels_for_test.json")

    tran = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Resize(256),
                          transforms.ToTensor(),
                      ])

    dt = BeardDataset(images_path=imgs_path, label_path=lbls_path,
                      transform=tran)

    tr, te = train_test_split(dt)

    plt.imshow(dt[1][0].permute(1, 2, 0))
    plt.show()
