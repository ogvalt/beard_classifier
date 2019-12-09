import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import BeardDataset
from model import get_model


def evaluate(model, device, data_loader):
    model.eval()
    df_report = pd.DataFrame(data=data_loader.dataset.classes, index=[0]
                             ).T.reset_index().drop(columns=0).rename(columns={'index': 'name'})
    df_report['TP'], df_report['Total'] = 0, 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for cl_idx in target.unique():
                cl_idx = cl_idx.item()
                single_class_prediction = pred[target == cl_idx]
                tp = sum(single_class_prediction == cl_idx)
                class_total = sum(target == cl_idx)
                df_report.loc[cl_idx, 'TP'] += tp.item()
                df_report.loc[cl_idx, 'Total'] += class_total.item()

    df_report['Accuracy, %'] = np.round(df_report['TP'] / df_report['Total'], 4) * 100
    correct = int(df_report['TP'].sum())
    print(df_report)

    print('Average accuracy: {}/{} ({:.1f}%)'.format(
        correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))


def main():
    batch_size = 64
    use_cuda = torch.cuda.is_available()
    print("Is cuda? ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_path = 'dataset'
    models_path = 'checkpoints'
    imgs_path = os.path.join(dataset_path, "data_for_test")
    lbls_path = os.path.join(dataset_path, "labels_for_test.json")

    beards = BeardDataset(images_path=imgs_path, label_path=lbls_path,
                          transform=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                          ]))

    data_loader = DataLoader(beards, batch_size=batch_size, num_workers=2, pin_memory=True)

    model = get_model(len(beards.classes))

    for model_name in os.listdir(models_path):
        model.load_state_dict(torch.load(os.path.join(models_path, model_name)))
        model.to(device)
        print(model_name)
        evaluate(model, device, data_loader)


if __name__ == "__main__":
    main()
