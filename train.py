import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchsummary import summary

import pandas as pd

from dataset import *
from model import get_model


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += sum(pred == target)

    train_loss /= len(train_loader.dataset)
    acc = correct.item() / len(train_loader.dataset)
    print('Train Epoch: {} Loss: {:.4f} Accuracy: {}/{} ({:.1f}%)'.format(epoch, train_loss, correct,
                                                                    len(train_loader.dataset), acc*100))


def test(model, device, data_loader):
    model.eval()
    df_report = pd.DataFrame(data=data_loader.dataset.dataset.classes, index=[0]
                             ).T.reset_index().drop(columns=0).rename(columns={'index': 'name'})
    df_report['TP'], df_report['FN'] = 0, 0
    test_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            for cl_idx in target.unique():
                cl_idx = cl_idx.item()
                single_class_prediction = pred[target == cl_idx]
                tp = sum(single_class_prediction == cl_idx)
                fn = sum(single_class_prediction != cl_idx)
                df_report.loc[cl_idx, 'TP'] += tp.item()
                df_report.loc[cl_idx, 'FN'] += fn.item()

    test_loss /= len(data_loader.dataset)
    df_report['Recall'] = df_report['TP'] / (df_report['TP'] + df_report['FN'])
    correct = int(df_report['TP'].sum())
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    print(df_report)


def main():
    seed = 16
    batch_size = 4
    epochs = 45
    use_cuda = torch.cuda.is_available()
    print("Is cuda? ", use_cuda)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_path = 'dataset'
    imgs_path = os.path.join(dataset_path, "data_for_test")
    lbls_path = os.path.join(dataset_path, "labels_for_test.json")

    beards = BeardDataset(images_path=imgs_path, label_path=lbls_path)
    trainset, testset = train_test_split(beards, test_fraction=0.15)

    # resnet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.RandomAffine(degrees=5, translate=(.15, 0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])

    trainset.dataset.transform = train_transforms
    testset.dataset.transform = test_transforms

    train_sampler = WeightedRandomSampler(weights=trainset.samples_weights, num_samples=len(trainset.samples_weights))

    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=2, pin_memory=True)

    model = get_model(num_classes=len(beards.classes))
    model = model.to(device)

    summary(model, input_size=(3, 224, 224))

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        torch.save(model.state_dict(),
                   os.path.join("checkpoints",
                                'beard_resnet34_epoch_{}.pt'.format(epoch)))


if __name__ == "__main__":
    main()
