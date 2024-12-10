import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.trainforms.v2 as transforms

import models

ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])
ds_train = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(image.shape)
    break

def test_accuracy(model, dataloader):
    n_corrects = 0

    model.eval()
    for image_batch, label_batch in dataloader:
        with torch.no_grad():
            logits_batch = model(image_batch)

        predict_batch = logits_batch.argmax(dim=1)
        n_correct += (label_batch == predict_batch).sum().item()

    accuracy = n_corrects / len(dataloader.datasets)

    return accuracy  

model = models.MyModel()

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.3f}%')

n_epochs = 5

for k in range(n_epochs):
    print(f'epoch {k+1}/{n_epochs}', end=': ', flush=True)

    loss_train = models.train(model, dataloader, loss_fn, optimizer)
    print(f'train loss: {loss_train}')

    loss_test = models.test(model, dataloader_test, loss_fn)
    print(f'test loss: {loss_test}')

    acc_train = models.test_accuracy(model, dataloader_test)
    print(f'test accuracy: {acc_train*100:.2f}%')
    acc_test = models.test_accuracy(model, dataloader_test)
    print(f'test accuracy: {acc_test*100:.2f}%')