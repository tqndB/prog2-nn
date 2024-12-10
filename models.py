from torch import nn


class MyModel(nn.Model):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.netwark(x)
        return logits


def test(model,deraloader, loss_fn):
    loss_total = 0.0

    model.eval()
    for image_batch, label_batch in dataloader:
        with torch.no_grad():
            logits_batch = model(image_batch)

        loss = loss_fn(logits_batch, label_batch)
        loss_total += loss.item()

    return loss_total / len(datalosder)

def test(model, dataloader, loss_fn):
    loss_total = 0.0

    model.eval()
    for image_batch, label_batch in dataloader:
        with torch.no_grad():
            logits_batch = model(image_batch)

        loss = loss_fn(logits_batch, label_batch)
        loss_total += loss.item()

    return loss_total / len(dataloader)