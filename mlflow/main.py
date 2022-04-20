import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
import mlflow

from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchmetrics.functional import accuracy

from mlflow.tracking import MlflowClient

class MnistCNN(pl.LightningModule):
    def __init__(self, dropout=0.1) -> None:
        super().__init__()
        self.dropout = dropout

        # L1 shape = (?, 28, 28, 1)
        #    Conv  -> (?, 28, 28, 32)
        #    Pool  -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # L2 shape = (?, 14, 14, 32)
        #    Conv  -> (?, 14, 14, 64)
        #    Pool  -> (?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # L3 shape = (?, 7, 7, 64)
        #    Conv  -> (?, 7, 7, 128)
        #    Pool  -> (?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        # L4 FC 4x4x128 -> 625
        self.fc1 = nn.Linear(4 * 4 * 128, 625, bias=True)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )

        # L5 FC 625 -> 10
        self.fc2 = nn.Linear(625, 10, bias=True)

    def forward(self, x):
        outputs = self.layer1(x)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.layer4(outputs)
        outputs = self.fc2(outputs)

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        tensorboard_logs = {'train_loss': loss, 'acc': acc}
        metrics = {"acc":acc, "loss":loss, "log": tensorboard_logs}
        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x)
        # acc = accuracy(pred_y, y)
        loss = F.cross_entropy(pred_y, y)
        metrics = {"val_loss":loss, "y":y.detach(), "y_hat":pred_y.detach()}
        return metrics

    def validation_epoch_end(self, outputs):
        # print(outputs)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        acc = (y_hat.argmax(dim=1) == y).float().mean().item()
        # print(f"Epoch {self.current_epoch} acc:{acc} loss:{avg_loss}\n")

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'avg_val_loss': avg_loss,
                'val_acc': acc,
                'log': tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

if __name__ == "__main__":
    # setup data
    mnist_train = MNIST(root=os.getcwd(),
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

    mnist_test = MNIST(root=os.getcwd(),
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
    train_loader = utils.data.DataLoader(mnist_train, batch_size=64)
    val_loader = utils.data.DataLoader(mnist_test, batch_size=64)
    
    mnist = MnistCNN()

    trainer = pl.Trainer(max_epochs=20)

    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        trainer.fit(mnist, train_dataloaders=train_loader, val_dataloaders=val_loader)