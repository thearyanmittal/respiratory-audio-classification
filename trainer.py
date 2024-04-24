from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from losses.focal_loss import FocalLoss

class Trainer:
    def __init__(self, model, trainset, testset, num_epochs, batch_size, lr, device='cpu'):
        self.model = model.to(device)
        self.trainset = trainset
        self.testset = testset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        self.loss_per_epoch_train = []
        self.loss_per_epoch_test = []
        self.acc_per_epoch_train = []
        self.acc_per_epoch_test = []

    def train(self, criterion=None, optimizer=None, cls_list=None):
        if criterion == None:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = FocalLoss(cls_list, gamma=1, device=self.device)
        if optimizer == None:
            optimizer = Adam(self.model.parameters(), lr=self.lr)

        train_dataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0
            correct = 0
            total = 0

            with tqdm(train_dataloader, unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {epoch + 1}/{self.num_epochs}')
                for idx, data in enumerate(tepoch):
                    images, labels = data['image'], data['sound']
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total += len(labels)
                    correct += (outputs.argmax(dim=1) == labels).sum().item()
                    running_loss += loss.item()
                    tepoch.set_postfix(
                        loss=running_loss / (idx + 1), accuracy=correct / total
                    )
            
            self.loss_per_epoch_train.append(running_loss / len(train_dataloader))
            self.acc_per_epoch_train.append(correct / total)

            # validation
            self.model.eval()
            with torch.no_grad():
                test_loss = 0
                test_correct = 0
                test_total = 0
                for idx, data in enumerate(test_dataloader):
                    inputs, labels = data['image'], data['sound']
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    test_total += len(labels)
                    test_correct += (outputs.argmax(dim=1) == labels).sum().item()
                print(
                    f"Epoch {epoch + 1}: Validation Loss: {test_loss / len(test_dataloader):.2f}, Validation Accuracy: {test_correct / test_total:.3f}"
                )
                self.loss_per_epoch_test.append(test_loss / len(test_dataloader))
                self.acc_per_epoch_test.append(test_correct / test_total)
        
    def history(self):
        return self.loss_per_epoch_train, self.loss_per_epoch_test, self.acc_per_epoch_train, self.acc_per_epoch_test

    def eval(self):
        test_dataloader = DataLoader(self.testset, batch_size=1)
        self.model.eval()

        predict_probs = []
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data['image'], data['sound']
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                predict_probs.append(F.softmax(outputs, dim=1))
                predictions.append(outputs.argmax(dim=1))
                ground_truth.append(labels)

        predict_probs = torch.cat(predict_probs).to(self.device)
        predictions = torch.cat(predictions).to(self.device)
        ground_truth = torch.cat(ground_truth).to(self.device)

        sound_correct = len(ground_truth[(ground_truth > 0) & (ground_truth == predictions)])
        sound_total = len(ground_truth[ground_truth > 0])
        neither_correct = len(ground_truth[(ground_truth == 0) & (ground_truth == predictions)])
        neither_total = len(ground_truth[ground_truth == 0])
        
        return {
            'sensitivity': sound_correct / sound_total,
            'specificity': neither_correct / neither_total,
            'icbhi': 1/2 * ((sound_correct / sound_total) + (neither_correct / neither_total)),
            'raw': (sound_correct, neither_correct, sound_total, neither_total),
            'accuracy': (sound_correct + neither_correct) / (sound_total + neither_total)
        }
