#!/usr/bin/env python3
import configparser
from utils import read_csv_data
from model import NeuralNetwork
import pdb

import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import numpy as np



if __name__ == '__main__':
    config = configparser.RawConfigParser()
    config.read('config.cfg')
    train_data_filename = config.get('data', 'train_data')
    test_data_filename = config.get('data', 'test_data')
    train_x, train_y = read_csv_data(train_data_filename, data_type='train')
    test_x, _ = read_csv_data(train_data_filename, data_type='test')
    train_dataset = Data.TensorDataset(
      torch.from_numpy(np.expand_dims(train_x, axis=1)),
      torch.from_numpy(train_y))
    test_dataset = Data.TensorDataset(
      torch.from_numpy(np.expand_dims(test_x, axis=1)))

    batch_size = config.getint('train', 'batch_size')
    train_loader = Data.DataLoader(
      dataset=train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=2)

    model = NeuralNetwork()
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    ceriation = nn.CrossEntropyLoss()

    ave_loss = 0
    for epoch in range(20):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            x = x.view(-1, 1, 28, 28)
            out = model(x)
            loss = ceriation(out, y)
            ave_loss = ave_loss * 0.9 + loss.data * 0.1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
                total_cnt = 0
                correct_cnt = 0
                _, pred_label = torch.max(out.data, 1)
                total_cnt += x.data.size()[0]
                correct_cnt += (pred_label == y.data).sum().type(torch.FloatTensor)

                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}, acc.: {:.6f}'\
                  .format(epoch, batch_idx+1, ave_loss, correct_cnt / total_cnt))

