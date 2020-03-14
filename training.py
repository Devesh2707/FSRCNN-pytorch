from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net  
from data import get_training_set, get_test_set


upscale_factor = 2
batchSize = 4
testBatchSize = 1
nEpochs = 150
lr = 0.001
cuda = True
#Uncomment threads while using a powerful gpu
#threads = 4
seed = 123


if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run with cuda == False")
else:
    device = 'cuda'

torch.manual_seed(seed)

#device = torch.device("cuda" if cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(upscale_factor)
test_set = get_test_set(upscale_factor)


#To use with threads
#training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchSize, shuffle=True)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=testBatchSize, shuffle=False)


training_data_loader = DataLoader(dataset=train_set, batch_size=batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, batch_size=testBatchSize, shuffle=False)

print('===> Building model')
model = Net(num_channels = 1, upscale_factor=upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam([
    {'params':model.first_part.parameters(), 'lr': 1e-3},
    {'params':model.mid_part.parameters(), 'lr': 1e-3},
    {'params':model.last_part.parameters(), 'lr':1e-4},], lr=lr)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if iteration%10 == 0:
          print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    if epoch%10 == 0:
      print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def checkpoint(epoch):
    if epoch%50 == 0:
      model_out_path = "checkpoints/model_up_{}_epoch_{}.pth".format(upscale_factor,epoch)
      torch.save(model.state_dict(), model_out_path)
      print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)