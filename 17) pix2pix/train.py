import utils
from torchvision import transforms
from network import Generator, Discriminator
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn.functional as F



# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the Discriminator
netD = Discriminator().to(device)

# Create the generator
netG = Generator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(utils.weights_init)
netD.apply(utils.weights_init)

# Print the model
print(netG)
print(netD)


# loss
BCE_loss = nn.BCELoss().to(device)
L1_loss = nn.L1Loss().to(device)


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# set hypermaters
lr = 0.0002
beta1 = 0.5
num_epochs = 200
L1_lambda = 100

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Lists to keep track of progress

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []


# DataLoader
# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainloader = utils.dloader(path, resize=(512, 256), transform=transform, shuffle=True)

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (X, Y) in enumerate(trainloader, 0):

        D_losses = []
        G_losses = []

        ##############################
        # Train D with all-real batch
        ##############################

        netD.zero_grad()
        # Format batch
        x_ = X.to(device)
        y_ = Y.to(device)

        b_size = x_.size(0)

        labelR = torch.full((30, 30), real_label,
                            dtype=torch.float, device=device)
        labelF = torch.full((30, 30), fake_label,
                            dtype=torch.float, device=device)

        # Forward pass real batch through D
        outputD = netD(x_, y_).squeeze()
        # print(outputD.size())

        # Calculate loss on all-real batch
        D_real_loss = BCE_loss(outputD, labelR)

        G_result = netG(x_)
        D_result = netD(x_, G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, labelF)

        D_train_loss = (D_real_loss + D_fake_loss) * 0.5
        D_train_loss.backward()
        optimizerD.step()

        train_hist['D_losses'].append(D_train_loss.item())

        D_losses.append(D_train_loss.item())

        ############################
        # Train Generator
        ###########################

        netG.zero_grad()

        G_result = netG(x_)
        D_result = netD(x_, G_result).squeeze()

        G_train_loss = BCE_loss(D_result, labelR) + \
            L1_lambda * L1_loss(G_result, y_)
        G_train_loss.backward()
        optimizerG.step()

        train_hist['G_losses'].append(G_train_loss.item())

        G_losses.append(G_train_loss.item())

    print('[%d/%d] - loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), num_epochs,
                                                    torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))


root = '/content/drive/My Drive/data/pix2pix/maps/'
print("Training finish!... save training results")
torch.save(netG.state_dict(), root + '/generator_param.pkl')
torch.save(netD.state_dict(), root + '/discriminator_param.pkl')