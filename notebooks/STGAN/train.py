import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.autograd import Variable

import STGAN.data.Loader as ld

import STGAN.model.DiscriminatorNet as DiscriminatorNet
import STGAN.model.Generator as Generator

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Nets
generatorNet = Generator.GeneratorNet()
discriminatorNet = DiscriminatorNet.DiscriminatorNet()

# Hyperparameters
lr = 0.001
n_epochs = 20
latent_dim = 100

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Configure data loader
train_data_loader = ld.get_data()

# Optimizers
optimizer_G = optim.Adam(generatorNet.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminatorNet.parameters(), lr=lr, betas=(0.5, 0.999))

# ----------
#  Training
# ----------

Tensor = torch.FloatTensor

#-----------
#  Loading
#-----------

PATH_GENERATOR_NET = "/home/student/Documents/FLORA/notebooks/STGAN/state_dict_generatornet.pth"
#generatorNet.load_state_dict(torch.load(PATH_GENERATOR_NET))
#
PATH_DISCRIMINATOR_NET = "/home/student/Documents/FLORA/notebooks/STGAN/state_dict_discriminatornet.pth"
#discriminatorNet.load_state_dict(torch.load(PATH_DISCRIMINATOR_NET))


for epoch in range(n_epochs):
    for input_img, real_img in tqdm(train_data_loader):

        # Adversarial ground truths
        valid = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(real_img.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        x = input_img.unsqueeze(0)

        gen_imgs = generatorNet(x)

        plt.imshow((gen_imgs).squeeze().permute(1, 2, 0).detach().numpy())

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminatorNet(gen_imgs), valid)

        g_loss.backward(retain_graph=True)
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminatorNet(real_imgs.unsqueeze(dim=0)), valid)
        fake_loss = adversarial_loss(discriminatorNet(gen_imgs), fake)
        d_loss = (real_loss + fake_loss) / 2
        print(real_loss, fake_loss, d_loss)
        d_loss.backward()
        optimizer_D.step()

    if epoch % 5 == 0:
        print(epoch)

torch.save(generatorNet.state_dict(), PATH_GENERATOR_NET)
torch.save(discriminatorNet.state_dict(), PATH_DISCRIMINATOR_NET)



