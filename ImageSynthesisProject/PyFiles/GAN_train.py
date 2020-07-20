import argparse
import json
import os
import sys
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from torchvision import transforms
from torch.utils.data import DataLoader
from PyFiles.GAN_model import Discriminator, Generator
from PyFiles.GAN_imageloader import CustomDataLoader

#--------------------------------------------------------------------------------------------
# Define a function to load the Pytoch model from the your respective directory
# This function will load the Generator
def model_fn(model_dir):
    print("Loading model.")

    device = torch.device("cuda is available" if torch.cuda.is_available() 
                            else "cpu is available")

    G = Generator(z_size=100, conv_dim=64)

    model_info_path = os.path.join(model_dir, 'generator_model.pt')

    with open(model_info_path, 'rb') as f:
        G.load_state_dict(torch.load(f))
    
    G.to(device).eval()
    print('Finished loading model.')
    return G
#--------------------------------------------------------------------------------------------
# Define a function that will rescale an the image so that pixel values are
# between -1 and 1 by default
def scale(x, feature_range=(-1, 1)):
    min, max = feature_range 
    x = x * (max - min) + min
    return x
#--------------------------------------------------------------------------------------------
# Define Loss functions
def real_loss(D_out, train_on_gpu):
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size)

    if train_on_gpu:
        labels = labels.cuda()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)

    return loss

def fake_loss(D_out, train_on_gpu):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)

    if train_on_gpu:
        labels = labels.cuda()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)

    return loss
#--------------------------------------------------------------------------------------------
# Define a function to initialize the weights of a certain layers in a model
def weights_init_normal(m):
    classname = m.__class__.__name__

    std_dev = 0.02
    mean = 0

    if hasattr(m, 'weight') and (classname.find ('Conv') != -1 or 
            classname.find('Linear') != -1):
                init.normal_(m.weight.data, mean, std_dev)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, mean, std_dev)
#--------------------------------------------------------------------------------------------
# Define a function to Generate the full model with Generator & Disciminator
# This function will have different sizes of layers but will still have 
# the same 4 layers and 1 fully connected layers in the Generator & Discriminator
def build_network(d_conv_dim, g_conv_dim, z_size):
    D = Discriminator(conv_dim=d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    D.apply(weights_init_normal)
    G.apply(weigths_init_normal)

    return D, G
#--------------------------------------------------------------------------------------------
# Define a function that will load data
def get_dataloader(batch_size, image_size, data_dir, csv_file):
    transform_image = transforms.Compose([transforms.Resize(image_size),
                        transforms.RandomChoice([transforms.RandomVerticalFlip(p=0.7),
                                                 transforms.RandomHorizontalFlip(p=0.7),
                                                 transforms.ColorJitter(contrast=1.0, 
                                                 saturation=2.0, hue=0.2)]),
                        transforms.RandomOrder([transforms.RandomVerticalFlip(p=0.7),
                                                 transforms.RandomHorizontalFlip(p=0.7),
                                                 transforms.ColorJitter(contrast=1.0,
                                                 saturation=2.0, hue=0.2)]),
                        transforms.ToTensor()])

    data_ImageFolder = CustomDataLoader(csv_file, data_dir, transform=transform_image)
    data_loader = torch.utils.data.DataLoader(data_ImageFolder, batch_size=batch_size,
                                              shuffle=True)
    return data_loader
#--------------------------------------------------------------------------------------------
# Define a function to train your model
def train(D, G, z_size, train_loader, n_epochs, d_optimizer, g_optimizer, train_on_gpu, print_every=20):
    if train_on_gpu:
        D.cuda()
        G.cuda()
    
    samples = []
    losses = []

    sample_size = 20
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    if train_on_gpu:
        fixed_z = fixed_z.cuda()
    
    for epoch in range(n_epochs):
        for batch_i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size()[0]
            real_images = scale(real_images)

            # ======================================
            #           TRAIN DISCRIMINATOR
            # ======================================
            d_optimizer.zero_grad()
            if train_on_gpu:
                real_images = real_images.cuda()

            D_real = D(real_images)
            d_real_loss = real_loss(D_real, train_on_gpu)

            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()

            if train_on_gpu:
                z = z.cuda()
            
            fake_images = G(z)
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake, train_on_gpu)

            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            d_optimizer.step()

            # ======================================
            #           TRAIN GENERATOR
            # ======================================
            g_optimizer.zero_grad()

            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()

            if train_on_gpu:
                z = z.cuda()
            
            fake_images = G(z)
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake, train_on_gpu)

            g_loss.backward()
            g_optimizer.step()
            
            # Print loss values 
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))

        G.eval() # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to training mode

    # Save model params and training generator samples
    with open(f'epoch:{epoch}_generator_model.pt', 'wb') as f:
        torch.save(G.state_dict(), f)

    with open(f'epoch:{epoch}_train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    return G
#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--z_size', type=int, default=100, metavar='N',
                        help='input z-size for training (default: 100)')

    
    # Model Parameters
    parser.add_argument('--conv_dim', type=int, default=64, metavar='N',
                        help='size of the convolution dim (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                        help='Learning rate default 0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='N',
                        help='beta1 default value 0.5')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='N',
                        help='beta2 default value 0.999')
    parser.add_argument('--img_size', type=int, default=256, metavar='N',
                        help='Image size default value 256')
    
    
    # SageMaker Parameters/SageMaker Environment related | MUST INCLUDE
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus',type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.cuda.is_available()
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = get_dataloader(args.batch_size, args.img_size, args.data_dir)
    
    
    # Build the model.
    D, G = build_network(args.conv_dim, args.conv_dim, z_size=args.z_size)
    

    # Create optimizers for the discriminator and generator
    d_optimizer = optim.Adam(D.parameters(), args.lr, [args.beta1, args.beta2])
    g_optimizer = optim.Adam(G.parameters(), args.lr, [args.beta1, args.beta2])
    
    G = train(D, G, args.z_size, train_loader, args.epochs, d_optimizer, g_optimizer, device)

	# Save the model parameters
    G_path = os.path.join(args.model_dir, 'generator_model_main.pt')
    with open(G_path, 'wb') as f:
        torch.save(G.cpu().state_dict(), f) 