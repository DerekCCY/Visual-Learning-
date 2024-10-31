import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision.utils as vutils
from networks import *

def interpolate_latent_space(gen, path):
    steps = 10
    dim = 128  # Dimension of the latent space
    z_fixed = torch.zeros(dim - 2)  # Fixed part of the latent vector (0 for all dimensions)

    # Create a grid for interpolation
    z_samples = []
    for x in torch.linspace(-1, 1, steps):
        for y in torch.linspace(-1, 1, steps):
            # Create a latent vector with interpolated x and y
            z = torch.cat((torch.tensor([x, y]), z_fixed))
            z_samples.append(z)

    # Stack to create a tensor of shape (100, 128)
    z_samples = torch.stack(z_samples).cuda()  # Shape: (100, 128)

    # Step 2: Forward the samples through the generator
    n_samples = z_samples.size(0)
    gen.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        generated_images = gen(n_samples)  # Only pass z_samples

    # Step 3: Save out an image holding all 100 samples
    # Step 3: Save out an image holding all 100 samples
    grid = vutils.make_grid(generated_images, nrow=10, normalize=True)# Remove range
    vutils.save_image(grid, path)

# Load the model
gen = torch.jit.load('/home/ccy/CCY/data_wgan_gp/generator.pt').cuda()  
gen.eval()
interpolate_latent_space(gen, 'latent_space_interpolations.png')
