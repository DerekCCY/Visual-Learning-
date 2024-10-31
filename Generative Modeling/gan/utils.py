import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision.utils as vutils


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    ##################################################################
    # TODO: 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors. Do so by linearly
    # interpolating for 10 steps across each of the first two
    # dimensions between -1 and 1. Keep the rest of the z vector for
    # the samples to be some fixed value (e.g. 0).
    # 2. Forward the samples through the generator.
    # 3. Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    #################################################################
    steps = 10
    dim = 128  # Dimension of the latent space
    #z_fixed = torch.zeros(dim - 2)  # Fixed part of the latent vector (0 for all dimensions)
#
    ## Create a grid for interpolation
    #z_samples = []
    #for x in torch.linspace(-1, 1, steps):
    #    for y in torch.linspace(-1, 1, steps):
    #        # Create a latent vector with interpolated x and y
    #        z = torch.cat((torch.tensor([x, y]), z_fixed))
    #        z_samples.append(z)
#
    ## Stack to create a tensor of shape (100, 128)
    #z_samples = torch.stack(z_samples)  # Shape: (100, 128)
#
    ## Step 2: Forward the samples through the generator
    #gen.eval()  # Set generator to evaluation mode
    #with torch.no_grad():
    #    generated_images = gen(z_samples)
#
    ## Step 3: Save out an image holding all 100 samples
    ## Reshape the images if needed (assuming generated_images has shape (100, C, H, W))
    #grid = vutils.make_grid(generated_images, nrow=10, normalize=True, range=(-1, 1))
    #vutils.save_image(grid, path)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
