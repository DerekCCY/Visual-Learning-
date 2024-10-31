import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    """
    Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (3): ReLU()
        (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (5): ReLU()
        (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    """
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        ##################################################################
        # TODO 2.1: Set up the network layers. First create the self.convs.
        # Then create self.fc with output dimension == self.latent_dim
        ##################################################################
        self.convs = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        )
        # Calculate flattened size after conv layers
        self.conv_out_size = self._get_conv_out(input_shape)

        # Fully connected layer, input is the flattened size, output is latent_dim
        self.fc = nn.Linear(self.conv_out_size, latent_dim)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def _get_conv_out(self, shape):
        # Pass a dummy tensor to calculate the conv layers output size
        o = torch.zeros(1, *shape)
        o = self.convs(o)
        return int(np.prod(o.size()))  # Flatten all dimensions except batch size

    def forward(self, x):
        ##################################################################
        # Forward pass through the network, output should be
        # of dimension == self.latent_dim
        ##################################################################
        x1 = self.convs(x)
        x1 = torch.flatten(x1, start_dim=1)  # Flatten all dimensions except batch size
        output = self.fc(x1)
        return output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        ##################################################################
        # TODO 2.4: Fill in self.fc, such that output dimension is
        # 2*self.latent_dim
        ##################################################################
        self.fc = nn.Linear(self.conv_out_size, 2 * latent_dim)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.1: Forward pass through the network, should return a
        # tuple of 2 tensors, mu and log_std
        ##################################################################
        x1 = self.convs(x)
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.fc(x1)

        
        mu, log_std = torch.chunk(x1, 2, dim=1)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return mu, log_std


class Decoder(nn.Module):
    """
    Sequential(
        (0): ReLU()
        (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (2): ReLU()
        (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (4): ReLU()
        (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (6): ReLU()
        (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    """
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        ##################################################################
        # TODO 2.1: Set up the network layers. First, compute
        # self.base_size, then create the self.fc and self.deconvs.
        ##################################################################
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,3,kernel_size=3,stride=1,padding=1)
        )
        # Calculate the base size for deconvolution
        self.base_size = output_shape[1] // 8  # 3 deconv layers, each reducing size by a factor of 2
        # e.g. if output_shape is (3, 64, 64), base_size = 64 // 16 = 4

        # Fully connected layer to reshape latent_dim to base_size
        self.fc = nn.Linear(latent_dim, 256 * self.base_size * self.base_size)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, z):
        #TODO 2.1: forward pass through the network, 
        ##################################################################
        # TODO 2.1: Forward pass through the network, first through
        # self.fc, then self.deconvs.
        ##################################################################
        z = self.fc(z)  # Apply fully connected layer
        z = z.view(-1, 256, self.base_size, self.base_size)  # Reshape to (batch_size, 256, base_size, base_size)

        x_reconstructed = self.deconvs(z)  # Apply deconvolution layers
        return x_reconstructed  # Return reconstructed output
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    # NOTE: You don't need to implement a forward function for AEModel.
    # For implementing the loss functions in train.py, call model.encoder
    # and model.decoder directly.