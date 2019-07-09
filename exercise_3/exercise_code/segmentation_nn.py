"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        self.features = models.vgg16(pretrained=True).features
        
        fc1 = nn.Conv2d(512, 2048, 7)
        fc2 = nn.Conv2d(2048, 4096, 1)
        fc3 = nn.Conv2d(4096, num_classes, 1)
        self.fcn = nn.Sequential(
                                fc1,
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                fc2,
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                fc3
                                )
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        x_size = x.size()
        x = self.features(x)
        x = self.fcn(x)
        x_z = nn.Upsample(x_size[2:], mode='bilinear')
        x = x_z(x)
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

