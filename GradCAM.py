import cv2
from copy import deepcopy
from torch.nn import functional as F
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class CamGenerator():
    """
    CAM: Class Activation Map
    Generate a heatmap on top of the existing image
    To help visualize the convolutional nueral network.
    For more details please see Zhou et. al (2016)
    """
    def __init__(self, model, name_of_the_last_conv_layer):

        self.model = model
        self.name_of_the_last_conv_layer = name_of_the_last_conv_layer

        self.feature_maps = None
        self.gradient = None
        self._register()

    def _register(self):

        def forward_recorder(module, input_, output):
            self.feature_maps = output.data.cpu()

        def backward_recorder(module, grad_in, grad_out):
            self.gradient = grad_out[0].data.cpu()

        for i, j in self.model.named_modules():
            if i ==  self.name_of_the_last_conv_layer:
                try:
                    self.length_of_filter = j.out_channels
                    self.size_of_kernel = j.kernel_size
                except AttributeError:
                    print("Target layer is not conv2d layer")

                j.register_forward_hook(forward_recorder)
                j.register_backward_hook(backward_recorder)
                break
        else:
            print("Cannot find the given layer, maybe try another name.")
            return None

    def load_data(self, data_loader):
        try:
            self.image_tensor, self.label = next(iter(data_loader))
        except:
            print('Need to pass a PyTorch DataLoader object as positional argument')
            return None

        assert len(self.label) == 1, 'Set the batch size of dataloader to 1'
        self.image_numpy = self.image_tensor[0].numpy().transpose(1,2,0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        self.image_numpy = std * self.image_numpy + mean

    def forward_pass(self):
        self.output = self.model(self.image_tensor)
        self.size_of_feature_maps = self.feature_maps.size()[2:]

    def backward_pass(self):
        _, self.argmax = torch.max(self.output,1)
        one_hot = torch.FloatTensor(1, self.output.size()[-1]).zero_()
        one_hot[0][self.argmax] = 1.0
        self.output.backward(one_hot, retain_graph = True)

    def find_gradient_CAM(self, epsilon = 1e-5):
        assert self.feature_maps  is not None, "Feature map is empty, need to call .forward_pass() first"
        assert self.gradient is not None, "Gradient is empty, need to call .backward_pass() first"

        self.gradient = self.gradient/(torch.sqrt(torch.mean(torch.pow(self.gradient, 2))) + epsilon) #normalize gradient
        average_grad =  nn.AvgPool2d(self.gradient.size()[2:])(self.gradient) #get mean gradient for each layer
        assert self.length_of_filter == average_grad.size()[1], "Number of filter does not match gradient dimension"
        average_grad.resize_(average_grad.size()[1]) # lots of squeeze

        if len(self.feature_maps.size()) == 4:
            self.feature_maps = self.feature_maps.squeeze(0)

        gradient_CAM = torch.FloatTensor(self.size_of_feature_maps).zero_()
        for feature_map, weight in zip(self.feature_maps, average_grad):
                gradient_CAM = gradient_CAM + feature_map * weight.data
        self.gradient_CAM = gradient_CAM
        return self.gradient_CAM

    def _generate_heat_map(self):

        try:
            gradient_CAM = deepcopy(self.gradient_CAM)
            gradient_CAM = F.relu(gradient_CAM)

            gradient_CAM -= gradient_CAM.min(); gradient_CAM /= gradient_CAM.max()
            gradient_CAM = cv2.resize(gradient_CAM.numpy(), (224, 224))
            gradient_CAM = cv2.applyColorMap(np.uint8(gradient_CAM * 255.0), cv2.COLORMAP_JET)

            gradient_CAM = gradient_CAM.astype(np.float) + self.image_numpy*255
            gradient_CAM = gradient_CAM / gradient_CAM.max()*255
            return gradient_CAM

        except AttributeError:
            print('Need to first call find_gradient_CAM()')
            return None

    def show(self):
        try:
            guessed_name = 'Bee' if self.argmax == 1 else 'Ant'
            true_name = 'Bee' if self.label == 1 else 'Ant'

            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(self.image_numpy)

            plt.title('Reality: {}'.format(true_name))
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(cam_generator._generate_heat_map()/255)
            plt.title('Guess: {}'.format(guessed_name))
            plt.show()

        except AttributeError:
            print('Need to first call find_gradient_CAM()')
            return None

if __name__ == '__main__':

    data_dir = '/Users/haigangliu/ImageData/ImageNet/'
    import os
    from torchvision import datasets, models, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
    dataloaders = DataLoader(image_datasets,
                                 batch_size= 1,
                                 num_workers=4,
                                 pin_memory = True,
                                 shuffle = True
                                )
    model = torch.load('/Users/haigangliu/Desktop/bees.pth', map_location='cpu')

    cam_generator = CamGenerator(model, 'layer4.1.conv2')
    cam_generator.load_data(dataloaders)
    cam_generator.forward_pass()
    cam_generator.backward_pass()
    cam_generator.find_gradient_CAM()
    cam_generator.show()
