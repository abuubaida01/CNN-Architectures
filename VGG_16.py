import torch
import torch.nn as nn # all NN are there


# Architecture
VGG = [64, 64, 'M' , 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 512, 'M'] # Outputs
# Flatten Linear Layer 4096, 4096, 1000 Linear Layers Output

# let's create Class
class VGG_16(nn.Module):
    def __init__(self, in_channels=3, num_channels=1000):
        super(VGG_16, self).__init__()
        self.in_channels = in_channels
        # creating Convolutional Layers
        self.conv_layer = self.create_conv_layers(VGG)
        # create fully connected part From Paper
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),# as we have 5 max pools and last output layers = 224/2**5 = 7 #224 size of Imgae
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_channels)
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        # in_channels = self.in_channels # else we can directly use self.In_channels
        layers = []
        # VGG out_channel List
        for cha in architecture:
            if type(cha) == int:
                out_channels = cha
                layers += [nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)), nn.BatchNorm2d(out_channels), nn.ReLU()]
                self.in_channels = cha
            # if it's M
            else:
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        # here, all those layers will be connected one by one
        return nn.Sequential(*layers)


## After creating VGG
model = VGG_16(in_channels=3, num_channels=1000)
x = torch.randn(1, 3, 224, 224)
print(model(x).shape)

