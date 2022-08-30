import torch
import torch.nn as nn # All NN modules are here

# leNet Architecture
"""
1x32x32 input --> (5x5), s=1, p=0 --> avg_pool, s=2, p=0 --> (5x5), s=2, p=0 --> avg pool s=2, p=0 
--> cov(5x5) to 120 channels -->  linear 120 --> linear 84  --> linear 10
# this architecture is made for Minist dataset 
# it has has use Tanh and sigmoid Activation function
# LeNet is a convolutional neural network structure proposed by Yann LeCun et al. in 1989
"""

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.Linear1 = nn.Linear(120, 84)
        self.Linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x =self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x)) # num_examples x 120 x 1 x 1 --> num_examples x 120
        x = x.reshape(x.shape[0], -1) # we would have to reshape this
        x = self.relu(self.Linear1(x))
        x = self.Linear2(x)

        return x

# let's run
x = torch.randn(64, 1, 32, 32)
model = LeNet()
print(model(x).shape)


### torch.Size([64, 10]) output 





















