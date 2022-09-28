import torch 
import torch.nn as nn
'''
Function double_conv is created for completing this requirement
    It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels


self.up_trans_1  = Tranpose Convolution or Up convolution & Function Crop_img
    Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU.

'''

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv



def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2] # [2] to get the height
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size # tensor size will always be larger
    delta = delta//2   # here we will get, for suppose 64-56 = 8 // 2  = 4
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]  # here, we are cropping  


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2) # acc to article
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512) 
        self.down_conv_5 = double_conv(512,1024)
        
    
        # for Decoding 
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024, # down_conv_5 output_channels
            out_channels=512, 
            kernel_size=2, 
            stride=2
        )
        self.up_conv_1 = double_conv(1024, 512)                       
        
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,  
            out_channels=256, 
            kernel_size=2, 
            stride=2
        )
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,  
            out_channels=128, 
            kernel_size=2, 
            stride=2
        )
        self.up_conv_3 = double_conv(256, 128)
        
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128, 
            out_channels=64, 
            kernel_size=2, 
            stride=2
        )
        self.up_conv_4 = double_conv(128, 64)

        # At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes.
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2, # if we have multiple no of Segments so we can increase no of chn
            kernel_size=1
        )
        

    def forward(self, image):
        # bc, c, h, w 
        # Encoding part
        x1 = self.down_conv_1(image)   # here x1 will concatenate with last x``
        # print(x1.size()) 
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2) # this will concatenate as well
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4) # this will concatenate as well
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6) # this will concatenate as well
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        # print(x9.size())


        # Decoding Part
        x = self.up_trans_1(x9) # here from 28x28 we increased size to 56x56
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))  #512 + 512 = 1024, 56,56
        
        # here we will concate x5 with x
        x = self.up_trans_2(x) 
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1)) 
        
        # here we will concate x3 with x
        x = self.up_trans_3(x) 
        y = crop_img(x3, x)
        x= self.up_conv_3(torch.cat([x, y], 1)) 
        
        # here we will concate x1 with x
        x = self.up_trans_4(x) 
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1)) 
        
        x = self.out(x)
        print(x.size())
        return x
        # print(x.size())  # here we are getting 1, 64, 388, 388 
        # print(x7.size())
        # print(x.size())
        # print(y.size())

if __name__ == "__main__":
    # in Paper image size is 1, 1, 572 x 572
    image = torch.rand((1,1,572, 572))
    model = UNet()
    # print(model(image))