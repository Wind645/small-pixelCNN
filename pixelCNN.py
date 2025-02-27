import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Masked_Conv2d(nn.Conv2d):
    
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask_type
        self.mask = self.create_mask()
        
    def create_mask(self):
        #Two kinds of mask, for mask A, we can not observe the certral element,
        #B, however, we can see the central element, this makes differences in the
        #design.
        k = self.kernel_size[0] #For simplicity, I assume that the kernels are all square.
        self.mask[:, :, k // 2 + 1: , :] = 0
        self.mask[:, :, :, k // 2 + 1:] = 0
        if self.mask_type == 'A':
            self.mask[:, :, k // 2, k // 2] = 0
        # To be honest, it took me quite long to understand the core idea of the mask here,
        # but it's really some amazing ideas.    
    def forward(self, x):
        weight = self.weight.data * self.mask
        return F.conv2d(weight, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)  # prevent from modifying directly the data

class pixelCNN(nn.Module):
    def __init__(self, in_channels=3, colors=256, hidden_dim = 64, n_layers = 3):
        super().__init__()
        self.in_channels = in_channels
        self.colors = colors
        self.Conv_in = Masked_Conv2d('A', in_channels, hidden_dim, kernel_size=7, padding=3)
        self.Conv_layers = nn.Sequential([Masked_Conv2d('B', hidden_dim, hidden_dim, 
                                                        kernel_size=3, padding=1)
                                          for _ in range(n_layers)])
        self.Conv_out = nn.Conv2d(hidden_dim, colors * in_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = F.relu(self.Conv_in(x))
        x = F.relu(self.Conv_layers(x))
        x = F.relu(self.Conv_out(x))
        return x.view(x.size[0], self.in_channels, self.colors, x.size[2], x.size[3])
    
    def generate_image(self, device, img_size = 32):
        with torch.no_grad:
            img = torch.zeros((1, 3, img_size, img_size), device=device)
            for i in range(img_size):
                for j in range(img_size):
                    output = self.forward(img)
                    for c in range(3):
                        prob = F.softmax(output[0, c, :, i, j], dim=0)
                        img[0, c, i, j] = torch.multinomial(prob, 1).item()
            return img
                    
                    
        
        
            
        
        