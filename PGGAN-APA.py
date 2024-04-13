#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from comet_ml import Experiment
#experiment = Experiment()


# In[ ]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image, ImageFile
from pickle import load, dump
import random
import cv2
import time
import argparse
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


class WeightScaledConv2d(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=0, spectral_norm=False):
        super().__init__()
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding))
        else:
            self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding)
        nn.init.kaiming_normal_(self.conv.weight)
        #self.scale = torch.sqrt(2 / (kernel_size ** 2 * output_nc))
        self.scale = nn.Parameter(torch.sqrt(torch.mean(self.conv.weight.data ** 2)))
        self.conv.weight.data /= self.scale

    def forward(self, x):
        x = x * self.scale
        x = self.conv(x)
        return x


# In[ ]:


class PixelwiseNormalization(nn.Module):
    def pixel_norm(self, x):
        eps = 1e-8
        return x * torch.rsqrt(torch.mean(x * x, 1, keepdim=True) + eps)
    
    def forward(self, x):
        return self.pixel_norm(x)


# In[ ]:


class GeneratorBlock(nn.Module):
    def __init__(self, input_nc, output_nc, num_channels, first=False):
        super().__init__()
        
        if first:
            self.model = nn.Sequential(
                WeightScaledConv2d(input_nc, output_nc, kernel_size=4, stride=1, padding=3),
                PixelwiseNormalization(),
                nn.LeakyReLU(0.2, inplace=True),
                WeightScaledConv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1),
                PixelwiseNormalization(),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.model = nn.Sequential(
                WeightScaledConv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1),
                PixelwiseNormalization(),
                nn.LeakyReLU(0.2, inplace=True),
                WeightScaledConv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1),
                PixelwiseNormalization(),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.toRGB = WeightScaledConv2d(output_nc, num_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, image, last=False):
        image = self.model(image)
        if last:
            image = self.toRGB(image)
        return image


# In[ ]:


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_nc, output_nc, num_channels, last=False):
        super().__init__()
        
        self.fromRGB = WeightScaledConv2d(num_channels, input_nc, kernel_size=3, stride=1, padding=1)
        
        self.last = last
        if not last:
            self.model = nn.Sequential(
                WeightScaledConv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                WeightScaledConv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.model = nn.Sequential(
                WeightScaledConv2d(input_nc + 1, output_nc, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                WeightScaledConv2d(output_nc, output_nc, kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(0.2, inplace=True)
            )
    
    def minibatch_standard_deviation(self, x):
        eps = 1e-8
        return torch.cat([x, torch.sqrt(((x - x.mean())**2).mean() + eps).expand(x.shape[0], 1, *x.shape[2:])], dim=1)
    
    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        if self.last:
            x = self.minibatch_standard_deviation(x)
        x = self.model(x)
        return x


# In[ ]:


class Generator(nn.Module):
    def __init__(self, num_depth, num_channels, num_fmap):
        super().__init__()
        
        self.blocks = nn.ModuleList(
            [GeneratorBlock(num_fmap(0), num_fmap(1), num_channels, first=True)]
            + [GeneratorBlock(num_fmap(i-1), num_fmap(i), num_channels) for i in range(2, num_depth + 1)]
        )
        
        self.depth = 0
        self.alpha = 1.0
    
    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        rgb = x = self.blocks[0](x, self.depth == 0)
        
        if self.depth > 0:
            for i in range(self.depth - 1):
                x = F.interpolate(x, scale_factor=2) # upsample
                x = self.blocks[i+1](x)
            
            x = F.interpolate(x, scale_factor=2) # upsample
            rgb = self.blocks[self.depth](x, last=True)
            
            if self.alpha < 1.0:
                prev_rgb = self.blocks[self.depth - 1].toRGB(x)
                rgb = (1 - self.alpha) * prev_rgb + self.alpha * rgb
        
        return rgb


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, num_depth, num_channels, num_fmap):
        super().__init__()
        
        self.blocks = nn.ModuleList(
            [DiscriminatorBlock(num_fmap(i), num_fmap(i-1), num_channels) for i in range(num_depth, 1, -1)]
            + [DiscriminatorBlock(num_fmap(1), num_fmap(0), num_channels, last=True)]
        )
        
        self.linear = nn.Linear(num_fmap(0), 1)
        
        self.depth = 0
        self.alpha = 1.0
        
    def forward(self, x):
        x_high = x
        
        h = self.blocks[-(self.depth + 1)](x_high, first=True)
        
        if self.depth > 0:
            h = F.avg_pool2d(h, 2) # downsample
            
            if self.alpha < 1.0:
                x_low = F.avg_pool2d(x_high, 2) # downsample
                prev = self.blocks[-self.depth].fromRGB(x_low)
                h = self.alpha * h + (1 - self.alpha) * prev
                
        for i in range(self.depth, 0, -1):
            h = self.blocks[-i](h)
            if i > 1:
                h = F.avg_pool2d(h, 2)  # downsample
        
        out = self.linear(h.squeeze(-1).squeeze(-1))
        
        return out


# In[ ]:


class RandomErasing:
    def __init__(self, p=0.5, erase_low=0.02, erase_high=0.33, aspect_rl=0.3, aspect_rh=3.3):
        self.p = p
        self.erase_low = erase_low
        self.erase_high = erase_high
        self.aspect_rl = aspect_rl
        self.aspect_rh = aspect_rh
    
    def __call__(self, image):
        if np.random.rand() <= self.p:
            c, h, w = image.shape
            
            mask_area = np.random.uniform(self.erase_low, self.erase_high) * (h * w)
            mask_aspect_ratio = np.random.uniform(self.aspect_rl, self.aspect_rh)
            mask_w = int(np.sqrt(mask_area / mask_aspect_ratio))
            mask_h = int(np.sqrt(mask_area * mask_aspect_ratio))
            
            mask = torch.Tensor(np.random.rand(c, mask_h, mask_w) * 255)
            
            left = np.random.randint(0, w)
            top = np.random.randint(0, h)
            right = left + mask_w
            bottom = top + mask_h
            
            if right <= w and bottom <= h:
                image[:, top:bottom, left:right] = mask
        
        return image


# In[ ]:


class Util:
    @staticmethod
    def loadImages(batch_size, folder_path, size):
        imgs = ImageFolder(folder_path, transform=transforms.Compose([
            transforms.Resize(int(size)),
            transforms.RandomCrop(size),
            transforms.ToTensor()
        ]))
        return DataLoader(imgs, batch_size=batch_size, shuffle=True, drop_last=True)
    
    @staticmethod
    def augment(images):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            RandomErasing()
        ])
        device = images.device
        return torch.cat([transform(img).unsqueeze(0) for img in images.cpu()], 0).to(device)
    
    @staticmethod
    def showImages(dataloader):
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        
        PIL = transforms.ToPILImage()
        ToTensor = transforms.ToTensor()

        for images in dataloader:
            for image in images[0]:
                img = PIL(image)
                fig = plt.figure(dpi=200)
                ax = fig.add_subplot(1, 1, 1) # (row, col, num)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(img)
                #plt.gray()
                plt.show()


# In[ ]:


class Solver:
    def __init__(self, args):
        use_cuda = torch.cuda.is_available() if not args.cpu else False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        print(f'Use Device: {self.device}')
        
        def num_fmap(stage):
            base_size = self.args.image_size
            fmap_base = base_size * 4
            fmap_max = base_size // 2
            fmap_decay = 1.0
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        
        self.args = args
        self.num_channels = 3
        self.feed_dim = num_fmap(0)
        self.max_depth = int(np.log2(self.args.image_size)) - 1
        self.depth = 0
        
        self.pseudo_aug = 0.0
        self.epoch = 0
        self.num_train = 0
        
        self.netG = Generator(self.max_depth, self.num_channels, num_fmap).to(self.device)
        self.netD = Discriminator(self.max_depth, self.num_channels, num_fmap).to(self.device)
        
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=self.args.lr, betas=(0, 0.9))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.args.lr * self.args.mul_lr_dis, betas=(0, 0.9))
        #self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=4, eta_min=self.args.lr/4)
        #self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=4, eta_min=(self.args.lr * self.args.mul_lr_dis)/4)
            
    def load_dataset(self):
        self.batch_size  = self.args.batch_size_base * (self.max_depth - self.depth)
        image_size = self.args.image_size / 2 ** (self.max_depth - self.depth - 1)
        self.dataloader = Util.loadImages(self.batch_size, self.args.image_dir, image_size)
        self.max_iters = len(iter(self.dataloader))
    
    def save_state(self, epoch):
        self.netG.cpu(), self.netD.cpu()
        torch.save(self.netG.state_dict(), os.path.join(self.args.weight_dir, f'weight_G.{epoch}.pth'))
        torch.save(self.netD.state_dict(), os.path.join(self.args.weight_dir, f'weight_D.{epoch}.pth'))
        self.netG.to(self.device), self.netD.to(self.device)
    
    def load_state(self):
        if (os.path.exists('weight_G.pth') and os.path.exists('weight_D.pth')):
            self.netG.load_state_dict(torch.load('weight_G.pth', map_location=self.device))
            self.netD.load_state_dict(torch.load('weight_D.pth', map_location=self.device))
            self.state_loaded = True
            print('Loaded network state.')
    
    def save_resume(self):
        with open(os.path.join('.', f'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    def load_resume(self):
        if os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                return load(f)
        else:
            return self
    
    def trainGAN(self, epoch, iters, max_iters, real_img, a=0, b=1, c=1):
        ### Train with LSGAN.
        ### for example, (a, b, c) = 0, 1, 1 or (a, b, c) = -1, 1, 0
        
        random_data = torch.randn(real_img.size(0), self.feed_dim).to(self.device)
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #
        
        # Compute loss with real images.
        real_src_score = self.netD(real_img)
        real_src_loss = torch.sum((real_src_score - b) ** 2)
        
        # Compute loss with fake images.
        fake_img = self.netG(random_data)
        fake_src_score = self.netD(fake_img)
        
        p = random.uniform(0, 1)
        if 1 - self.pseudo_aug < p:
            fake_src_loss = torch.sum((fake_src_score - b) ** 2) # Pseudo: fake is real.
        else:
            fake_src_loss = torch.sum((fake_src_score - a) ** 2)
        
        # Update Probability Augmentation.
        lz = (torch.sign(torch.logit(real_src_score)).mean()
              - torch.sign(torch.logit(fake_src_score)).mean()) / 2
        if lz > self.args.aug_threshold:
            self.pseudo_aug += self.args.aug_increment
        else:
            self.pseudo_aug -= self.args.aug_increment
        self.pseudo_aug = min(1, max(0, self.pseudo_aug))
        
        # Backward and optimize.
        d_loss = 0.5 * (real_src_loss + fake_src_loss) / self.batch_size
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
              
        # Logging.
        loss = {}
        loss['D/loss'] = d_loss.item()
        loss['Augment/prob'] = self.pseudo_aug
        
        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        # Compute loss with reconstruction loss
        fake_img = self.netG(random_data)
        fake_src_score = self.netD(fake_img)
        fake_src_loss = torch.sum((fake_src_score - c) ** 2)
        
        # Backward and optimize.
        g_loss = 0.5 * fake_src_loss / self.batch_size
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # Logging.
        loss['G/loss'] = g_loss.item()
        
        # Save
        if iters == max_iters:
            self.save_state(epoch)
            img_name = str(epoch) + '_' + str(iters) + '.png'
            img_path = os.path.join(self.args.result_dir, img_name)
            save_image(fake_img, img_path)
        
        return loss
    
    def train(self, resume=True):
        hyper_params = {}
        hyper_params['Image Dir'] = self.args.image_dir
        hyper_params['Result Dir'] = self.args.result_dir
        hyper_params['Weight Dir'] = self.args.weight_dir
        hyper_params['Image Size'] = self.args.image_size
        hyper_params['Learning Rate'] = self.args.lr
        hyper_params["Mul Discriminator's LR"] = self.args.mul_lr_dis
        hyper_params['Batch Size Base'] = self.args.batch_size_base
        hyper_params['Num Train Base'] = self.args.num_train_base
        hyper_params['Probability Aug-Threshold'] = self.args.aug_threshold
        hyper_params['Probability Aug-Increment'] = self.args.aug_increment
        hyper_params['Max Depth'] = self.max_depth
        hyper_params['Start Depth'] = self.depth + 1
        
        for key in hyper_params.keys():
            print(f'{key}: {hyper_params[key]}')
        #experiment.log_parameters(hyper_params)
        
        self.netG.train()
        self.netD.train()
        self.load_dataset()
        
        while True:
            #max_train = self.num_train_base
            max_train = int(self.args.num_train_base // np.log2(self.max_depth - self.depth + 1))
            self.num_train += 1
            self.epoch += 1
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            
            for iters, (data, _) in enumerate(tqdm(self.dataloader)):
                iters += 1
                
                alpha = self.num_train / max_train
                alpha = min(1.0, alpha * 2)
                self.netG.alpha = self.netD.alpha = alpha
                
                data = data.to(self.device)
                
                loss = self.trainGAN(self.epoch, iters, self.max_iters, data)
                
                epoch_loss_D += loss['D/loss']
                epoch_loss_G += loss['G/loss']
                #experiment.log_metrics(loss)
            
            epoch_loss = epoch_loss_G + epoch_loss_D
            
            print(f'Epoch[{self.epoch}] Depth[{self.depth+1}/{self.max_depth}]'
                  + f' DepthTrain[{self.num_train}/{max_train}] BatchSize[{self.batch_size}]'
                  #+ f' LR[G({self.scheduler_G.get_last_lr()[0]:.5f}) D({self.scheduler_D.get_last_lr()[0]:.5f})]'
                  + f' Loss[G({epoch_loss_G}) + D({epoch_loss_D}) = {epoch_loss}]')
                
            #self.scheduler_G.step()
            #self.scheduler_D.step()
            
            if self.num_train >= max_train:
                if self.depth+1 < self.max_depth:
                    self.depth += 1
                    self.netG.depth = self.netD.depth = self.depth
                    self.load_dataset()  # Change batch-size and image-size.
                    self.num_train = 0
                else:
                    break
                    
            if not self.args.noresume:
                self.save_resume()
    
    def generate(self, num=100):
        self.netG.eval()
        
        for _ in range(num):
            random_data = [torch.randn(1, self.netG.input_size).to(self.device)]
            fake_img = self.netG(random_data)[0].cpu().data[0]
            save_image(fake_img, os.path.join(self.args.result_dir, f'generated_{time.time()}.png'))
            
        print('New picture was generated.')
    
    def showImages(self):
        depth = self.depth
        self.depth = self.max_depth - 1
        self.load_dataset()
        Util.showImages(self.dataloader)
        self.depth = depth
        self.load_dataset()


# In[ ]:


def main(args):
    solver = Solver(args)
    solver.load_state()
    
    if not args.noresume:
        solver = solver.load_resume()
    
    if args.start_depth != 1:
        solver.netG.depth = solver.netD.depth = solver.depth = args.start_depth - 1
    
    if args.generate > 0:
        solver.generate(args.generate)
        return
    
    #solver.showImages()
    solver.train()
    
    #experiment.end()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mul_lr_dis', type=float, default=4)
    parser.add_argument('--batch_size_base', type=int, default=16)
    parser.add_argument('--num_train_base', type=int, default=128)
    parser.add_argument('--start_depth', type=int, default=1)
    parser.add_argument('--aug_threshold', type=float, default=0.6)
    parser.add_argument('--aug_increment', type=float, default=0.01)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--noresume', action='store_true')
    
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

