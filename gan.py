from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from inception_score import *
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm
from torchvision.transforms import RandomHorizontalFlip, ColorJitter, RandomCrop
import torchvision.models as models
from collections import defaultdict

cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
# set manual seed to a constant get a consistent output
manualSeed = 8356
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# loading the dataset
# dataset = dset.MNIST(root="./data", download=True,
#                      transform=transforms.Compose([
#                          transforms.Resize(64),  # 调整图像大小为 64x64
#                          transforms.ToTensor(),  # 转换为张量
#                          transforms.Normalize((0.5,), (0.5,)),  # 单通道均值和标准差
#                      ]))
dataset = dset.CIFAR10(root="./data", download=True,
                       transform=transforms.Compose([
                           transforms.Resize(64),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))

# 设置通道数为 1（MNIST 是灰度图像）
# nc = 1
nc = 3

dataloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                         shuffle=True, num_workers=2)

# checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
# number of discriminator filters
ndf = 64


class LeakyReLU6(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLU6, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x >= 0, torch.clamp(x, max=6), self.negative_slope * x)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU6(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU6(inplace=True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU6(inplace=True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU6(inplace=True),

            # Adding an additional ConvTranspose2d layer
            nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU6(inplace=True),

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),  # 新增一层
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU6(inplace=True),

            # state size. (ngf//2) x 64 x 64
            nn.ConvTranspose2d(ngf // 2, nc, 3, 1, 1, bias=False),  # 输出层调整
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
# load weights to test the model
# netG.load_state_dict(torch.load('weights/netG_epoch_24.pth'))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.ReLU6(inplace=True),
            # state size. (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU6(inplace=True),
            # state size. (ndf*2) x 16 x 16

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU6(inplace=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.ReLU6(inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def smooth_labels(true_label_size, fake_label_size, smoothing=0.1):
    # 真实标签的平滑范围从1.0-smoothing到1.0
    # 假标签的平滑范围从0.0到smoothing
    true_labels = (1.0 - smoothing) + (torch.rand(true_label_size) * smoothing)
    fake_labels = torch.rand(fake_label_size) * smoothing
    return true_labels.to(device), fake_labels.to(device)


def reverse_labels(labels, percentage=0.1):
    batch_size = labels.size(0)
    num_reverse = int(batch_size * percentage)  # 计算需要翻转的标签数量
    reverse_indices = random.sample(range(batch_size), num_reverse)  # 随机选择标签进行翻转
    labels[reverse_indices] = 1 - labels[reverse_indices]  # 翻转选定的标签
    return labels


def mixed_loss(real_output, fake_output, real_label, fake_label, epoch):
    # 计算 BCE 损失
    criterion = nn.BCELoss()
    bce_loss_real = criterion(real_output, real_label)
    bce_loss_fake = criterion(fake_output, fake_label)

    # 计算 Hinge 损失
    hinge_loss_real = torch.mean(F.relu(1 - real_output))
    hinge_loss_fake = torch.mean(F.relu(1 + fake_output))

    # 混合损失（可以调整权重）
    hinge_weight = min(epoch / 50, 0.6)  # 动态权重 # 动态增加 Hinge 损失的权重
    bce_weight = 1 - hinge_weight  # BCE 损失的权重动态减小
    total_loss = bce_weight * (bce_loss_real + bce_loss_fake) + hinge_weight * (hinge_loss_real + hinge_loss_fake)
    return total_loss


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:26].eval()  # 例如使用VGG的前23层
        self.vgg = vgg
        self.vgg = self.vgg.to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, fake, real):
        # fake = fake.repeat(1, 3, 1, 1)  # (N, 1, H, W) -> (N, 3, H, W)
        # real = real.repeat(1, 3, 1, 1)  # (N, 1, H, W) -> (N, 3, H, W)
        fake_vgg = self.vgg(fake)
        real_vgg = self.vgg(real)
        loss = F.mse_loss(fake_vgg, real_vgg)
        return loss


perceptual_loss = PerceptualLoss()
lambda_perceptual = 0.01

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
# load weights to test the model
# netD.load_state_dict(torch.load('weights/netD_epoch_24.pth'))
print(netD)

criterion = nn.BCELoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0015, betas=(0.5, 0.999))
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.5)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.5)

fixed_noise = torch.randn(100, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# 初始化翻转比例
increment_per_epoch = 0.01  # 每轮增加的翻转比例
current_reverse_percentage = 0.0  # 当前翻转比例
reverse_start_epoch = 40  # 从第50轮开始翻转标签
reverse_interval = 5  # 每5轮进行一次翻转
real_images_dir = './cifar/real_images'
fake_images_dir = './cifar/fake_images'
log_file_path = './cifar/training_log.txt'
os.makedirs(real_images_dir, exist_ok=True)
os.makedirs(fake_images_dir, exist_ok=True)
# 用于记录每个类别保存的图像数量
class_image_count = defaultdict(int)
max_images_per_class = 100  # 每个类别保存 100 张图像

# 遍历 CIFAR-10 数据集并保存图像
image_index = 0  # 用于保存文件名的计数

# 在训练开始之前，创建并写入文件头
with open(log_file_path, 'w') as log_file:
    log_file.write('Epoch\tLoss_D\tLoss_G\tD(x)\tD(G(z))\tFID\n')

niter = 150
g_loss = []
d_loss = []
num_images = 1000
is_scores = []
best_is_mean = float('-inf')

for epoch in range(niter):

    fake_images = []
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        # label = torch.full((batch_size,), real_label, device=device)
        real_labels_smooth, fake_labels_smooth = smooth_labels(batch_size, batch_size)

        output = netD(real_cpu)
        # errD_real = criterion(output, label)
        # errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        # label.fill_(fake_label)
        output_fake = netD(fake.detach())
        # errD_fake = criterion(output, label)
        # errD_fake.backward()
        D_G_z1 = output_fake.mean().item()
        errD = mixed_loss(output, output_fake, real_labels_smooth, fake_labels_smooth, epoch)
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        output = netD(fake)
        D_G_z2 = output.mean().item()
        bce_loss = criterion(output, real_labels_smooth)  # 使用二元交叉熵损失
        percep_loss = perceptual_loss(fake, real_cpu)
        lambda_percep = min(epoch / 50, 0.6)  # 动态权重
        normalized_percep_loss = percep_loss / (percep_loss.detach().mean() + 1e-8)
        errG = (1 - lambda_percep) * bce_loss + lambda_percep * normalized_percep_loss
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
            epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # 记录日志
    for _ in range(8):  # 8次循环，每次生成128张图像
        noise = torch.randn(128, nz, 1, 1, device=device)
        fake_batch = netG(noise)
        fake_images.append(fake_batch)

        # 将生成的图像转换为适合 Inception Score 计算的格式
    fake_images = torch.cat(fake_images, dim=0)[:1000].detach().cpu()  # 只取前1000张图像

    print('Calculating Inception Score for epoch %d...' % (epoch))
    # 分批计算IS
    batch_size = 32
    is_mean, is_std = inception_score(fake_images, cuda=True, batch_size=32, resize=True,
                                      splits=10)  # 增加splits以提高评估的稳定性
    is_scores.append((is_mean, is_std))
    print('Epoch %d - Inception Score: Mean: %.4f, Std: %.4f' % (epoch, is_mean, is_std))
    with open(log_file_path, 'a') as log_file:
        log_file.write(
            f'{epoch}\t{errD.item():.4f}\t{errG.item():.4f}\t{D_x:.4f}\t{D_G_z1:.4f}\t{is_mean:.4f}\t{is_std:.4f}\n')
    #Check pointing for every epoch
    torch.save(netG.state_dict(), 'weights3/netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), 'weights3/netD_epoch_%d.pth' % (epoch))
