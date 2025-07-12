import copy
from torch.utils.data import ConcatDataset, DataLoader
from torch import optim
import torch.nn.init as init
import time
from torch.backends import cudnn
from torchvision.transforms import transforms
import loan_train
import image_train
import config
import random
import torchvision.models as models
from gan_model import Generator, Discriminator
from inception_score import *


def train(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys, client_settings, combined_scores,
          sorted_clients_index):
    epochs_submit_update_dict = {}
    epochs_submit_update_dict_fool = {}
    num_samples_dict = {}
    if helper.params['type'] == config.TYPE_LOAN:
        epochs_submit_update_dict, num_samples_dict = loan_train.LoanTrain(helper, start_epoch, local_model,
                                                                           target_model, is_poison, agent_name_keys)
    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_TINYIMAGENET:
        epochs_submit_update_dict, epochs_submit_update_dict_fool, num_samples_dict = image_train.ImageTrain(helper, start_epoch, local_model,
                                                                             target_model, is_poison, agent_name_keys,
                                                                             client_settings, combined_scores,
                                                                             sorted_clients_index)
    return epochs_submit_update_dict,epochs_submit_update_dict_fool, num_samples_dict


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def smooth_labels(true_label_size, fake_label_size, smoothing=0.1):
    # 真实标签的平滑范围从1.0-smoothing到1.0
    # 假标签的平滑范围从0.0到smoothing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        vgg = models.vgg16(pretrained=True).features[:8].eval()  # 例如使用VGG的前8层
        self.vgg = vgg
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vgg = self.vgg.to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    # def forward(self, fake, real):
    #     fake = fake.repeat(1, 3, 1, 1)  # (N, 1, H, W) -> (N, 3, H, W)
    #     real = real.repeat(1, 3, 1, 1)  # (N, 1, H, W) -> (N, 3, H, W)
    #     fake_vgg = self.vgg(fake)
    #     real_vgg = self.vgg(real)
    #     loss = F.mse_loss(fake_vgg, real_vgg)
    #     return loss
    def forward(self, fake, real):
        fake_vgg = self.vgg(fake)
        real_vgg = self.vgg(real)
        loss = F.mse_loss(fake_vgg, real_vgg)
        return loss

def train_dcgan(helper, device, client_index, num_epochs=150):#mnist 100轮，cifar 150轮
    cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    manualSeed = 8356
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    ngpu = 1
    nz = 100
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

    model_dir = f"models/dcgan_model/{client_index}/cifar"
    ensure_dir(model_dir)
    perceptual_loss = PerceptualLoss()
    criterion = nn.BCELoss()

    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0015, betas=(0.5, 0.999))
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.5)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.5)
    real_label = 1
    fake_label = 0

    adversary_list = helper.params['adversary_list']

    # 提取 adversary_list 中对应的客户端数据加载器
    datasets_to_merge = []
    for adv_client_index in adversary_list:
        _, data_loader = helper.gan_train_data[adv_client_index]  # 获取指定客户端的数据加载器
        datasets_to_merge.append(data_loader.dataset)  # 提取数据加载器中的数据集

    # 合并所有数据集
    merged_dataset = ConcatDataset(datasets_to_merge)

    # 创建新的数据加载器
    merged_data_loader = DataLoader(merged_dataset, batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        # _, data_iterator = helper.gan_train_data[client_index]

        fake_images = []
        for i, (data, labels) in enumerate(merged_data_loader):

            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            real_cpu = data.to(device)  # 使用整个批次的数据
            batch_size = real_cpu.size(0)
            real_labels_smooth, fake_labels_smooth = smooth_labels(batch_size, batch_size)
            output = netD(real_cpu)  # 传递四维张量
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            D_G_z1 = output_fake.mean().item()
            errD = mixed_loss(output, output_fake, real_labels_smooth, fake_labels_smooth, epoch)
            errD.backward()
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            output = netD(fake)
            D_G_z2 = output.mean().item()
            bce_loss = criterion(output, real_labels_smooth)
            percep_loss = perceptual_loss(fake, real_cpu) # 注意：real_cpu 仍然是四维张量
            lambda_percep = min(epoch / 50, 0.6)
            normalized_percep_loss = percep_loss / (percep_loss.detach().mean() + 1e-8)
            errG = (1 - lambda_percep) * bce_loss + lambda_percep * normalized_percep_loss
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
                epoch, num_epochs, i, len(merged_data_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        schedulerD.step()
        schedulerG.step()
        for _ in range(8):  # 8次循环，每次生成128张图像
            noise = torch.randn(128, 100, 1, 1, device=device)
            fake_batch = netG(noise)
            fake_images.append(fake_batch)

            # 将生成的图像转换为适合 Inception Score 计算的格式
        fake_images = torch.cat(fake_images, dim=0)[:1000].detach().cpu()  # 只取前1000张图像

        print('Calculating Inception Score for epoch %d...' % (epoch))
        # 分批计算IS
        batch_size = 32
        is_mean, is_std = inception_score(fake_images, cuda=True, batch_size=32, resize=True,
                                          splits=10)  # 增加splits以提高评估的稳定性
        print('Epoch %d - Inception Score: Mean: %.4f, Std: %.4f' % (epoch, is_mean, is_std))
        if epoch == num_epochs - 1:
            torch.save(netG.state_dict(), f'{model_dir}/netG_final.pth')
            torch.save(netD.state_dict(), f'{model_dir}/netD_final.pth')


# def train_dcgan(helper, device, client_index, num_batches=70380):
#     cudnn.benchmark = True
#     torch.autograd.set_detect_anomaly(True)
#     manualSeed = 8356
#     print("Random Seed: ", manualSeed)
#     random.seed(manualSeed)
#     torch.manual_seed(manualSeed)
#
#     ngpu = 1
#     nz = 100
#     netG = Generator(ngpu).to(device)
#     netG.apply(weights_init)
#     netD = Discriminator(ngpu).to(device)
#     netD.apply(weights_init)
#
#     model_dir = f"models/dcgan_model/{client_index}/cifar"
#     ensure_dir(model_dir)
#     perceptual_loss = PerceptualLoss()
#     criterion = nn.BCELoss()
#
#     optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizerG = optim.Adam(netG.parameters(), lr=0.0015, betas=(0.5, 0.999))
#     schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.5)
#     schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.5)
#     real_label = 1
#     fake_label = 0
#
#     batch_counter = 0  # 记录总批次数
#
#     # # 获取数据加载器并创建无限循环的迭代器
#     # _, dataloader = helper.gan_train_data[client_index]
#     # data_iterator = iter(itertools.cycle(dataloader))  # 无限循环的数据迭代器
#
#     def get_new_dataloader():
#         """动态获取新的 DataLoader"""
#         _, dataloader = helper.gan_train_data[client_index]
#         return iter(dataloader)
#
#     data_iterator = get_new_dataloader()  # 初始加载 DataLoader
#
#     while batch_counter < num_batches:
#         # data, labels = next(data_iterator)
#         try:
#             # 尝试从当前的迭代器中获取数据
#             data, labels = next(data_iterator)
#         except StopIteration:
#             # 如果当前 DataLoader 数据用尽，重新加载一个新的 DataLoader
#             # print(f"Data exhausted. Reloading DataLoader at batch {batch_counter}...")
#             data_iterator = get_new_dataloader()
#             data, labels = next(data_iterator)  # 获取新 DataLoader 的第一批数据
#
#         batch_counter += 1  # 每训练一个批次，计数器加 1
#
#         ############################
#         # (1) Update D network
#         ###########################
#         netD.zero_grad()
#         real_cpu = data.to(device)  # 使用整个批次的数据
#         batch_size = real_cpu.size(0)
#         real_labels_smooth, fake_labels_smooth = smooth_labels(batch_size, batch_size)
#         output = netD(real_cpu)  # 传递四维张量
#         D_x = output.mean().item()
#
#         # train with fake
#         noise = torch.randn(batch_size, nz, 1, 1, device=device)
#         fake = netG(noise)
#         output_fake = netD(fake.detach())
#         D_G_z1 = output_fake.mean().item()
#         errD = mixed_loss(output, output_fake, real_labels_smooth, fake_labels_smooth, batch_counter)
#         errD.backward()
#         optimizerD.step()
#
#         ############################
#         # (2) Update G network
#         ###########################
#         netG.zero_grad()
#         output = netD(fake)
#         D_G_z2 = output.mean().item()
#         bce_loss = criterion(output, real_labels_smooth)
#         percep_loss = perceptual_loss(fake, real_cpu)  # 注意：real_cpu 仍然是四维张量
#         lambda_percep = min(batch_counter / 39100, 0.6)  # 随批次增长调整权重
#         normalized_percep_loss = percep_loss / (percep_loss.detach().mean() + 1e-8)
#         errG = (1 - lambda_percep) * bce_loss + lambda_percep * normalized_percep_loss
#         errG.backward()
#         optimizerG.step()
#
#         if batch_counter % 782 == 0:  # 这里可以调整间隔，比如 100 个批次
#             schedulerD.step()
#             schedulerG.step()
#         # 打印日志
#         print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
#             batch_counter, num_batches, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
#         # 如果达到目标批次数，停止训练
#         if batch_counter >= num_batches:
#             break
#
#     # 最后保存最终模型
#     torch.save(netG.state_dict(), f'{model_dir}/netG_final.pth')
#     torch.save(netD.state_dict(), f'{model_dir}/netD_final.pth')
#     print("Training finished and final model saved.")

# def train_dcgan(helper, device, client_index, num_epochs=150):
# def train_dcgan(helper, device, client_index, netG, netD, num_epochs=150):
#
#     global_netG_params = copy.deepcopy(netG.state_dict())
#     global_netD_params = copy.deepcopy(netD.state_dict())
#     cudnn.benchmark = True
#     torch.autograd.set_detect_anomaly(True)
#     manualSeed = 8356
#     print("Random Seed: ", manualSeed)
#     random.seed(manualSeed)
#     torch.manual_seed(manualSeed)
#     nz = 100
#     model_dir = f"models/dcgan_model/{client_index}/cifar"
#     ensure_dir(model_dir)
#     perceptual_loss = PerceptualLoss()
#     criterion = nn.BCELoss()
#
#     optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizerG = optim.Adam(netG.parameters(), lr=0.0015, betas=(0.5, 0.999))
#     schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.5)
#     schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.5)
#     real_label = 1
#     fake_label = 0
#
#     for epoch in range(num_epochs):
#         _, data_iterator = helper.gan_train_data[client_index]
#         for i, (data, labels) in enumerate(data_iterator):
#
#             ############################
#             # (1) Update D network
#             ###########################
#             netD.zero_grad()
#             real_cpu = data.to(device)  # 使用整个批次的数据
#             batch_size = real_cpu.size(0)
#             real_labels_smooth, fake_labels_smooth = smooth_labels(batch_size, batch_size)
#             output = netD(real_cpu)  # 传递四维张量
#             D_x = output.mean().item()
#
#             # train with fake
#             noise = torch.randn(batch_size, nz, 1, 1, device=device)
#             fake = netG(noise)
#             output_fake = netD(fake.detach())
#             D_G_z1 = output_fake.mean().item()
#             errD = mixed_loss(output, output_fake, real_labels_smooth, fake_labels_smooth, epoch)
#             errD.backward()
#             optimizerD.step()
#
#             ############################
#             # (2) Update G network
#             ###########################
#             netG.zero_grad()
#             output = netD(fake)
#             D_G_z2 = output.mean().item()
#             bce_loss = criterion(output, real_labels_smooth)
#             percep_loss = perceptual_loss(fake, real_cpu) # 注意：real_cpu 仍然是四维张量
#             lambda_percep = min(epoch / 50, 0.6)
#             normalized_percep_loss = percep_loss / (percep_loss.detach().mean() + 1e-8)
#             errG = (1 - lambda_percep) * bce_loss + lambda_percep * normalized_percep_loss
#             errG.backward()
#             optimizerG.step()
#
#             print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
#                 epoch, num_epochs, i, len(data_iterator), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
#         schedulerD.step()
#         schedulerG.step()
#         if epoch == num_epochs - 1:
#             torch.save(netG.state_dict(), f'{model_dir}/netG_final.pth')
#             torch.save(netD.state_dict(), f'{model_dir}/netD_final.pth')
#     delta_netG = {key: netG.state_dict()[key] - global_netG_params[key] for key in global_netG_params.keys()}
#     delta_netD = {key: netD.state_dict()[key] - global_netD_params[key] for key in global_netD_params.keys()}
#     return {
#         'delta_netG': delta_netG,
#         'delta_netD': delta_netD
#     }

def Train_capabilities(helper, local_model, target_model, agent_name_keys, adversary_client_capabilities):
    training_time_dict = {}
    communication_time_dict = {}
    combined_scores = {}
    dis_time = 60  # 基础训练时间（秒）
    model_size = 11700000  # 模型大小（字节）

    # 遍历所有模型
    for model_id in range(len(helper.params['adversary_list'])):
        epochs_local_update_list = []
        last_local_model = dict()

        # 复制目标模型的参数作为上一次的本地模型状态
        for name, data in target_model.state_dict().items():
            last_local_model[name] = target_model.state_dict()[name].clone()

        agent_name_key = helper.params['adversary_list'][model_id]
        print(f"Processing agent: {agent_name_key}")
        model = local_model
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()

        # 获取当前客户端的计算和通信能力
        capabilities = adversary_client_capabilities[agent_name_key]
        upload_bandwidth = capabilities['upload_bandwidth']  # Mbps
        download_bandwidth = capabilities['download_bandwidth']  # Mbps
        latency = capabilities['latency']  # 毫秒
        # 动态计算 compute_speed
        compute_speed = capabilities['compute_speed']

        # 初始化总训练时间和通信时间
        total_training_time = 0
        total_communication_time = 0

        # 模拟多轮训练
        for epoch in range(helper.params['aggr_epoch_interval']):
            start_time = time.time()
            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
            for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                temp_local_epoch += 1

                _, data_iterator = helper.train_data[agent_name_key]
                total_loss = 0.
                correct = 0
                dataset_size = 0
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(data_iterator, batch, evaluation=False)

                    dataset_size += len(data)
                    output = model(data)
                    loss = nn.functional.cross_entropy(output, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.data
                    pred = output.data.max(1)[1]
                    correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

            # 计算训练时间
            end_time = time.time()
            # dis_time = end_time - start_time
            training_time = dis_time / compute_speed  # 使用动态计算的 compute_speed
            total_training_time += training_time

            # 通信时间计算

            upload_time = model_size / (upload_bandwidth * 1e6 / 8) + latency / 1000.0  # 秒
            download_time = model_size / (download_bandwidth * 1e6 / 8) + latency / 1000.0  # 秒
            total_communication_time += (upload_time + download_time) / 2  # 平均上传和下载时间

        # 记录训练时间和通信时间
        training_time_dict[agent_name_key] = total_training_time
        communication_time_dict[agent_name_key] = total_communication_time

    # 计算训练时间和通信时间的评分
    training_scores = evaluate_performance(training_time_dict)
    communication_scores = evaluate_performance(communication_time_dict)

    # 综合得分
    for client in training_scores:
        combined_scores[client] = {
            'training_score': training_scores[client],
            'communication_score': communication_scores[client],
            'training_time': training_time_dict[client],
            'total_communication_time': communication_time_dict[client]
        }

    return combined_scores


def evaluate_performance(duration_dict):
    # 确保字典非空
    if not duration_dict:
        raise ValueError("Input dictionary is empty.")

    performance_scores = dict()

    # 找出最小和最大时间以进行归一化
    min_duration = min(duration_dict.values())
    max_duration = max(duration_dict.values())

    # 如果所有值都相同，最小和最大值会相同，需要避免除以零的错误
    if min_duration == max_duration:
        raise ValueError("All durations are identical; normalization is not possible.")

    # 为每个客户端计算性能分数
    for client, duration in duration_dict.items():
        # 归一化时间
        normalized_time = (duration - min_duration) / (max_duration - min_duration)

        # 计算分数，这里我们将分数范围反转，使得时间越短，分数越高
        score = 1 - normalized_time
        performance_scores[client] = score * 100

    return performance_scores


