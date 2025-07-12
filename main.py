import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import train
import test
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import csv
from torchvision import transforms
from loan_helper import LoanHelper
from image_helper import ImageHelper
from utils.utils import dict_html
import utils.csv_record as csv_record
import yaml
import time
import visdom
import numpy as np
import random
import config
import copy
import matplotlib.pyplot as plt
import os
from inception_score import *
from gan_model import Generator, Discriminator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")

vis = visdom.Visdom(port=8098)
criterion = torch.nn.CrossEntropyLoss()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)


def trigger_test_byindex(helper, index, vis, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_trigger(helper=helper, model=helper.target_model,
                                   adver_trigger_index=index)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_index_" + str(index) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    if helper.params['vis_trigger_split_test']:
        helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                   eid=helper.params['environment_name'],
                                                   name="global_in_index_" + str(index) + "_trigger")


def trigger_test_byname(helper, agent_name_key, vis, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_" + str(agent_name_key) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    if helper.params['vis_trigger_split_test']:
        helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                   eid=helper.params['environment_name'],
                                                   name="global_in_" + str(agent_name_key) + "_trigger")


def vis_agg_weight(helper, names, weights, epoch, vis, adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0, len(names)):
        _name = names[i]
        _weight = weights[i]
        _is_poison = False
        if _name in adversarial_name_keys:
            _is_poison = True
        helper.target_model.weight_vis(vis=vis, epoch=epoch, weight=_weight, eid=helper.params['environment_name'],
                                       name=_name, is_poisoned=_is_poison)


def vis_fg_alpha(helper, names, alphas, epoch, vis, adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0, len(names)):
        _name = names[i]
        _alpha = alphas[i]
        _is_poison = False
        if _name in adversarial_name_keys:
            _is_poison = True
        helper.target_model.alpha_vis(vis=vis, epoch=epoch, alpha=_alpha, eid=helper.params['environment_name'],
                                      name=_name, is_poisoned=_is_poison)


def generate_random_client_capabilities(adversary_list, alpha=0.1, beta=0.05):
    """
    基于硬件特性模拟恶意客户端的计算能力，包含 RAM 和存储速度对计算能力的影响。
    :param num_adversaries: 恶意客户端数量
    :param alpha: RAM 的权重
    :param beta: 存储速度的权重
    :return: 一个字典，包含每个恶意客户端的硬件特性和计算能力
    """
    capabilities = {}

    # 定义硬件特性的范围
    hardware_specs = {
        "very_high": {  # 非常高端设备
            "cpu_cores": (10, 12),  # 高端设备核心数
            "cpu_frequency": (3.2, 3.5),  # 主频接近上限
            "ram": (24, 32),  # 高端设备内存
            "storage_speed": (2500, 3000),  # NVMe SSD 的高端速度
            "upload_bandwidth": (400, 500),  # 高速上传带宽
            "download_bandwidth": (900, 1000),  # 高速下载带宽
            "latency": (5, 10)  # 网络延迟极低
        },
        "high_end": {  # 高端设备
            "cpu_cores": (8, 10),
            "cpu_frequency": (3.0, 3.2),
            "ram": (16, 24),
            "storage_speed": (1500, 2500),
            "upload_bandwidth": (200, 400),
            "download_bandwidth": (600, 900),
            "latency": (10, 20)
        },
        "mid_range": {  # 中端设备
            "cpu_cores": (4, 8),
            "cpu_frequency": (2.5, 3.0),
            "ram": (8, 16),
            "storage_speed": (800, 1500),
            "upload_bandwidth": (100, 200),
            "download_bandwidth": (300, 600),
            "latency": (20, 50)
        },
        "low_end": {  # 低端设备
            "cpu_cores": (2, 4),
            "cpu_frequency": (1.8, 2.5),
            "ram": (4, 8),
            "storage_speed": (100, 800),
            "upload_bandwidth": (10, 100),
            "download_bandwidth": (50, 300),
            "latency": (50, 100)
        }
    }

    # 定义正态分布的均值和标准差
    mean = 0.5  # 假设高端设备占大多数，均值偏向高端设备
    std_dev = 0.2  # 标准差，控制分布的宽度

    # 生成正态分布的设备类型比例
    raw_ratios = np.random.normal(loc=mean, scale=std_dev, size=len(adversary_list))
    raw_ratios = np.clip(raw_ratios, 0, 1)  # 将比例限制在 [0, 1] 范围内

    # 为每个恶意客户端分配设备类型
    adversary_device_distribution = []
    for ratio in raw_ratios:
        if ratio > 0.8:  # 比例大于 0.66，分配为高端设备
            adversary_device_distribution.append("very_high")
        elif ratio > 0.6:  # 比例在 0.33 到 0.66 之间，分配为中端设备
            adversary_device_distribution.append("high_end")
        elif ratio > 0.3:  # 比例在 0.33 到 0.66 之间，分配为中端设备
            adversary_device_distribution.append("mid_range")
        else:  # 比例小于等于 0.33，分配为低端设备
            adversary_device_distribution.append("low_end")

    # 为每个恶意客户端生成硬件特性和能力
    for i, adversary_id in enumerate(adversary_list):
        if adversary_id == adversary_list[-1]:
            # 为最后一个恶意客户端分配最高配置
            device_type = "high_end"
            cpu_cores = hardware_specs[device_type]["cpu_cores"][1]  # 最大 CPU 核心数
            cpu_frequency = hardware_specs[device_type]["cpu_frequency"][1]  # 最大 CPU 主频
            ram = hardware_specs[device_type]["ram"][1]  # 最大内存
            storage_speed = hardware_specs[device_type]["storage_speed"][1]  # 最大存储速度
            upload_bandwidth = hardware_specs[device_type]["upload_bandwidth"][1]  # 最大上传带宽
            download_bandwidth = hardware_specs[device_type]["download_bandwidth"][1]  # 最大下载带宽
            latency = hardware_specs[device_type]["latency"][0]  # 最小网络延迟
        else:
            # 根据设备类型生成硬件特性
            device_type = adversary_device_distribution[i]
            cpu_cores = np.random.randint(*hardware_specs[device_type]["cpu_cores"])
            cpu_frequency = np.random.uniform(*hardware_specs[device_type]["cpu_frequency"])
            ram = np.random.uniform(*hardware_specs[device_type]["ram"])
            storage_speed = np.random.uniform(*hardware_specs[device_type]["storage_speed"])
            upload_bandwidth = np.random.uniform(*hardware_specs[device_type]["upload_bandwidth"])
            download_bandwidth = np.random.uniform(*hardware_specs[device_type]["download_bandwidth"])
            latency = np.random.uniform(*hardware_specs[device_type]["latency"])

        # 计算综合计算能力
        compute_speed = (cpu_cores * cpu_frequency) + alpha * ram + beta * storage_speed

        # 确保生成的值在合理范围内
        upload_bandwidth = max(1, upload_bandwidth)  # 上传带宽不能小于 1 Mbps
        download_bandwidth = max(1, download_bandwidth)  # 下载带宽不能小于 1 Mbps
        latency = max(1, latency)  # 延迟不能小于 1 毫秒

        capabilities[adversary_id] = {
            'upload_bandwidth': round(upload_bandwidth, 2),
            'download_bandwidth': round(download_bandwidth, 2),
            'latency': round(latency, 2),
            'compute_speed': round(compute_speed, 2)  # 综合计算能力
        }

    return capabilities


def save_plots(losses, accuracies, is_poison, output_dir='./output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if is_poison == True:
        label = "benign"
    else:
        label = "malicious"
    epochs = range(200, 200 + len(losses))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Train Loss')
    plt.title(f'{label}_Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Train Accuracy')
    plt.title(f'{label}_Training Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{label}_training.png'))
    plt.close()


def save_model(model, path, filename="final_model.pth"):
    """
    保存PyTorch模型的状态字典。
    :param model: 要保存的模型。
    :param path: 保存模型的路径。
    :param filename: 保存的文件名。
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, filename))
    print(f"Model saved to {os.path.join(path, filename)}")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def gan_aggregate_updates(global_netG, global_netD, client_updates,eta=0.1):
    # 初始化聚合后的更新量
    aggregated_delta_netG = {key: torch.zeros_like(value) for key, value in global_netG.state_dict().items()}
    aggregated_delta_netD = {key: torch.zeros_like(value) for key, value in global_netD.state_dict().items()}

    # 累加所有客户端的更新
    for update in client_updates:
        for key in aggregated_delta_netG.keys():
            aggregated_delta_netG[key] += update['delta_netG'][key]
        for key in aggregated_delta_netD.keys():
            aggregated_delta_netD[key] += update['delta_netD'][key]

    # 对更新量取平均
    num_clients = len(client_updates)
    for key in aggregated_delta_netG.keys():
        aggregated_delta_netG[key] = aggregated_delta_netG[key].float()
        aggregated_delta_netG[key] /= num_clients
    for key in aggregated_delta_netD.keys():
        aggregated_delta_netD[key] = aggregated_delta_netD[key].float()
        aggregated_delta_netD[key] /= num_clients

    # 将聚合后的更新量乘以学习率 eta
    for key in aggregated_delta_netG.keys():
        aggregated_delta_netG[key] *= eta
    for key in aggregated_delta_netD.keys():
        aggregated_delta_netD[key] *= eta

    # 将聚合后的更新应用到全局模型
    global_netG_state = global_netG.state_dict()
    global_netD_state = global_netD.state_dict()
    for key in global_netG_state.keys():
        global_netG_state[key] = global_netG_state[key].float()
        global_netG_state[key] += aggregated_delta_netG[key]
    for key in global_netD_state.keys():
        global_netD_state[key] = global_netD_state[key].float()
        global_netD_state[key] += aggregated_delta_netD[key]

    # 加载更新后的参数到全局模型
    global_netG.load_state_dict(global_netG_state)
    global_netD.load_state_dict(global_netD_state)

    return global_netG, global_netD
if __name__ == '__main__':
    print('Start training')
    np.random.seed(1)
    time_start_load_everything = time.time()
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f, Loader=yaml.SafeLoader)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    accuracy_log_file = os.path.join('./output', 'accuracy_log.txt')
    if not os.path.exists('./output'):
        os.makedirs('./output')
    with open(accuracy_log_file, 'w') as f:
        f.write("Epoch, Benign Accuracy, Poison Accuracy\n")  # 写入表头    
    if params_loaded['type'] == config.TYPE_LOAN:
        helper = LoanHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'loan'))
        helper.load_data(params_loaded)
    elif params_loaded['type'] == config.TYPE_CIFAR:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'cifar'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_TINYIMAGENET:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'tiny'))
        helper.load_data()
    else:
        helper = None

    logger.info(f'load data done')
    helper.create_model()
    logger.info(f'create model done')
    ### Create models
    if helper.params['is_poison']:
        logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")

    best_loss = float('inf')

    vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
             env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)
    weight_accumulator_fool = helper.init_weight_accumulator(helper.target_model)

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    submit_update_dict = None
    num_no_progress = 0

    global_epoch_accuracies = []
    global_epoch_losses = []
    poison_global_epoch_accuracies = []
    poison_global_epoch_losses = []

    if helper.params['is_dcgan']:
        for agent_name_key in helper.params['adversary_list']:
            # 设置设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train.train_dcgan(helper=helper, device=device, client_index=agent_name_key)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ngpu = 1
        # global_netG = Generator(ngpu).to(device)
        # global_netG.apply(weights_init)
        # global_netD = Discriminator(ngpu).to(device)
        # global_netD.apply(weights_init)
        # # 联邦学习参数
        # num_rounds = 50
        # num_epochs_per_client = 5
        #
        # # 联邦学习循环
        # for round in range(num_rounds):
        #     print(f"Round {round + 1}/{num_rounds}")
        #     client_updates = []
        #     fake_images = []
        #     # 遍历所有客户端
        #     for agent_name_key in helper.params['adversary_list']:
        #         print(f"Training on client: {agent_name_key}")
        #
        #         # 将全局模型发送到客户端
        #         local_netG = copy.deepcopy(global_netG)
        #         local_netD = copy.deepcopy(global_netD)
        #
        #         # 客户端本地训练
        #         client_update = train.train_dcgan(helper, device, agent_name_key, local_netG, local_netD,
        #                                           num_epochs=num_epochs_per_client)
        #         client_updates.append(client_update)
        #
        #     # 聚合客户端更新
        #     global_netG, global_netD = gan_aggregate_updates(global_netG, global_netD, client_updates)
        #     for _ in range(8):  # 8次循环，每次生成128张图像
        #         noise = torch.randn(128, 100, 1, 1, device=device)
        #         fake_batch = global_netG(noise)
        #         fake_images.append(fake_batch)
        #
        #         # 将生成的图像转换为适合 Inception Score 计算的格式
        #     fake_images = torch.cat(fake_images, dim=0)[:1000].detach().cpu()  # 只取前1000张图像
        #
        #     print('Calculating Inception Score for epoch %d...' % (round))
        #     # 分批计算IS
        #     batch_size = 32
        #     is_mean, is_std = inception_score(fake_images, cuda=True, batch_size=32, resize=True,
        #                                       splits=10)  # 增加splits以提高评估的稳定性
        #     print('Epoch %d - Inception Score: Mean: %.4f, Std: %.4f' % (round, is_mean, is_std))
        # torch.save(global_netG.state_dict(), "models/dcgan_model/global_netG.pth")
        # torch.save(global_netD.state_dict(), "models/dcgan_model/global_netD.pth")
    else:
        client_settings = {}
        combined_scores = {}
        sorted_clients_index = {}
        adversary_client_capabilities = generate_random_client_capabilities(helper.params['adversary_list'])

        adversarial_name_keys = []
        # _cifar_p
        training_power_levels_cifar_p = {
            'low': {'threshold': 25, 'learning_rate': 0.1, 'epochs': 4, 'batch_size': 24, 'poisoning_per_batch': 1},
            'medium': {'threshold': 50, 'learning_rate': 0.075, 'epochs': 5, 'batch_size': 43,
                       'poisoning_per_batch': 4},
            'high': {'threshold': 75, 'learning_rate': 0.05, 'epochs': 6, 'batch_size': 64, 'poisoning_per_batch': 5},
            'very_high': {'threshold': 100, 'learning_rate': 0.025, 'epochs': 8, 'batch_size': 96,
                          'poisoning_per_batch': 7},
        }
        training_power_levels = {
            'low': {'threshold': 25, 'learning_rate': 0.08, 'epochs': 4, 'batch_size': 32, 'poisoning_per_batch': 3},
            'medium': {'threshold': 50, 'learning_rate': 0.08, 'epochs': 4, 'batch_size': 32, 'poisoning_per_batch': 3},
            'high': {'threshold': 75, 'learning_rate': 0.08, 'epochs': 4, 'batch_size': 32, 'poisoning_per_batch': 3},
            'very_high': {'threshold': 100, 'learning_rate': 0.08, 'epochs': 4, 'batch_size': 32,
                          'poisoning_per_batch': 3},
        }
        # _cifar_p_fangyu
        training_power_levels_cifar_p_fangyu = {
            'low': {'threshold': 25, 'learning_rate': 0.1, 'epochs': 4, 'batch_size': 24, 'poisoning_per_batch': 1},
            'medium': {'threshold': 50, 'learning_rate': 0.075, 'epochs': 5, 'batch_size': 43,
                       'poisoning_per_batch': 2},
            'high': {'threshold': 75, 'learning_rate': 0.05, 'epochs': 6, 'batch_size': 64, 'poisoning_per_batch': 3},
            'very_high': {'threshold': 100, 'learning_rate': 0.025, 'epochs': 8, 'batch_size': 96,
                          'poisoning_per_batch': 4},
        }
        # _cifar
        training_power_levels_cifar = {
            'low': {'threshold': 25, 'learning_rate': 0.05, 'epochs': 6, 'batch_size': 64, 'poisoning_per_batch': 5},
            'medium': {'threshold': 50, 'learning_rate': 0.05, 'epochs': 6, 'batch_size': 64, 'poisoning_per_batch': 5},
            'high': {'threshold': 75, 'learning_rate': 0.05, 'epochs': 6, 'batch_size': 64, 'poisoning_per_batch': 5},
            'very_high': {'threshold': 100, 'learning_rate': 0.05, 'epochs': 6, 'batch_size': 64,
                          'poisoning_per_batch': 5},
        }
        # _mnist_p_fangyu
        training_power_levels_mnist_p_fangyu = {
            'low': {'threshold': 25, 'learning_rate': 0.025, 'epochs': 5, 'batch_size': 32, 'poisoning_per_batch': 5},
            'medium': {'threshold': 50, 'learning_rate': 0.0375, 'epochs': 7, 'batch_size': 48,
                       'poisoning_per_batch': 7},
            'high': {'threshold': 75, 'learning_rate': 0.05, 'epochs': 10, 'batch_size': 64, 'poisoning_per_batch': 10},
            'very_high': {'threshold': 100, 'learning_rate': 0.05, 'epochs': 12, 'batch_size': 76,
                          'poisoning_per_batch': 12},
        }
        # _mnist_p
        training_power_levels_mnist_p = {
            'low': {'threshold': 25, 'learning_rate': 0.025, 'epochs': 5, 'batch_size': 32, 'poisoning_per_batch': 10},
            'medium': {'threshold': 50, 'learning_rate': 0.0375, 'epochs': 7, 'batch_size': 48,
                       'poisoning_per_batch': 15},
            'high': {'threshold': 75, 'learning_rate': 0.05, 'epochs': 10, 'batch_size': 64, 'poisoning_per_batch': 20},
            'very_high': {'threshold': 100, 'learning_rate': 0.05, 'epochs': 12, 'batch_size': 76,
                          'poisoning_per_batch': 24},
        }
        # _mnist
        training_power_levels_mnist = {
            'low': {'threshold': 25, 'learning_rate': 0.05, 'epochs': 10, 'batch_size': 64, 'poisoning_per_batch': 20},
            'medium': {'threshold': 50, 'learning_rate': 0.05, 'epochs': 10, 'batch_size': 64,
                       'poisoning_per_batch': 20},
            'high': {'threshold': 75, 'learning_rate': 0.05, 'epochs': 10, 'batch_size': 64, 'poisoning_per_batch': 20},
            'very_high': {'threshold': 100, 'learning_rate': 0.05, 'epochs': 10, 'batch_size': 64,
                          'poisoning_per_batch': 20},
        }

        communication_power_levels = {
            'low': {'threshold': 20, 'is_pruning': True},
            'high': {'threshold': 100, 'is_pruning': False},
        }
        for client in helper.params['adversary_list']:
            adversarial_name_keys.append(client)
        combined_scores = train.Train_capabilities(helper=helper,
                                                   local_model=helper.local_model,
                                                   target_model=helper.target_model,
                                                   agent_name_keys=adversarial_name_keys,
                                                   adversary_client_capabilities=adversary_client_capabilities)

        sorted_clients_index = {index: client for index, (client, _) in enumerate(
            sorted(combined_scores.items(), key=lambda item: item[1]['training_score'], reverse=True))}

        for client, scores in combined_scores.items():
            training_score = scores['training_score']
            communication_score = scores['communication_score']

            if training_score <= training_power_levels['low']['threshold']:
                training_level = 'low'
            elif training_score <= training_power_levels['medium']['threshold']:
                training_level = 'medium'
            elif training_score <= training_power_levels['high']['threshold']:
                training_level = 'high'
            else:
                training_level = 'very_high'

            if communication_score <= communication_power_levels['low']['threshold']:
                communication_level = 'low'
            else:
                communication_level = 'high'

            client_settings[client] = {
                'learning_rate': training_power_levels[training_level]['learning_rate'],
                'epochs': training_power_levels[training_level]['epochs'],
                'batch_size': training_power_levels[training_level]['batch_size'],
                'is_pruning': communication_power_levels[communication_level]['is_pruning'],
                'poisoning_per_batch': training_power_levels[training_level]['poisoning_per_batch'],
            }

        for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
            start_time = time.time()
            t = time.time()

            agent_name_keys = helper.participants_list
            adversarial_name_keys = []
            if helper.params['is_random_namelist']:
                # if helper.params['is_random_adversary']:  # random choose , maybe don't have advasarial
                #     agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
                #     for _name_keys in agent_name_keys:
                #         if _name_keys in helper.params['adversary_list']:
                #             adversarial_name_keys.append(_name_keys)
                if helper.params[
                    'is_random_adversary']:  # Randomly choose clients, ensuring specified number of adversarial clients
                    # Step 1: Randomly select adversarial clients
                    if len(helper.params['adversary_list']) > 0:

                        # 确保选择的数量不超过恶意客户端列表的长度
                        num_adversaries_to_select = min(5, len(helper.params['adversary_list']))
                        selected_adversaries = random.sample(helper.params['adversary_list'], num_adversaries_to_select)
                        adversarial_name_keys.extend(selected_adversaries)  # 添加到 adversarial_name_keys 列表
                    else:
                        selected_adversaries = []  # 如果没有恶意客户端，则为空列表

                    # Step 2: Randomly select remaining clients from benign clients
                    benign_num = helper.params['no_models'] - len(adversarial_name_keys)  # 剩余需要选择的普通客户端数量
                    benign_clients = [client for client in helper.participants_list if
                                      client not in helper.params['adversary_list']]
                    if benign_num > 0:
                        random_agent_name_keys = random.sample(benign_clients, benign_num)
                    else:
                        random_agent_name_keys = []

                    # Step 3: Combine adversarial and benign clients
                    agent_name_keys = adversarial_name_keys + random_agent_name_keys
                else:  # must have advasarial if this epoch is in their poison epoch
                    #                     ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
                    #                     for idx in range(0, len(sorted_clients_index)):
                    #                         for ongoing_epoch in ongoing_epochs:
                    #                             if ongoing_epoch in helper.params[str(idx) + '_poison_epochs']:
                    #                                 if sorted_clients_index[idx] not in adversarial_name_keys:
                    #                                     adversarial_name_keys.append(sorted_clients_index[idx])

                    #                     nonattacker = []
                    #                     for adv in helper.params['adversary_list']:
                    #                         if adv not in adversarial_name_keys:
                    #                             nonattacker.append(copy.deepcopy(adv))
                    #                     benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                    #                     random_agent_name_keys = random.sample(helper.benign_namelist + nonattacker, benign_num)
                    #                     agent_name_keys = adversarial_name_keys + random_agent_name_keys

                    for idx in range(0, len(sorted_clients_index)):
                        intercal = epoch - helper.params[str(idx) + '_poison_epochs'][0]
                        # 检查当前轮次是否是该客户端的攻击轮次
                        if (intercal % helper.params[str(idx) + '_poison_interval'] == 0) and intercal >= 0:
                            adversarial_name_keys.append(sorted_clients_index[idx])

                    # 处理非攻击客户端
                    nonattacker = []
                    for adv in helper.params['adversary_list']:
                        if adv not in adversarial_name_keys:
                            nonattacker.append(copy.deepcopy(adv))
                    # if len(adversarial_name_keys) > 2:
                    #     removed_adv = random.choice(adversarial_name_keys)  # 随机选择一个要剔除的客户端
                    #     adversarial_name_keys.remove(removed_adv)  # 从列表中移除    
                    benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                    random_agent_name_keys = random.sample(helper.benign_namelist + nonattacker, benign_num)
                    agent_name_keys = adversarial_name_keys + random_agent_name_keys
            else:
                if helper.params['is_random_adversary'] == False:
                    adversarial_name_keys = copy.deepcopy(helper.params['adversary_list'])
            logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')
            epochs_submit_update_dict, epochs_submit_update_dict_fool, num_samples_dict = train.train(helper=helper,
                                                                                                      start_epoch=epoch,
                                                                                                      local_model=helper.local_model,
                                                                                                      target_model=helper.target_model,
                                                                                                      is_poison=
                                                                                                      helper.params[
                                                                                                          'is_poison'],
                                                                                                      agent_name_keys=agent_name_keys,
                                                                                                      client_settings=client_settings,
                                                                                                      combined_scores=combined_scores,
                                                                                                      sorted_clients_index=sorted_clients_index)
            logger.info(f'time spent on training: {time.time() - t}')

            weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                                   agent_name_keys, num_samples_dict)

            is_updated = True
            if helper.params['aggregation_methods'] == config.AGGR_MEAN:
                # Average the models
                is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                          target_model=helper.target_model,
                                                          epoch_interval=helper.params['aggr_epoch_interval'])
                num_oracle_calls = 1
            elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
                maxiter = helper.params['geom_median_maxiter']
                num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(
                    helper.target_model,
                    updates,
                    maxiter=maxiter)
                vis_agg_weight(helper, names, weights, epoch, vis, adversarial_name_keys)
                vis_fg_alpha(helper, names, alphas, epoch, vis, adversarial_name_keys)

            elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)
                weight_accumulator_fool, _ = helper.accumulate_weight_fool(weight_accumulator_fool,
                                                                           epochs_submit_update_dict_fool,
                                                                           agent_name_keys, num_samples_dict,
                                                                           weights)
                is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator_fool,
                                                          target_model=helper.target_model,
                                                          epoch_interval=helper.params['aggr_epoch_interval'])

                vis_agg_weight(helper, names, weights, epoch, vis, adversarial_name_keys)
                vis_fg_alpha(helper, names, alphas, epoch, vis, adversarial_name_keys)
                num_oracle_calls = 1

            # clear the weight_accumulator
            weight_accumulator = helper.init_weight_accumulator(helper.target_model)
            weight_accumulator_fool = helper.init_weight_accumulator(helper.target_model)

            temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

            epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                           model=helper.target_model,
                                                                           is_poison=False,
                                                                           visualize=True, agent_name_key="global")
            global_epoch_losses.append(epoch_loss)
            global_epoch_accuracies.append(epoch_acc)

            csv_record.test_result.append(
                ["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
            if len(csv_record.scale_temp_one_row) > 0:
                csv_record.scale_temp_one_row.append(round(epoch_acc, 4))

            if helper.params['is_poison']:

                epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                        epoch=temp_global_epoch,
                                                                                        model=helper.target_model,
                                                                                        is_poison=True,
                                                                                        visualize=True,
                                                                                        agent_name_key="global")
                poison_global_epoch_losses.append(epoch_loss)
                poison_global_epoch_accuracies.append(epoch_acc_p)

                csv_record.posiontest_result.append(
                    ["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])

                # test on local triggers
                csv_record.poisontriggertest_result.append(
                    ["global", "combine", "", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret,
                     epoch_total])
                if helper.params['vis_trigger_split_test']:
                    helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc_p, loss=None,
                                                               eid=helper.params['environment_name'],
                                                               name="global_combine")
                if len(helper.params['adversary_list']) == 1:  # centralized attack
                    if helper.params[
                        'centralized_test_trigger'] == True:  # centralized attack test on local triggers
                        for j in range(0, helper.params['trigger_num']):
                            trigger_test_byindex(helper, j, vis, epoch)
                else:  # distributed attack
                    for agent_name_key in helper.params['adversary_list']:
                        trigger_test_byname(helper, agent_name_key, vis, epoch)

            # helper.save_model(epoch=epoch, val_loss=epoch_loss)
            with open(accuracy_log_file, 'a') as f:
                f.write(f"{temp_global_epoch}, {epoch_acc}, {epoch_acc_p}\n")
            logger.info(f'Done in {time.time() - start_time} sec.')
            csv_record.save_result_csv(epoch, helper.params['is_poison'], helper.folder_path)

        model_save_path = os.path.join('DBA-master', 'saved_models', 'saved_models')
        save_model(helper.target_model, model_save_path, "final_global_model.pth")

        save_plots(global_epoch_losses, global_epoch_accuracies, is_poison=False)
        save_plots(poison_global_epoch_losses, poison_global_epoch_accuracies, is_poison=False)

        logger.info('Saving all the graphs.')
        logger.info(f"This run has a label: {helper.params['current_time']}. "
                    f"Visdom environment: {helper.params['environment_name']}")

        vis.save([helper.params['environment_name']])
