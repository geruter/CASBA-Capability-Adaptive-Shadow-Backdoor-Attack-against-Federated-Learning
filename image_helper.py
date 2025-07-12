from collections import defaultdict, Counter
import matplotlib.pyplot as plt

import torch
import torch.utils.data

from torch.utils.data import Dataset

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.resnet_cifar import ResNet18
from models.MnistNet import MnistNet
from models.resnet_tinyimagenet import resnet18

logger = logging.getLogger("logger")
import config
from config import device
import copy

import yaml

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import datetime
import json
from gan_model import Generator, Discriminator
from PIL import Image


class ImageHelper(Helper):

    def create_model(self):
        local_model = None
        target_model = None
        if self.params['type'] == config.TYPE_CIFAR:
            local_model = ResNet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = ResNet18(name='Target',
                                    created_time=self.params['current_time'])

        elif self.params['type'] == config.TYPE_MNIST:
            local_model = MnistNet(name='Local',
                                   created_time=self.params['current_time'])
            target_model = MnistNet(name='Target',
                                    created_time=self.params['current_time'])

        elif self.params['type'] == config.TYPE_TINYIMAGENET:

            local_model = resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = resnet18(name='Target',
                                    created_time=self.params['current_time'])

        local_model = local_model.to(device)
        target_model = target_model.to(device)
        if self.params['resumed_model']:
            if torch.cuda.is_available():
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}", map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch'] + 1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model

    def build_classes_dict(self):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes

    # def sample_dirichlet_train_data(self, no_participants, alpha=0.9, share_ratio=0.2):
    #     """
    #         Input: Number of participants and alpha (param for distribution)
    #         Output: A list of indices denoting data in CIFAR training set.
    #         Requires: cifar_classes, a preprocessed class-indice dictionary.
    #         Sample Method: take a uniformly sampled 10-dimension vector as parameters for
    #         dirichlet distribution to sample number of images in each class.
    #     """
    #
    #     cifar_classes = self.classes_dict
    #     class_size = len(cifar_classes[0])  # for cifar: 5000
    #     per_participant_list = defaultdict(list)
    #     no_classes = len(cifar_classes.keys())  # for cifar: 10
    #
    #     image_nums = []
    #     for n in range(no_classes):
    #         image_num = []
    #         random.shuffle(cifar_classes[n])
    #         sampled_probabilities = class_size * np.random.dirichlet(
    #             np.array(no_participants * [alpha]))
    #         for user in range(no_participants):
    #             no_imgs = int(round(sampled_probabilities[user]))
    #             sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
    #             image_num.append(len(sampled_list))
    #             per_participant_list[user].extend(sampled_list)
    #             cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
    #         image_nums.append(image_num)
    #     # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
    #     return per_participant_list
    def sample_dirichlet_train_data(self, no_participants, alpha=0.9, share_ratio=0.2):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """
        cifar_classes = self.classes_dict
        class_size = len(cifar_classes[0])  # for cifar: 5000
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())  # for cifar: 10

        # 用于存储每个客户端的数据量
        participant_data_count = [0] * no_participants

        image_nums = []
        for n in range(no_classes):
            image_num = []
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                participant_data_count[user] += len(sampled_list)  # 累计每个客户端的数据量
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)

        # 找出数据量最多的 10 个客户端
        sorted_participants = sorted(
            enumerate(participant_data_count), key=lambda x: x[1], reverse=True
        )
        top_10_participants = sorted_participants[:10]  # 前 10 个客户端
        top_10_indices = [x[0] for x in top_10_participants]  # 客户端索引
        top_10_counts = [x[1] for x in top_10_participants]  # 数据量

        print("Top 10 participants with the most data:")
        for i, (participant, count) in enumerate(zip(top_10_indices, top_10_counts)):
            print(f"Rank {i + 1}: Client {participant} with {count} samples")

        # 返回分配的客户端数据和数据量信息
        return per_participant_list

    def draw_dirichlet_plot(self, no_classes, no_participants, image_nums, alpha):
        fig = plt.figure(figsize=(10, 5))
        s = np.empty([no_classes, no_participants])
        for i in range(0, len(image_nums)):
            for j in range(0, len(image_nums[0])):
                s[i][j] = image_nums[i][j]
        s = s.transpose()
        left = 0
        y_labels = []
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, no_participants))
        for k in range(no_classes):
            y_labels.append('Label ' + str(k))
        vis_par = [0, 10, 20, 30]
        for k in range(no_participants):
            # for k in vis_par:
            color = category_colors[k]
            plt.barh(y_labels, s[k], left=left, label=str(k), color=color)
            widths = s[k]
            xcenters = left + widths / 2
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # for y, (x, c) in enumerate(zip(xcenters, widths)):
            #     plt.text(x, y, str(int(c)), ha='center', va='center',
            #              color=text_color,fontsize='small')
            left += s[k]
        plt.legend(ncol=20, loc='lower left', bbox_to_anchor=(0, 1), fontsize=4)  #
        # plt.legend(ncol=len(vis_par), bbox_to_anchor=(0, 1),
        #            loc='lower left', fontsize='small')
        plt.xlabel("Number of Images", fontsize=16)
        # plt.ylabel("Label 0 ~ 199", fontsize=16)
        # plt.yticks([])
        fig.tight_layout(pad=0.1)
        # plt.ylabel("Label",fontsize='small')
        fig.savefig(self.folder_path + '/Num_Img_Dirichlet_Alpha{}.pdf'.format(alpha))

    def poison_test_dataset(self):
        logger.info('get poison test loader')
        # delete the test data with target label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               range_no_id)), \
            torch.utils.data.DataLoader(self.test_dataset,
                                        batch_size=self.params['batch_size'],
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                            poison_label_inds))

    def save_images_to_folders_by_label(self,dataset, indices_per_participant, output_dir, num_classes=10):
        """
        将数据集中的图像按照标签保存到不同参与者的文件夹中。

        Args:
            dataset (Dataset): PyTorch 数据集对象。
            indices_per_participant (dict): 每个参与者对应的数据索引。
            output_dir (str): 输出文件夹路径。
            num_classes (int): 数据集的类别数量，默认为 10。
        """
        import os
        from torchvision.transforms import ToPILImage

        # 创建根目录
        os.makedirs(output_dir, exist_ok=True)

        for participant, indices in indices_per_participant.items():
            # 为每个参与者创建文件夹（以客户端 ID 命名）
            participant_dir = os.path.join(output_dir, f"{participant}")
            os.makedirs(participant_dir, exist_ok=True)

            # 在每个参与者的文件夹中创建子文件夹（0-num_classes-1，按标签分类）
            for label in range(num_classes):
                label_dir = os.path.join(participant_dir, str(label))
                os.makedirs(label_dir, exist_ok=True)

            # 遍历该参与者的所有数据索引
            for idx in indices:
                # 从数据集中获取图像和标签
                image, label = dataset[idx]

                # 如果图像是 Tensor，需要转换为 PIL Image
                if isinstance(image, torch.Tensor):
                    image = ToPILImage()(image)

                # 保存图像到对应的标签文件夹
                image_path = os.path.join(participant_dir, str(label), f"image_{idx}.png")
                image.save(image_path)

        print(f"所有图像已保存到 {output_dir} 文件夹中。")

    def load_data(self):
        logger.info('Loading data')
        dataPath = './data'
        if self.params['type'] == config.TYPE_CIFAR:
            ### data load
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_gan_train = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                                  transform=transform_train)
            self.gan_train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                                      transform=transform_gan_train)

            self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=transform_test)

        elif self.params['type'] == config.TYPE_MNIST:

            self.train_dataset = datasets.MNIST('data', train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    # transforms.Normalize((0.1307,), (0.3081,))
                                                ]))
            self.test_dataset = datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]))
        elif self.params['type'] == config.TYPE_TINYIMAGENET:

            _data_transforms = {
                'train': transforms.Compose([
                    # transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    # transforms.Resize(224),
                    transforms.ToTensor(),
                ]),
            }
            _data_dir = 'data/tiny-imagenet-200/'
            self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                                      _data_transforms['train'])
            self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                                     _data_transforms['val'])
            logger.info('reading data done')

        self.classes_dict = self.build_classes_dict()
        logger.info('build_classes_dict done')
        if self.params['sampling_dirichlet']:
            ## sample indices for participants using Dirichlet distribution
            self.indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'],  # 100
                alpha=self.params['dirichlet_alpha'])
            # 调用保存图像的方法（只保存一次）
            self.save_images_to_folders_by_label(
                dataset=self.train_dataset,  # 假设你的训练数据集是 self.train_dataset
                indices_per_participant=self.indices_per_participant,
                output_dir="./output_dir",  # 指定输出文件夹
                num_classes=len(self.classes_dict)  # 根据类别数量动态调整
            )
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             self.indices_per_participant.items()]
            gan_train_loaders = [(pos, self.get_gan_train(indices)) for pos, indices in
                                 self.indices_per_participant.items()]

        else:
            ## sample indices for participants that are equally
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]

        logger.info('train loaders done')
        self.train_data = train_loaders
        self.gan_train_data = gan_train_loaders

        self.test_data = self.get_test()
        self.test_data_poison, self.test_targetlabel_data = self.poison_test_dataset()

        self.advasarial_namelist = self.params['adversary_list']

        if self.params['is_random_namelist'] == False:
            self.participants_list = self.params['participants_namelist']
        else:
            self.participants_list = list(range(self.params['number_of_total_participants']))
        # random.shuffle(self.participants_list)
        self.benign_namelist = list(set(self.participants_list) - set(self.advasarial_namelist))

    def create_train_loader_for_client(self, client_id, batch_size):
        indices_per_participant = self.indices_per_participant
        if client_id in indices_per_participant:
            indices = indices_per_participant[client_id]
            return self.get_adversary_train(indices, batch_size)
        else:
            raise ValueError("Client ID not found in the participant list")

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       indices), pin_memory=True, num_workers=8)
        return train_loader
    #
    def get_gan_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.gan_train_dataset,
                                                   batch_size=64,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       indices), pin_memory=True, num_workers=8)
        return train_loader
    # def get_gan_train(self, indices):
    #     """
    #     This method is used along with Dirichlet distribution.
    #     It applies data augmentation to the samples specified by indices.
    #     :param indices: A list of indices specifying the subset of the dataset.
    #     :return: A PyTorch DataLoader with data augmentation applied to the subset.
    #     """
    #     # 定义数据增强操作
    #     transform = transforms.Compose([
    #         transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    #         transforms.RandomChoice([  # 随机选择以下旋转角度之一
    #             transforms.RandomRotation((0, 0)),  # 旋转 0 度
    #             transforms.RandomRotation((90, 90)),  # 旋转 90 度
    #             transforms.RandomRotation((180, 180)),  # 旋转 180 度
    #             transforms.RandomRotation((270, 270))  # 旋转 270 度
    #         ]),
    #         transforms.RandomApply([  # 随机缩放
    #             transforms.Resize((int(0.9 * 64), int(0.9 * 64))),  # 缩放到 90%
    #             transforms.Resize((int(1.1 * 64), int(1.1 * 64)))  # 缩放到 110%
    #         ], p=0.5),  # 50% 概率进行缩放
    #         transforms.Resize((64, 64)),  # 确保图像最终大小为 64x64
    #         transforms.ToTensor(),  # 转为张量
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    #
    #     # 自定义数据集类，应用数据增强
    #     class AugmentedDataset(Dataset):
    #         def __init__(self, dataset, indices, transform=None):
    #             """
    #             :param dataset: 原始数据集
    #             :param indices: 要采样的索引
    #             :param transform: 数据增强操作
    #             """
    #             self.dataset = dataset
    #             self.indices = indices
    #             self.transform = transform
    #
    #         def __len__(self):
    #             return len(self.indices)
    #
    #         def __getitem__(self, idx):
    #             # 获取原始数据集中的样本
    #             original_idx = self.indices[idx]
    #             sample = self.dataset[original_idx]
    #
    #             # 如果数据集返回的是 (data, label) 格式
    #             if isinstance(sample, tuple) and len(sample) == 2:
    #                 data, label = sample
    #                 if self.transform:
    #                     data = self.transform(data)  # 对数据部分进行增强
    #                 return data, label
    #
    #             # 如果数据集只返回数据（无标签）
    #             if self.transform:
    #                 sample = self.transform(sample)
    #             return sample
    #
    #     # 创建增强后的子集数据集
    #     augmented_dataset = AugmentedDataset(self.gan_train_dataset, indices, transform)
    #
    #     # 创建 DataLoader
    #     train_loader = torch.utils.data.DataLoader(
    #         augmented_dataset,
    #         batch_size=64,
    #         shuffle=True,  # 随机打乱数据
    #         pin_memory=True,
    #         num_workers=8
    #     )
    #     return train_loader
    def get_gan_train(self, indices):
        """
        This method is used along with Dirichlet distribution.
        It applies data augmentation to the samples specified by indices,
        explicitly generating 16 augmented versions for each sample.
        :param indices: A list of indices (IDs) specifying the subset of the dataset.
        :return: A PyTorch DataLoader with 16 augmented versions per sample.
        """

        # 自定义数据集类，生成 16 个增强版本
        class AugmentedDataset(Dataset):
            def __init__(self, dataset, indices):
                """
                :param dataset: 原始数据集
                :param indices: 要采样的索引（ID）
                """
                self.dataset = dataset
                self.indices = indices

                # 定义数据增强操作
                self.flip_transforms = [transforms.RandomHorizontalFlip(p=0),  # 不翻转
                                        transforms.RandomHorizontalFlip(p=1)]  # 水平翻转
                self.rotation_transforms = [
                    transforms.RandomRotation((0, 0)),       # 0°
                    transforms.RandomRotation((90, 90)),    # 90°
                    transforms.RandomRotation((180, 180)),  # 180°
                    transforms.RandomRotation((270, 270))   # 270°
                ]
                self.scale_transforms = [
                    transforms.Compose([
                        transforms.Resize((int(0.9 * 64), int(0.9 * 64))),  # 缩小到 90%
                        transforms.Pad((int(0.1 * 64) // 2, int(0.1 * 64) // 2))  # 填充到 64x64
                    ]),
                    transforms.Compose([
                        transforms.Resize((int(1.1 * 64), int(1.1 * 64))),  # 放大到 110%
                        transforms.CenterCrop((64, 64))  # 裁剪到 64x64
                    ])
                ]
                self.final_resize = transforms.Resize((64, 64))  # 最后统一调整回 64x64
                self.to_tensor = transforms.ToTensor()           # 转为张量
                self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化

            def __len__(self):
                # 数据集长度是原始子集长度的 16 倍
                return len(self.indices) * 16

            def __getitem__(self, idx):
                # 获取原始数据集中的样本
                original_idx = self.indices[idx // 16]  # 每个样本重复 16 次
                sample = self.dataset[original_idx]

                # 如果数据集返回的是 (data, label) 格式
                if isinstance(sample, tuple) and len(sample) == 2:
                    data, label = sample
                else:
                    data, label = sample, None

                # 计算增强操作的组合索引
                augmentation_idx = idx % 16
                flip_idx = (augmentation_idx // 8) % 2  # 水平翻转（2 种）
                rotation_idx = (augmentation_idx // 2) % 4  # 旋转（4 种）
                scale_idx = augmentation_idx % 2  # 缩放（2 种）

                # 应用增强操作
                data = Image.fromarray(data.numpy())  # 如果数据是 NumPy 格式，转为 PIL 图像
                data = self.flip_transforms[flip_idx](data)  # 水平翻转
                data = self.rotation_transforms[rotation_idx](data)  # 旋转
                data = self.scale_transforms[scale_idx](data)  # 缩放
                data = self.final_resize(data)  # 调整回原始大小
                data = self.to_tensor(data)  # 转为张量
                data = self.normalize(data)  # 归一化

                if label is not None:
                    return data, label
                return data

        # 创建增强后的子集数据集
        augmented_dataset = AugmentedDataset(self.gan_train_dataset, indices)

        # 使用 SubsetRandomSampler 采样增强后的数据
        sampler = torch.utils.data.sampler.SubsetRandomSampler(range(len(augmented_dataset)))

        # 创建 DataLoader
        train_loader = torch.utils.data.DataLoader(
            augmented_dataset,
            batch_size=64,
            sampler=sampler,  # 使用 SubsetRandomSampler
            pin_memory=True,
            num_workers=8
        )
        return train_loader
    def get_adversary_train(self, indices, batch_size):

        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       indices), pin_memory=True, num_workers=8)
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       sub_indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)
        return test_loader

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target
    # f'/wuhongwei/DBA/DBA-master/fid_weights4/netG_epoch_149.pth'
    def get_cgan_poison_batch(self, bptt, adversarial_index=-1, evaluation=False, current_model_id=-1,
                              poisoning_per_batch=5):
        images, targets = bptt
        ngpu = 1
        # # generator_model_path =  f'/wuhongwei/DBA/DBA-master/weights3/netG_epoch_149.pth'
        # generator_model_path =  f'/wuhongwei/DBA/DBA-master/mnist_/netG_epoch_99.pth'
        generator_model_path =  f"models/dcgan_model/{current_model_id}/cifar"
        # generator_model_path =  f"models/dcgan_model/{current_model_id}/mnist"
        generator = Generator(ngpu).to(device)
        generator.load_state_dict(torch.load(generator_model_path))
        generator.eval()
        # 生成噪声和对应的标签
        noise = torch.randn(poisoning_per_batch, 100, 1, 1, device=device)
        poisoned_images = []
        # 使用生成器生成伪造图像
        generated_images = generator(noise)
        generated_images = (generated_images + 1) / 2
        # 调整图像大小为 32x32
        resize_transform = transforms.Resize((32, 32))
        # resize_transform = transforms.Resize((28, 28))
        for i in range(poisoning_per_batch):
            single_image = generated_images[i]  # 提取单张图像，形状: torch.Size([3, 64, 64])
            resized_image = resize_transform(single_image.unsqueeze(0))  # 调整大小，保持批量维度
            resized_image = resized_image.squeeze(0)  # 去掉批量维度，形状: torch.Size([3, 32, 32])
            poisoned_images.append(resized_image.detach().clone())
        # 初始化"毒化"计数器
        poison_count = 0
        # 初始化新的图像和标签数组
        new_images = images
        new_targets = targets
        # 为前 n 个图像加入触发器并替换原始图像
        for index in range(len(images)):
            if index < poisoning_per_batch:
                poisoned_image = self.add_pixel_pattern(poisoned_images[poison_count], adversarial_index)
                new_images[index] = poisoned_image
                new_targets[index] = self.params['poison_label_swap']  # 修改标签为毒化标签
                poison_count = poison_count + 1
            else:
                # 保持其他图像不变
                new_images[index] = images[index]
                new_targets[index] = targets[index]
        # 将图像和标签转移到指定的设备上
        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()

        # 如果是评估模式，设置不需要梯度
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)

        return new_images, new_targets, poison_count

    def get_cgan_poison_batch_a(self, bptt, adversarial_index=-1, evaluation=False, current_model_id=-1,
                              poisoning_per_batch=6, min_classes_to_include=4):
        images, targets = bptt
        ngpu = 1
        generator_model_path = f'/wuhongwei/DBA/DBA-master/fid_weights4/netG_epoch_149.pth'
        # generator_model_path = f'/wuhongwei/DBA/DBA-master/mnist_/netG_epoch_99.pth'
        generator = Generator(ngpu).to(device)
        generator.load_state_dict(torch.load(generator_model_path))
        generator.eval()
        # min_target = self.scan_least_frequent_label(targets)
        # 生成噪声和对应的标签
        noise = torch.randn(poisoning_per_batch * 3, 100, 1, 1, device=device)
        # 使用生成器生成伪造图像
        generated_images = generator(noise).to(device)
        generated_images = (generated_images + 1) / 2
        # 调整图像大小为 32x32
        resize_transform = transforms.Resize((32, 32))#cifar
        # resize_transform = transforms.Resize((28, 28))#mnist
        generated_images = resize_transform(generated_images)
        # generated_image = generated_images.detach().clone()
        #
        # # 加载 ResNet18 特征提取器
        feature_extractor = self.load_resnet18_feature_extractor(device)
        #
        # 提取真实图像的特征
        real_images_path = f'/wuhongwei/DBA/DBA-master/models/dcgan_model/{current_model_id}/real/images'
        real_image_classes = [os.path.join(real_images_path, str(i)) for i in range(10)]
        real_features_per_class = self.extract_real_features(real_image_classes, feature_extractor, device)

        # 提取生成图像的特征
        generated_features = []

        with torch.no_grad():
            for gen_img in generated_images:
                gen_img = gen_img.unsqueeze(0).to(device)
                features = feature_extractor(gen_img)
                features = features.view(features.size(0), -1)
                generated_features.append(features.squeeze())
        generated_features = torch.stack(generated_features)  # 转换为 PyTorch 张量

        # 计算生成图像的马氏距离
        min_distances, min_classes = self.calculate_mahalanobis_distance(generated_features, real_features_per_class,
                                                                         device)
        class_to_images = {}
        for dist, cls, img in zip(min_distances, min_classes, generated_images):
            if cls not in class_to_images:
                class_to_images[cls] = []
            class_to_images[cls].append((dist, img))

        # 对每个类别的图像按马氏距离排序
        for cls in class_to_images:
            class_to_images[cls].sort(key=lambda x: x[0])  # 按马氏距离升序排序

        # 找到马氏距离最小的 4 个类别
        all_classes_sorted = sorted(class_to_images.keys(), key=lambda cls: class_to_images[cls][0][0])  # 按类别的最小距离排序
        selected_classes = all_classes_sorted[:min_classes_to_include]  # 选择最小的 4 个类别

        # 从选定的类别中每个类别选择一个图像
        selected_images = []
        for cls in selected_classes:
            selected_images.append(class_to_images[cls][0][1])  # 选择该类别马氏距离最小的图像
            class_to_images[cls].pop(0)  # 从类别中移除已选择的图像

        # 计算还需要补充的图像数量
        remaining_images_needed = poisoning_per_batch - len(selected_images)

        # 将剩余的图像按马氏距离排序
        remaining_images = []
        for cls, images1 in class_to_images.items():
            remaining_images.extend(images1)  # 将所有剩余图像加入列表
        remaining_images.sort(key=lambda x: x[0])  # 按马氏距离升序排序

        # 从剩余图像中选择距离最小的几个图像
        selected_images.extend([img for _, img in remaining_images[:remaining_images_needed]])

        # 返回最终选择的图像
        selected_image = [img.detach().clone() for img in selected_images[:poisoning_per_batch]]

        #使用生成器生成伪造图像

        # selected_image = generated_images.detach().clone()

        # 初始化"毒化"计数器
        poison_count = 0
        # 初始化新的图像和标签数组
        new_images = images
        new_targets = targets
        # 为前 n 个图像加入触发器并替换原始图像
        for index in range(len(images)):
            if index < poisoning_per_batch:
                # poisoned_image = self.add_pixel_pattern(poisoned_images[poison_count], adversarial_index)
                poisoned_image = self.add_pixel_pattern(selected_image[poison_count], adversarial_index)
                new_images[index] = poisoned_image
                new_targets[index] = self.params['poison_label_swap']  # 修改标签为毒化标签
                poison_count = poison_count + 1
            else:
                # 保持其他图像不变
                new_images[index] = images[index]
                new_targets[index] = targets[index]
        # 将图像和标签转移到指定的设备上
        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()

        # 如果是评估模式，设置不需要梯度
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)

        return new_images, new_targets, poison_count

    def get_poison_batch(self, bptt, adversarial_index=-1, evaluation=False):
        """
        对一个批次的图像数据进行"毒化"处理，用于对抗性训练或测试。

        Args:
            bptt (tuple): 包含图像和标签的元组，通常为一个批次的数据。
            adversarial_index (int): 用于指定添加到图像中的特定像素模式的索引。
            evaluation (bool): 指示当前操作是评估（测试）还是训练。在评估模式下，所有数据都会被"毒化"。

        Returns:
            tuple: 返回一个元组，包含处理后的图像、标签和被"毒化"的图像数量。
        """
        # 解包输入的批次数据
        images, targets = bptt

        # 初始化"毒化"计数器
        poison_count = 0
        # 初始化新的图像和标签数组
        new_images = images
        new_targets = targets

        # 遍历每个图像
        for index in range(0, len(images)):
            if evaluation:
                # 在评估模式下，对所有图像进行"毒化"
                new_targets[index] = self.params['poison_label_swap']  # 修改标签
                new_images[index] = self.add_pixel_pattern(images[index], adversarial_index)  # 添加像素模式
                poison_count += 1
            else:
                # 在训练模式下，只对部分图像进行"毒化"
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']  # 修改标签
                    new_images[index] = self.add_pixel_pattern(images[index], adversarial_index)  # 添加像素模式
                    poison_count += 1
                else:
                    # 保持其他图像不变
                    new_images[index] = images[index]
                    new_targets[index] = targets[index]

        # 将图像和标签转移到指定的设备上
        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()

        # 如果是评估模式，设置不需要梯度
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)

        # 返回处理后的图像、标签和"毒化"计数
        return new_images, new_targets, poison_count

    def add_pixel_pattern(self, ori_image, adversarial_index):
        """
        在原始图像上添加特定的像素模式，用于生成对抗性样本。

        Args:
            ori_image (array): 原始图像数据。
            adversarial_index (int): 指定使用的对抗性像素模式的索引。如果为-1，则使用所有配置的模式。
            如果不为-1则使用cgan生成对抗性样本并在其中添加后门触发器

        Returns:
            array: 返回添加了像素模式的图像。
        """
        # 深拷贝原始图像，以避免修改原始数据
        image = copy.deepcopy(ori_image)

        # 初始化毒化模式列表
        poison_patterns = []

        # 根据 adversarial_index 选择毒化模式
        if adversarial_index == -1:
            # 如果索引为-1，则结合所有配置的毒化模式
            for i in range(self.params['trigger_num']):
                poison_patterns += self.params[str(i) + '_poison_pattern']
        else:
            # 否则，使用指定索引的毒化模式
            poison_patterns = self.params[str(adversarial_index) + '_poison_pattern']

        # 根据图像类型决定如何应用毒化模式
        if self.params['type'] == config.TYPE_CIFAR or self.params['type'] == config.TYPE_TINYIMAGENET:

            # 对于 CIFAR 或 TinyImageNet 类型的图像（彩色图像）
            for i in range(len(poison_patterns)):
                pos = poison_patterns[i]
                # 将指定位置的像素值设置为1（白色），对所有三个颜色通道生效
                image[0][pos[0]][pos[1]] = 1
                image[1][pos[0]][pos[1]] = 1
                image[2][pos[0]][pos[1]] = 1

        elif self.params['type'] == config.TYPE_MNIST:
            # 对于 MNIST 类型的图像（灰度图像）
            for i in range(len(poison_patterns)):
                pos = poison_patterns[i]
                # 将指定位置的像素值设置为1（白色）
                image[0][pos[0]][pos[1]] = 1

        # 返回修改后的图像
        return image

    def scan_least_frequent_label(self, targets):
        # 计算每个标签的出现次数
        label_counts = Counter()
        for label in targets:
            label_counts[int(label)] += 1

        # 获取所有标签的计数，并按计数升序排列
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1])
        # 初始化一个变量来存储找到的最少数据量的合适标签
        least_frequent_valid_label = None

        # 遍历排序后的标签列表，找到符合条件的最少数据量的标签
        for label, count in sorted_labels:
            if count > self.params['poisoning_per_batch']:
                least_frequent_valid_label = label
                break  # 找到第一个符合条件的最小标签后即停止搜索

        return least_frequent_valid_label

    def extract_real_features(self, real_image_classes, feature_extractor, device):
        real_features_per_class = []
        transform = transforms.ToTensor()

        for class_path in real_image_classes:
            class_features = []

            # 检查类别文件夹是否存在以及是否包含图像
            if not os.path.exists(class_path) or len(os.listdir(class_path)) == 0:
                print(f"警告: 类别文件夹 {class_path} 不存在或为空，跳过该类别。")
                real_features_per_class.append(torch.empty(0))  # 设置为空张量
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    # 加载图像并转换为张量
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(device)  # 图像转换为张量
                    features = self.extract_features(image_tensor, feature_extractor, device)  # 提取特征
                    features = features.view(-1)  # 转为 1D 向量
                    class_features.append(features)  # 添加到类别特征列表
                except Exception as e:
                    print(f"警告: 无法处理图像 {img_path}，错误: {e}")
                    continue

            # 如果类别中没有有效图像，设置为空张量
            if len(class_features) == 0:
                print(f"警告: 类别 {class_path} 中没有有效图像，设置为空。")
                real_features_per_class.append(torch.empty(0))  # 设置为空张量
            else:
                # 将类别特征列表转换为二维张量
                class_features = torch.stack(class_features)
                real_features_per_class.append(class_features)

        return real_features_per_class
    # def extract_real_features(self, real_image_classes, feature_extractor, device):
    #     real_features_per_class = []
    #     transform = transforms.ToTensor()
    #
    #     for class_path in real_image_classes:
    #         class_features = []
    #
    #         for img_name in os.listdir(class_path):
    #             img_path = os.path.join(class_path, img_name)
    #             image = Image.open(img_path).convert('RGB')
    #             image_tensor = transform(image).unsqueeze(0).to(device)  # 图像转换为张量
    #             features = self.extract_features(image_tensor, feature_extractor, device)  # 提取特征
    #             features = features.view(-1)  # 转为 NumPy 数组并展平为 1D 向量
    #             class_features.append(features)  # 添加到类别特征列表
    #
    #         # 将类别特征列表转换为二维 NumPy 数组
    #         class_features = torch.stack(class_features)  # 确保输出是二维张量
    #         real_features_per_class.append(class_features)
    #
    #     return real_features_per_class

    # 计算马氏距离
    def calculate_mahalanobis_distance(self, generated_features, real_features_per_class, device):
        min_distances = []
        min_classes = []

        # 确保生成特征和真实特征都在 GPU 上
        generated_features = generated_features.to(device)
        real_features_per_class = [features.to(device) for features in real_features_per_class]

        # 预计算每个类别的均值和协方差矩阵的逆
        precomputed_stats = []  # 缓存每个类别的 (均值, 协方差矩阵的逆)
        # for class_features in real_features_per_class:
        for idx, class_features in enumerate(real_features_per_class):
            # 跳过空张量的类别
            if class_features.nelement() == 0:  # 检查是否为空张量
                print(f"警告: 类别 {idx} 的特征为空，跳过该类别。")
                precomputed_stats.append(None)  # 用 None 占位，保持索引一致
                continue
            # 展平特征
            class_features_flat = class_features.view(class_features.size(0), -1)

            # 计算均值
            mean = class_features_flat.mean(dim=0)

            # 计算协方差矩阵
            cov = torch.cov(class_features_flat.T)

            # 添加小的正则化项，防止协方差矩阵不可逆
            cov += torch.eye(cov.size(0), device=device) * 1e-6

            # 计算协方差矩阵的伪逆
            cov_inv = torch.linalg.pinv(cov)

            # 将均值和协方差矩阵的逆保存起来
            precomputed_stats.append((mean, cov_inv))

        # 批量计算生成特征的马氏距离
        for gen_feature in generated_features:
            class_distances = []
            for mean, cov_inv in precomputed_stats:
                # 使用马氏距离公式计算
                diff = gen_feature - mean
                dist = torch.sqrt(torch.dot(diff, torch.matmul(cov_inv, diff)))
                class_distances.append(dist.item())  # 将 PyTorch 张量转换为标量

            # 找到最小距离及对应的类别
            min_distance = min(class_distances)
            min_class = class_distances.index(min_distance)
            min_distances.append(min_distance)
            min_classes.append(min_class)

        return min_distances, min_classes

    def load_resnet18_feature_extractor(self, device):
        model = ResNet18(name='Gan', created_time=None)
        loaded_params = torch.load((f"saved_models/{self.params['resumed_model_name']}"), map_location=device)
        model.load_state_dict(loaded_params['state_dict'])
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        feature_extractor.to(device)
        return model

    def extract_features(self, image, model, device):
        # 确保模型和输入数据都在同一设备上
        model = model.to(device)
        image = image.to(device)

        # 禁用梯度计算，加速推理
        with torch.no_grad():
            features = model(image)

        # 返回 PyTorch 张量，不转换为 NumPy 格式
        return features.squeeze()


if __name__ == '__main__':
    np.random.seed(1)
    with open(f'./utils/cifar_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, params=params_loaded,
                         name=params_loaded.get('name', 'mnist'))
    helper.load_data()

    pars = list(range(100))
    # show the data distribution among all participants.
    count_all = 0
    for par in pars:
        cifar_class_count = dict()
        for i in range(10):
            cifar_class_count[i] = 0
        count = 0
        _, data_iterator = helper.train_data[par]
        for batch_id, batch in enumerate(data_iterator):
            data, targets = batch
            for t in targets:
                cifar_class_count[t.item()] += 1
            count += len(targets)
        count_all += count
        print(par, cifar_class_count, count, max(zip(cifar_class_count.values(), cifar_class_count.keys())))

    print('avg', count_all * 1.0 / 100)

