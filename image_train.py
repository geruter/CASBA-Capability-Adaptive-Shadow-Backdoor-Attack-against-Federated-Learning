import threading

import utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import main
import test
import copy
import config
import psutil
import GPUtil
import os
import json
import statistics


# 定义ImageTrain函数，用于训练和可能的毒化攻击
def ImageTrain(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys, client_settings,
               combined_scores, sorted_clients_index):
    # 初始化一些字典和计数器
    epochs_submit_update_dict = dict()
    epochs_submit_update_dict_fool = dict()
    num_samples_dict = dict()
    current_number_of_adversaries = 0
    # 计算参与毒化的代理数量
    for temp_name in agent_name_keys:
        if temp_name in helper.params['adversary_list']:
            current_number_of_adversaries += 1

    # 遍历所有模型
    for model_id in range(helper.params['no_models']):
        epochs_local_update_list = []
        epochs_local_update_list_fool = []
        last_local_model = dict()
        client_grad = []  # 仅在aggr_epoch_interval=1时有效

        # 复制目标模型的参数作为上一次的本地模型状态
        for name, data in target_model.state_dict().items():
            last_local_model[name] = target_model.state_dict()[name].clone()

        agent_name_key = agent_name_keys[model_id]
        # 初始化该客户端的监控数据

        # 同步学习率和模型参数
        model = local_model
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()
        adversarial_index = -1
        localmodel_poison_epochs = helper.params['poison_epochs']
        client_attack_interval = None

        # 如果是毒化训练，设置毒化相关的参数
        if is_poison and agent_name_key in helper.params['adversary_list']:
            for temp_index in range(len(sorted_clients_index)):
                if int(agent_name_key) == sorted_clients_index[temp_index]:
                    adversarial_index = temp_index
                    localmodel_poison_epochs = helper.params[str(temp_index) + '_poison_epochs']
                    client_attack_interval = helper.params[str(temp_index) + '_poison_interval']
                    main.logger.info(f'poison local model {agent_name_key} index {adversarial_index} ')
                    break

            if len(helper.params['adversary_list']) == 1:
                adversarial_index = -1  # 全局模式
        start_time = time.time()
        all_time = 0

        # 训练周期
        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):
            relative_epoch = epoch - localmodel_poison_epochs[0]
            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)
            is_pruning = False
            # 如果当前周期需要执行毒化操作
            if is_poison and agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                # if (
                #         is_poison  # 当前客户端是恶意客户端
                #         and agent_name_key in helper.params['adversary_list']  # 客户端在恶意列表中
                #         and relative_epoch >= 0  # 当前轮次在 start_epoch 之后
                #         and relative_epoch % client_attack_interval == 0  # 满足攻击间隔
                #         # and epoch in localmodel_poison_epochs  # 当前轮次在毒化轮次列表中
                # ):
                main.logger.info('poison_now')

                learning_rate = client_settings[agent_name_key]['learning_rate']
                epochs = client_settings[agent_name_key]['epochs']
                batch_size = client_settings[agent_name_key]['batch_size']
                is_pruning = client_settings[agent_name_key]['is_pruning']
                train_time = combined_scores[agent_name_key]['training_time'] * epochs
                communication_time = combined_scores[agent_name_key]['total_communication_time']
                poisoning_per_batch = client_settings[agent_name_key]['poisoning_per_batch']
                all_time = train_time + communication_time
                poison_lr = helper.params['poison_lr']
                # internal_epoch_num = helper.params['internal_poison_epochs']
                internal_epoch_num = epochs
                step_lr = helper.params['poison_step_lr']

                # 设置毒化的优化器和学习率调度器
                # poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                #                                    momentum=helper.params['momentum'],
                #                                    weight_decay=helper.params['decay'])

                poison_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                                   momentum=helper.params['momentum'],
                                                   weight_decay=helper.params['decay'])
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * internal_epoch_num,
                                                                             0.8 * internal_epoch_num], gamma=0.1)
                temp_local_epoch = (epoch - 1) * internal_epoch_num
                for internal_epoch in range(1, internal_epoch_num + 1):
                    temp_local_epoch += 1
                    # _, data_iterator = helper.train_data[agent_name_key]
                    data_iterator = helper.create_train_loader_for_client(agent_name_key, batch_size)
                    print(batch_size)
                    poison_data_count = 0
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []

                    # 数据批次处理
                    for batch_id, batch in enumerate(data_iterator):
                        # 获取毒化的数据批次
                        data, targets, poison_num = helper.get_cgan_poison_batch(batch,
                                                                                 adversarial_index=adversarial_index,
                                                                                 evaluation=False,
                                                                                 current_model_id=int(agent_name_key),
                                                                                 poisoning_per_batch=poisoning_per_batch)
                        # data, targets, poison_num = helper.get_poison_batch(batch,
                        #                                                     adversarial_index=adversarial_index,
                        #                                                     evaluation=False)
                        poison_optimizer.zero_grad()
                        dataset_size += len(data)
                        poison_data_count += poison_num

                        # 前向传播和损失计算
                        output = model(data)
                        class_loss = nn.functional.cross_entropy(output, targets)
                        distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                        loss = helper.params['alpha_loss'] * class_loss + (
                                1 - helper.params['alpha_loss']) * distance_loss
                        loss.backward()
                        # 梯度获取和更新
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        poison_optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # 获取最大对数概率的索引
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                        # 如果是第一个批次，记录结束后的资源使用情况并计算差值
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'

                        # 获取显存信息
                        gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024 ** 2  # 转为 MB
                        gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024 ** 2  # 转为 MB
                        print(str(adversarial_index) + "  gpu_memory_allocated  " + str(
                            gpu_memory_allocated) + "  gpu_memory_reserved  " + str(gpu_memory_reserved))
                        torch.cuda.empty_cache()

                        # 距离跟踪和可视化
                        if helper.params["batch_track_distance"]:
                            temp_data_len = len(data_iterator)
                            distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                            dis2global_list.append(distance_to_global_model)
                            model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                           data_len=temp_data_len,
                                                           batch=batch_id,
                                                           distance_to_global_model=distance_to_global_model,
                                                           eid=helper.params['environment_name'],
                                                           name=str(agent_name_key), is_poisoned=True)

                    # 学习率调整和日志记录
                    if step_lr:
                        scheduler.step()
                        main.logger.info(f'Current lr: {scheduler.get_last_lr()}')

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main.logger.info(
                        '___PoisonTrain {} ,  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format(model.name, epoch,
                                                                                        agent_name_key,
                                                                                        internal_epoch,
                                                                                        total_l, correct, dataset_size,
                                                                                        acc, poison_data_count))
                    csv_record.train_result.append(
                        [agent_name_key, temp_local_epoch,
                         epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])
                    # 训练过程可视化
                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch,
                                        acc, loss=total_l, eid=helper.params['environment_name'], is_poisoned=True,
                                        name=str(agent_name_key))
                    num_samples_dict[agent_name_key] = dataset_size
                    if helper.params["batch_track_distance"]:
                        main.logger.info(
                            f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                            f'Distance to the global model: {dis2global_list}. ')

                # 毒化周期结束后的日志
                main.logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
                main.logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                                 f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

                # 如果非基线模型，进行模型测试和参数调整
                if not helper.params['baseline']:
                    main.logger.info(f'will scale.')
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                                   model=model, is_poison=False,
                                                                                   visualize=False,
                                                                                   agent_name_key=agent_name_key)
                    csv_record.test_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          is_poison=True,
                                                                                          visualize=False,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    clip_rate = helper.params['scale_weights_poison']
                    main.logger.info(f"Scaling by  {clip_rate}")
                    for key, value in model.state_dict().items():
                        target_value = last_local_model[key]
                        new_value = target_value + (value - target_value) * clip_rate
                        model.state_dict()[key].copy_(new_value)
                    distance = helper.model_dist_norm(model, target_params_variables)
                    main.logger.info(
                        f'Scaled Norm after poisoning: '
                        f'{helper.model_global_norm(model)}, distance: {distance}')
                    csv_record.scale_temp_one_row.append(epoch)
                    csv_record.scale_temp_one_row.append(round(distance, 4))
                    if helper.params["batch_track_distance"]:
                        temp_data_len = len(helper.train_data[agent_name_key][1])
                        model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                       data_len=temp_data_len,
                                                       batch=temp_data_len - 1,
                                                       distance_to_global_model=distance,
                                                       eid=helper.params['environment_name'],
                                                       name=str(agent_name_key), is_poisoned=True)

                distance = helper.model_dist_norm(model, target_params_variables)
                main.logger.info(f"Total norm for {current_number_of_adversaries} "
                                 f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

            else:
                # 正常训练周期
                temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
                for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                    temp_local_epoch += 1

                    _, data_iterator = helper.train_data[agent_name_key]
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []
                    for batch_id, batch in enumerate(data_iterator):

                        optimizer.zero_grad()
                        data, targets = helper.get_batch(data_iterator, batch, evaluation=False)

                        dataset_size += len(data)
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)
                        loss.backward()

                        # 梯度获取和更新
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # 获取最大对数概率的索引
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                        # 可视化训练批次损失
                        if helper.params["vis_train_batch_loss"]:
                            cur_loss = loss.data
                            temp_data_len = len(data_iterator)
                            model.train_batch_vis(vis=main.vis,
                                                  epoch=temp_local_epoch,
                                                  data_len=temp_data_len,
                                                  batch=batch_id,
                                                  loss=cur_loss,
                                                  eid=helper.params['environment_name'],
                                                  name=str(agent_name_key), win='train_batch_loss', is_poisoned=False)
                        # 距离跟踪和可视化
                        if helper.params["batch_track_distance"]:
                            temp_data_len = len(data_iterator)
                            distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                            dis2global_list.append(distance_to_global_model)
                            model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                           data_len=temp_data_len,
                                                           batch=batch_id,
                                                           distance_to_global_model=distance_to_global_model,
                                                           eid=helper.params['environment_name'],
                                                           name=str(agent_name_key), is_poisoned=False)

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main.logger.info(
                        '___Train {},  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, agent_name_key, internal_epoch,
                                                           total_l, correct, dataset_size,
                                                           acc))
                    csv_record.train_result.append([agent_name_key, temp_local_epoch,
                                                    epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

                    # 训练过程可视化
                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch,
                                        acc, loss=total_l, eid=helper.params['environment_name'], is_poisoned=False,
                                        name=str(agent_name_key))
                    num_samples_dict[agent_name_key] = dataset_size

                    # 距离跟踪和日志记录
                    if helper.params["batch_track_distance"]:
                        main.logger.info(
                            f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                            f'Distance to the global model: {dis2global_list}. ')

                # 测试本地模型在内部周期结束后的表现
                epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                               model=model, is_poison=False,
                                                                               visualize=True,
                                                                               agent_name_key=agent_name_key)
                csv_record.test_result.append([agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

            # 如果是毒化状态，进行毒化测试
            if is_poison:
                if agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          is_poison=True,
                                                                                          visualize=True,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                # 测试本地触发器
                if agent_name_key in helper.params['adversary_list']:
                    if helper.params['vis_trigger_split_test']:
                        model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                     eid=helper.params['environment_name'],
                                                     name=str(agent_name_key) + "_combine")

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
                        test.Mytest_poison_agent_trigger(helper=helper, model=model, agent_name_key=agent_name_key)
                    csv_record.poisontriggertest_result.append(
                        [agent_name_key, str(agent_name_key) + "_trigger", "", epoch, epoch_loss,
                         epoch_acc, epoch_corret, epoch_total])
                    if helper.params['vis_trigger_split_test']:
                        model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                     eid=helper.params['environment_name'],
                                                     name=str(agent_name_key) + "_trigger")

            # 更新模型权重
            local_model_update_dict = dict()
            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                if is_pruning:
                    update_difference = data - last_local_model[name]
                    # 计算更新差值的绝对值
                    abs_difference = torch.abs(update_difference)
                    # 创建一个掩码，其中小于0.05的差值对应的位置为True
                    prune_mask = abs_difference < 0.05
                    # 使用掩码设置小于0.05的更新为0，其他的保持原更新值
                    local_model_update_dict[name] = torch.where(prune_mask, torch.zeros_like(data), update_difference)
                else:
                    local_model_update_dict[name] = (data - last_local_model[name])
            #
            #     last_local_model[name] = copy.deepcopy(data)
            #

            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                local_model_update_dict[name] = (data - last_local_model[name])
                last_local_model[name] = copy.deepcopy(data)

            if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                epochs_local_update_list.append(client_grad)
                epochs_local_update_list_fool.append(local_model_update_dict)
            else:
                epochs_local_update_list.append(local_model_update_dict)

        epochs_submit_update_dict[agent_name_key] = epochs_local_update_list
        epochs_submit_update_dict_fool[agent_name_key] = epochs_local_update_list_fool

        end_time = time.time()
        dis_time = end_time - start_time + all_time
        training_time = end_time - start_time

    return epochs_submit_update_dict, epochs_submit_update_dict_fool, num_samples_dict
