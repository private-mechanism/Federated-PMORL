# from __future__ import absolute_import, division, print_function
# import argparse
# import pickle
# import numpy as np
# import torch
# from utils.monitor import Monitor
# from envs.mo_env import MultiObjectiveEnv
# from torch.autograd import Variable
# from collections import namedtuple
# import collections
# import copy
# from tqdm.contrib.concurrent import thread_map,process_map
# import os
#
# from collections import deque
# import multiprocessing
# import torch
# import torch.distributed as dist
#
# parser = argparse.ArgumentParser(description='MORL')
# # CONFIG
# parser.add_argument('--env-name', default='dst', metavar='dst',
#                     help='environment to train on: dst | ft | ft5 | ft7')
# parser.add_argument('--method', default='crl-naive', metavar='METHODS',
#                     help='methods: crl-naive | crl-envelope | crl-energy')
# parser.add_argument('--model', default='linear', metavar='MODELS',
#                     help='linear | cnn | cnn + lstm')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
#                     help='gamma for infinite horizonal MDPs')
#
#
# # TRAINING
# parser.add_argument('--sample_ratio', type=float, default=0.1, metavar='Sample_ratio',
#                     help='the sample ratio of clients in each round')
# parser.add_argument('--round', type=int, default=20, metavar='Round',
#                     help='the federated training round')
# parser.add_argument('--client_num', type=int, default=100, metavar='Client_num',
#                     help='the total number of the clients')
# parser.add_argument('--eps_per_round', type=int, default=1, metavar='Eps_per_round',
#                     help='the episode number for each client in each round')
# parser.add_argument('--dp', default=False, metavar='DP',
#                     help='whether to add dp noises into the modsel')
#
# #并行化
# # Process rank x represent client id from (x-1)*10 - (x-1)*10 +10
# # e.g. rank 5 <--> client 40-50
# parser.add_argument('--rank', default=10, metavar='Rank',
#                     help='the rank of the process')
# #param of pmorl
# parser.add_argument('--mem-size', type=int, default=4000, metavar='M',
#                     help='max size of the replay memory')
# parser.add_argument('--batch-size', type=int, default=25, metavar='B',
#                     help='batch size')
# parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
#                     help='learning rate')
# parser.add_argument('--epsilon', type=float, default=0.5, metavar='EPS',
#                     help='epsilon greedy exploration')
# parser.add_argument('--epsilon-decay', default=False, action='store_true',
#                     help='linear epsilon decay to zero')
# parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
#                     help='number of sampled weights per iteration')
# parser.add_argument('--episode-num', type=int, default=2000, metavar='EN',
#                     help='number of episodes for training')
# parser.add_argument('--optimizer', default='Adam', metavar='OPT',
#                     help='optimizer: Adam | RMSprop')
# parser.add_argument('--update-freq', type=int, default=100, metavar='OPT',
#                     help='optimizer: Adam | RMSprop')
# parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
#                     help='(initial) beta for evelope algorithm, default = 0.01')
# parser.add_argument('--homotopy', default=True, action='store_true',
#                     help='use homotopy optimization method')
# # LOG & SAVING
# parser.add_argument('--serialize', default=False, action='store_true',
#                     help='serialize a model')
# parser.add_argument('--save', default='crl/naive/saved/', metavar='SAVE',
#                     help='path for saving trained models')
# parser.add_argument('--actor_name', default='', metavar='actor_name',
#                     help='specify a name for saving the actor model')
# parser.add_argument('--critic_name', default='', metavar='critic_name',
#                     help='specify a name for saving the critic model')
# parser.add_argument('--log', default='crl/naive/logs/', metavar='LOG',
#                     help='path for recording training informtion')
#
#
# FloatTensor = torch.FloatTensor
# LongTensor = torch.LongTensor
# ByteTensor = torch.ByteTensor
# Tensor = FloatTensor
#
# class client_local(object):
#     def __init__(self,id,preference,env):
#         critic, actor = get_new_model_1(args.model, state_size, action_size, reward_size)
#         self.id=id
#         self.preference=preference
#         self.agent=MetaAgent(copy.deepcopy(critic), copy.deepcopy(actor), args, is_train=True)
#         self.env=copy.deepcopy(env)
#
#     def local_train(self,args):
#         # client_.agent.critic.load_state_dict(torch.load('global_model'))
#         ##########在本地训练多个episode
#         for num_eps in range(args.eps_per_round):
#             terminal = False
#             self.env.reset()
#             loss = 0
#             cnt = 0
#             tot_reward = 0
#             probe = None
#             if args.env_name == "dst":
#                 probe = FloatTensor([0.8, 0.2])
#             elif args.env_name in ['ft', 'ft5', 'ft7']:
#                 probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
#             while not terminal:
#                 state = self.env.observe()
#                 action = self.agent.act(state, self.preference)
#                 next_state, reward, terminal = self.env.step(action)
#                 self.agent.memorize(state, action, next_state, reward, terminal)
#                 self.agent.actor, self.agent.critic, critic_loss = self.agent.learn()
#                 loss += critic_loss
#                 if cnt > 100:
#                     terminal = True
#                     self.agent.reset()
#                 tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
#                 cnt = cnt + 1
#         # print('client',client_.id,'is training !',  'His preference:', client_.preference, ', length of trans mem:',
#         #       len(client_.agent.trans_mem),', example of trans:',[
#         #           (client_.agent.trans_mem[i].s, client_.agent.trans_mem[i].a, client_.agent.trans_mem[i].s_,
#         #       client_.agent.trans_mem[i].r, client_.agent.trans_mem[i].d) for i in range(5)])
#         print('This client is training:', self.id, 'with preference:', self.preference, 'length of trans mem:',
#               len(self.agent.trans_mem))
#         # return client_
#
# def local_train(client):
#     client.local_train(args)
#     return client
#
#
# def Initialize(args,env):
#     # 建立模型聚合中心，用于聚合本地模型，并进行算法验证和指标输出
#     # critic, actor = get_new_model_1(args.model, state_size, action_size, reward_size)
#     # agent = MetaAgent(critic, actor, args, is_train=True)
#     w_kept = torch.randn(reward_size)
#     preference = (torch.abs(w_kept) / torch.norm(w_kept, p=1)).type(FloatTensor)
#     # global_agent = MetaAgent(critic, actor, args, is_train=True)
#     # global_client = client_local(id=1000, preference=preference, agent=global_agent)
#     global_client=client_local(id=1000,preference=preference,env=env)
#
#     # 建立100个用户，随机定义用户的偏好，并初始化和环境交互的agent
#     client_list=[]
#     for id in range(100):
#         w_kept = torch.randn(reward_size)
#         preference = (torch.abs(w_kept) / torch.norm(w_kept, p=1)).type(FloatTensor)
#         # agent_ = MetaAgent(copy.deepcopy(critic), copy.deepcopy(actor), args, is_train=True)
#         client_ = client_local(id=id, preference=preference,env=env)
#         client_list.append(client_)
#     return global_client, client_list
#
#
# ###########初始化全局模型和本地模型
# # def Initialize(args):
# #     # 建立模型聚合中心，用于聚合本地模型，并进行算法验证和指标输出
# #     critic, actor = get_new_model_1(args.model, state_size, action_size, reward_size)
# #     agent = MetaAgent(critic, actor, args, is_train=True)
# #     w_kept = torch.randn(agent.critic.reward_size)
# #     preference = (torch.abs(w_kept) / torch.norm(w_kept, p=1)).type(FloatTensor)
# #     global_agent = MetaAgent(critic, actor, args, is_train=True)
# #     global_client = client_local(id=1000, preference=preference, agent=global_agent)
# #     # global_client=client_local(id=1000,preference=preference)
# #
# #     # 建立100个用户，随机定义用户的偏好，并初始化和环境交互的agent
# #     client_list=[]
# #     for id in range(100):
# #         w_kept = torch.randn(agent.critic.reward_size)
# #         preference = (torch.abs(w_kept) / torch.norm(w_kept, p=1)).type(FloatTensor)
# #         agent_ = MetaAgent(copy.deepcopy(critic), copy.deepcopy(actor), args, is_train=True)
# #         client_ = client_local(id=id, preference=preference, agent=agent_)
# #         client_list.append(client_)
# #     return global_client, client_list
#
# #############针对特定的client_进行本地训练，并输出本地更新后的client
# # def local_train(client_,rank,size):
# #     # client_.agent.critic.load_state_dict(torch.load('global_model'))
# #     ##########在本地训练多个episode
# #     for num_eps in range(args.eps_per_round):
# #         terminal = False
# #         local_env=copy.deepcopy(env)
# #         local_env.reset()
# #         loss = 0
# #         cnt = 0
# #         tot_reward = 0
# #         probe = None
# #         if args.env_name == "dst":
# #             probe = FloatTensor([0.8, 0.2])
# #         elif args.env_name in ['ft', 'ft5', 'ft7']:
# #             probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
# #         while not terminal:
# #             state = local_env.observe()
# #             action = client_.agent.act(state, client_.preference)
# #             next_state, reward, terminal = local_env.step(action)
# #             client_.agent.memorize(state, action, next_state, reward, terminal)
# #             client_.agent.actor, client_.agent.critic, critic_loss = client_.agent.learn()
# #             loss += critic_loss
# #             if cnt > 100:
# #                 terminal = True
# #                 client_.agent.reset()
# #             tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
# #             cnt = cnt + 1
# #     # print('client',client_.id,'is training !',  'His preference:', client_.preference, ', length of trans mem:',
# #     #       len(client_.agent.trans_mem),', example of trans:',[
# #     #           (client_.agent.trans_mem[i].s, client_.agent.trans_mem[i].a, client_.agent.trans_mem[i].s_,
# #     #       client_.agent.trans_mem[i].r, client_.agent.trans_mem[i].d) for i in range(5)])
# #     print('This client is training:', client_.id, 'with preference:', client_.preference, 'length of trans mem:',
# #           len(client_.agent.trans_mem))
# #     return client_
#
# # def Aggregation(model_parameters):
# #################初始化进程
# def init_processes(rank, size, fn, backend='gloo'):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     dist.init_process_group(backend, rank=rank, world_size=size)
#
#     fn(rank, size)
#
# def fed_avg(model):
#     """ Gradient averaging. """
#     size = float(dist.get_world_size())
#     # print(size)
#     for name, parameters in model.named_parameters():
#         # print('name:',name,parameters.data)
#         dist.reduce(parameters.data, op=dist.ReduceOp.SUM,group=0)
#         parameters.data /= size
#
# # def aggregation(model):
# #     for name, parameters in model.named_parameters():
# #         dist.broadcast(parameters.data, src=0, group=id_group)
#         # parameters /= size
#
# def run(rank,size):
#     # Process rank x represent client id from (x-1)*10 - (x-1)*10 +10
#     # e.g. rank 5 <--> client 40-50
#     # client_local=[]
#     # torch.manual_seed(1234)
#     # agent = MetaAgent(copy.deepcopy(critic), copy.deepcopy(actor), args, is_train=True)
#     # sampled_id_list=[]
#     # for client in sampled_client_list:
#     client=client_list[rank]  ##############在每一个进程里面采样一个client，保证每个进程采样的client不重复
#     #################对当前的client进行本地训练
#     print('11111111111111111111111111111111111111',rank)
#     # print(client_list[])
#     client.local_train(args)
#
#     print('2222222222222222222222222222222222222222')
#     fed_avg(client.agent.critic)
#
#     #####reduce的dst为rank=0;对所有进程中的client.agent.critic进行reduce并发送到rank=0的进程
#     # aggregation(client.agent.critic, sampled_id_list)
#
#
#
#
#
#
# #########利用torch.dist实现的分布式联邦训练
# # group = dist.new_group([i for i in range(500)])
# #
# # op=dist
# #
# # reduce(tensor, dst, op, group)#####reduce的dst为rank=0;
# #
# # broadcast(tensor, src, group)#####broadcast的dst为rank=group;
#
#
# # def aggregate(global_client):
# #     size = float(dist.get_world_size())
# #     fed_state_dict = collections.OrderedDict()
# #     key_list=list(global_client.agent.critic.state_dict().keys())
# #     for key in key_list:
# #         dist.all_reduce()
# #
# #     for param in global_client.agent.critic.parameters():
# #         dist.all_reduce(param, op=dist.reduce_op.SUM, group=0)
# #         param /= size
# #
# #
# #     worker_state_dict = [x.state_dict() for x in model_list]
# #     weight_keys = list(worker_state_dict[0].keys())
# #     fed_state_dict = collections.OrderedDict()
# #     for key in weight_keys:
# #         key_sum = 0
# #         for i in range(len(model_list)):
# #             key_sum = key_sum + worker_state_dict[i][key]
# #         fed_state_dict[key] = key_sum / len(model_list)
#
# ##############联邦训练流程:选取client, 每个client进行本地更新，全局聚合，重复直至收敛
# def fed_train(args,global_client,client_list):
#     act1 = []
#     act2 = []
#     for round_num in range(args.round):
#         print([len(client_list[i].agent.trans_mem) for i in range(args.client_num)])
#         ##根据设定的采样比例sample_ratio进行client的随机采样，如每次选取10%的用户参与本轮训练
#         id_list = np.random.choice([i for i in range(args.client_num)], size=int(args.client_num * args.sample_ratio),replace=False)
#         print('###############################The following clients will participate this round of training:', id_list)
#         Client_List=[client_list[id] for id in id_list]
#         # print('0', Client_List[0].agent.actor_.forward(FloatTensor([0, 1]), FloatTensor([0.8, 0.2])))
#         # print('1', Client_List[1].agent.actor_.forward(FloatTensor([0, 1]), FloatTensor([0.8, 0.2])))
#         # print('2', Client_List[2].agent.actor_.forward(FloatTensor([0, 1]), FloatTensor([0.8, 0.2])))
#         ##所有参与本轮训练的client，critic模型都与全局模型一致
#         for client_ in Client_List:
#             client_.agent.critic.load_state_dict(global_client.agent.critic.state_dict())
#             client_.agent.critic_.load_state_dict(global_client.agent.critic_.state_dict())
#         ##############基于for循环实现，多个client依次进行本地训练，再进行模型聚合，训练过程可以正常进行，且多个client有自己的trans_mem支撑训练
#         # output_list=[]
#         # for client in Client_List:
#         #     output_list.append(local_train(client))
#         ##############调用 thread_map 和 process_map分别实现多线程并行以及多进程并行，多线程并行可以正常运行，但多进程导致多个不同client的trans_mem一模一样
#         # torch.multiprocessing.set_sharing_strategy('file_system')
#         # output_list = process_map(local_train, Client_List)
#         output_list = thread_map(local_train, Client_List)
#         ###############调用multiprocessing.pool针对每个client进行并行化的本地模型训练，结果仍然存在多个client trans_mem一致的情况
#         # torch.multiprocessing.set_sharing_strategy('file_system')
#         # pool = torch.multiprocessing.Pool(processes = 10)
#         # output_list = pool.map(local_train, Client_List)
#         # pool.close()
#         # pool.join()
#         ###############同样调用multiprocessing库，但是采用pool.apply_async函数实现并行化
#         # torch.multiprocessing.set_sharing_strategy('file_system')
#         # pool = torch.multiprocessing.Pool(processes=10)
#         # result_list = []
#         # for client_ in Client_List:
#         #     result_list.append(pool.apply_async(local_train, (client_,)))
#         # pool.close()
#         # pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
#         # output_list = []
#         # for res in result_list:
#         #     temp_list = res.get()
#         #     output_list.append(temp_list)
#         #############基于pytorch.dist实现的多进程
#         # torch.multiprocessing.set_sharing_strategy('file_system')
#         # pool = torch.multiprocessing.Pool(processes = 10)
#         # output_list = pool.map(local_train, Client_List)
#         # pool.close()
#         # pool.join()
#         #############尝试在多个进程中加锁，防止内存读写的混乱，不奏效
#         # q = multiprocessing.Queue()
#         # process_lock = multiprocessing.Lock()
#         # jobs = []
#         # for i_arges in Client_List:
#         #     p = multiprocessing.Process(target=local_train, args=(i_arges, q))
#         #     jobs.append(p)
#         #     p.start()
#         # for i_p in jobs:
#         #     i_p.join()
#         # output_list1 = [q.get() for j in jobs]
#         # output_list = [pickle.loads(obj_j) for obj_j in output_list1]
#         # print('1111111111111111111111111111111111',output_list)
#             # end_time_list.append(time_list[1])
#             # log_time_list.append(time_list[2])
#             # log_time_list.append(time_list[2])
#         for clien in output_list:
#             client_list[clien.id]=copy.deepcopy(clien)
#         #     print(len(clien.agent.trans_mem))
#         # print()
#         critic_model_temp_list =[client_temp.agent.critic_ for client_temp in output_list]
#         critic_model_list = [copy.deepcopy(model) for model in critic_model_temp_list]
#         criti_model_temp_list = [client_temp.agent.critic for client_temp in output_list]
#         criti_model_list = [copy.deepcopy(model) for model in criti_model_temp_list]
#         ############计算模型更新并添加差分隐私噪声
#         if args.dp:
#             ###撰写差分隐私模块
#             pass
#         # weight_keys = list(model_temp.keys())
#         # update_state_dict = collections.OrderedDict()
#         # for key in weight_keys:
#         #     update_state_dict[key]=model_temp[key]-last_global[key]
#         # model_temp.load_state_dict
#         ########添加高斯噪声
#         # key_update+=noise
#         # for i in range(len(model_list)):
#         #     key_sum = key_sum + worker_state_dict[i][key]
#         # param_list.append(critic.state_dict())
#         ###########进行模型聚合
#         else:
#             print('model is aggregating!')
#             worker_state_dict = [x.state_dict() for x in critic_model_list]
#             weight_keys = list(worker_state_dict[0].keys())
#             fed_state_dict = collections.OrderedDict()
#             for key in weight_keys:
#                 key_sum = 0
#                 for i in range(len(critic_model_list)):
#                     key_sum = key_sum + worker_state_dict[i][key]
#                 fed_state_dict[key] = key_sum / len(critic_model_list)
#             #### update fed weights to fl model
#             global_client.agent.critic_.load_state_dict(fed_state_dict)
#
#             worker_state_dict = [x.state_dict() for x in criti_model_list]
#             weight_keys = list(worker_state_dict[0].keys())
#             fed_state_dict = collections.OrderedDict()
#             for key in weight_keys:
#                 key_sum = 0
#                 for i in range(len(criti_model_list)):
#                     key_sum = key_sum + worker_state_dict[i][key]
#                 fed_state_dict[key] = key_sum / len(criti_model_list)
#             #### update fed weights to fl model
#             global_client.agent.critic.load_state_dict(fed_state_dict)
#         # torch.save(global_client.agent.critic_.state_dict(),'global_model')
#         #############计算模型的评估指标
#         print('model is evaluating!')
#         probe = None
#         if args.env_name == "dst":
#             probe = FloatTensor([0.8, 0.2])
#         elif args.env_name in ['ft', 'ft5', 'ft7']:
#             probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
#         _, q = global_client.agent.predict(probe)
#         if args.env_name == "dst":
#             act_1 = q[0, 3]
#             act_2 = q[0, 1]
#         elif args.env_name in ['ft', 'ft5', 'ft7']:
#             act_1 = q[0, 1]
#             act_2 = q[0, 0]
#         if args.method == "crl-naive":
#             act_1 = act_1.data.cpu()
#             act_2 = act_2.data.cpu()
#         elif args.method == "crl-envelope":
#             act_1 = probe.dot(act_1.data)
#             act_2 = probe.dot(act_2.data)
#         elif args.method == "crl-energy":
#             act_1 = probe.dot(act_1.data)
#             act_2 = probe.dot(act_2.data)
#         act1.append(act_1.item())
#         act2.append(act_2.item())
#         # reward_list.append(tot_reward)
#         if (round_num + 1) % args.round == 0:
#             with open('copy_multi_fed_train_pmorl_dst.txt', 'a') as f:  # 设置文件对象
#                 f.write(
#                     'fed_PMORL_dst_sample_ratio#######################################################################################')
#                 f.write('\n')
#                 f.write('client_num=')
#                 f.write(str(args.client_num))
#                 f.write(',sample_ratio=')
#                 f.write(str(args.sample_ratio))
#                 f.write(',eps_per_round=')
#                 f.write(str(args.eps_per_round))
#                 f.write(',batch_size_per_client=')
#                 f.write(str(args.batch_size))
#                 f.write(',dp=')
#                 f.write(str(args.dp))
#                 f.write(':\n')
#                 f.write(str(act1))
#                 f.write('\n')
#                 f.write(str(act2))
#                 f.write('\n')
#                 # f.write('return: \t')
#                 # f.write(str(return_list))
#                 # f.write('\n')
#         print("end of round %d, the Q is %0.2f | %0.2f" % (
#             round_num,
#             act_1,
#             act_2,
#         ))
#         # monitor.update(round_num,1,
#         #                act_1,
#         #                act_2)
#         # # if num_eps+1 % 100 == 0:
#         # 	agent.save(args.save, args.model+args.name+"_tmp_{}".format(number))
#     global_client.agent.save(global_client.agent.actor, args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.actor_name))
#     global_client.agent.save(global_client.agent.critic, args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.critic_name))
#
# # def main():
# if __name__ == '__main__':
#
#     args = parser.parse_args()
#     # setup the environment
#     env = MultiObjectiveEnv(args.env_name)
#     # get state / action / reward sizes
#     state_size = len(env.state_spec)
#     action_size = env.action_spec[2][1] - env.action_spec[2][0]
#     reward_size = len(env.reward_spec)
#
#     # generate an agent for initial training
#     agent = None
#     if args.method == 'crl-naive':
#         from crl.naive.meta import MetaAgent
#         from crl.naive.models import get_new_model
#     elif args.method == 'crl-envelope':
#         from crl.envelope.fed_pmorl import MetaAgent
#         from crl.envelope.models import get_new_model_1
#     elif args.method == 'crl-energy':
#         from crl.energy.meta import MetaAgent
#         from crl.energy.models import get_new_model
#
#     if args.serialize:
#         model = torch.load("{}{}.pkl".format(args.save,
#                                              "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
#     else:
#         critic, actor = get_new_model_1(args.model, state_size, action_size, reward_size)
#
#     # client_id_list = [
#     #     i for i in range((args.rank - 1) * 10, (args.rank - 1) * 10 + 10)
#     # ]
#     preference_list = []
#     for i in range(args.client_num):
#         w_kept = torch.randn(reward_size)
#         preference = (torch.abs(w_kept) / torch.norm(w_kept, p=1)).type(FloatTensor)
#         preference_list.append(preference)
#     # client_list = [
#     #     client_local(id=i,preference=preference_list[i],env=env) for i in range((args.rank - 1) * 10, (args.rank - 1) * 10 + 10)
#     # ]
#     # client_list = [
#     #     client_local(id=i, preference=preference_list[i], env=env) for i in
#     #     range(args.client_num)
#     # ]
#     ##定义本地的client
#     # client_local = namedtuple('client_local', ['id', 'preference', 'agent'])
#     #初始化全局和本地模型
#     global_client, client_list=Initialize(args,env)
#     ##联邦训练
#     fed_train(args, global_client, client_list)
#     #多进程
#     # size = int(args.client_num)
#     # # size=args.client_num
#     # for round_num in range(args.round):
#     #     sampled_id_list = np.random.choice([i for i in range(args.client_num)],
#     #                                        size=int(args.client_num * args.sample_ratio), replace=False)
#     #     sampled_client_list = [client_list[i] for i in sampled_id_list]
#     #     processes = []
#     #     for rank in range(size):
#     #         p = torch.multiprocessing.Process(target=init_processes, args=(rank, size, run))
#     #         p.start()
#     #         processes.append(p)
#     #     for p in processes:
#     #         p.join()
#     #         p.close()
#
#     ######################################################################
from __future__ import absolute_import, division, print_function
import os

import sys
sys.path.append('/home/zfy/workspace/MORL-master/synthetic/')

# from crl.envelope.meta import MetaAgent
import argparse
import pickle
import numpy as np
import torch
# from utils.monitor import Monitor
from envs.mo_env import MultiObjectiveEnv
from torch.autograd import Variable
from collections import namedtuple
import collections
import copy
from tqdm.contrib.concurrent import thread_map,process_map
import os

from collections import deque
import multiprocessing
import torch
import torch.distributed as dist



parser = argparse.ArgumentParser(description='MORL')
# CONFIG
parser.add_argument('--env-name', default='dst', metavar='dst',
                    help='environment to train on: dst | ft | ft5 | ft7')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
#######原始参数配置--env-name dst --method crl-envelope --model linear --client_num 100 --sample_ratio 0.1 --dp False --round 1000 --eps_per_round 1 --gamma  0.99 --mem-size 4000 --batch-size 25 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 2000 --optimizer Adam --save /home/zfy/workspace/MORL-master/synthetic/fed_model/pmorl/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --critic_name fed_pmorl_dst_sample_01

# TRAINING
parser.add_argument('--sample_ratio', type=float, default=0.1, metavar='Sample_ratio',
                    help='the sample ratio of clients in each round')
parser.add_argument('--round', type=int, default=1000, metavar='Round',
                    help='the federated training round')
parser.add_argument('--client_num', type=int, default=100, metavar='Client_num',
                    help='the total number of the clients')
parser.add_argument('--eps_per_round', type=int, default=1, metavar='Eps_per_round',
                    help='the episode number for each client in each round')
parser.add_argument('--dp', default=False, metavar='DP',
                    help='whether to add dp noises into the modsel')

#并行化
# Process rank x represent client id from (x-1)*10 - (x-1)*10 +10
# e.g. rank 5 <--> client 40-50
parser.add_argument('--rank', default=10, metavar='Rank',
                    help='the rank of the process')
#param of pmorl
parser.add_argument('--mem-size', type=int, default=4000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=25, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.5, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=2000, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=100, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.98, metavar='BETA',
                    help='(initial) beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=True, action='store_true',
                    help='use homotopy optimization method')
# LOG & SAVING
parser.add_argument('--serialize', default=False, action='store_true',
                    help='serialize a model')
parser.add_argument('--save', default='crl/naive/saved/', metavar='SAVE',
                    help='path for saving trained models')
parser.add_argument('--actor_name', default='', metavar='actor_name',
                    help='specify a name for saving the actor model')
parser.add_argument('--critic_name', default='', metavar='critic_name',
                    help='specify a name for saving the critic model')
parser.add_argument('--log', default='crl/naive/logs/', metavar='LOG',
                    help='path for recording training informtion')


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor

class client_local(object):
    def __init__(self,id,preference,env):
        critic, actor = get_new_model_1(args.model, state_size, action_size, reward_size)
        self.id=id
        self.preference=preference
        self.agent=MetaAgent(copy.deepcopy(critic), copy.deepcopy(actor), args, is_train=True)
        self.env=copy.deepcopy(env)

    def local_train(self,args):
        # client_.agent.critic.load_state_dict(torch.load('global_model'))
        ##########在本地训练多个episode
        for num_eps in range(args.eps_per_round):
            terminal = False
            self.env.reset()
            loss = 0
            cnt = 0
            tot_reward = 0
            probe = None
            if args.env_name == "dst":
                probe = FloatTensor([0.8, 0.2])
            elif args.env_name in ['ft', 'ft5', 'ft7']:
                probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
            while not terminal:
                state = self.env.observe()
                action = self.agent.act(state, self.preference)
                next_state, reward, terminal = self.env.step(action)
                self.agent.memorize(state, action, next_state, reward, terminal)
                self.agent.actor, self.agent.critic, critic_loss = self.agent.learn()
                loss += critic_loss
                if cnt > 100:
                    terminal = True
                    self.agent.reset()
                tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
                cnt = cnt + 1
        # print('client',client_.id,'is training !',  'His preference:', client_.preference, ', length of trans mem:',
        #       len(client_.agent.trans_mem),', example of trans:',[
        #           (client_.agent.trans_mem[i].s, client_.agent.trans_mem[i].a, client_.agent.trans_mem[i].s_,
        #       client_.agent.trans_mem[i].r, client_.agent.trans_mem[i].d) for i in range(5)])
        print('This client is training:', self.id, 'with preference:', self.preference, 'length of trans mem:',
              len(self.agent.trans_mem))
        # return client_

def local_train(client):
    client.local_train(args)
    return client


def Initialize(args,env):
    # 建立模型聚合中心，用于聚合本地模型，并进行算法验证和指标输出
    # critic, actor = get_new_model_1(args.model, state_size, action_size, reward_size)
    # agent = MetaAgent(critic, actor, args, is_train=True)
    w_kept = torch.randn(reward_size)
    preference = (torch.abs(w_kept) / torch.norm(w_kept, p=1)).type(FloatTensor)
    # global_agent = MetaAgent(critic, actor, args, is_train=True)
    # global_client = client_local(id=1000, preference=preference, agent=global_agent)
    global_client=client_local(id=1000,preference=preference,env=env)

    # 建立100个用户，随机定义用户的偏好，并初始化和环境交互的agent
    client_list=[]
    for id in range(args.client_num):
        w_kept = torch.randn(reward_size)
        preference = (torch.abs(w_kept) / torch.norm(w_kept, p=1)).type(FloatTensor)
        # agent_ = MetaAgent(copy.deepcopy(critic), copy.deepcopy(actor), args, is_train=True)
        client_ = client_local(id=id, preference=preference,env=env)
        client_list.append(client_)
    print(len(client_list))
    return global_client, client_list


###########初始化全局模型和本地模型
# def Initialize(args):
#     # 建立模型聚合中心，用于聚合本地模型，并进行算法验证和指标输出
#     critic, actor = get_new_model_1(args.model, state_size, action_size, reward_size)
#     agent = MetaAgent(critic, actor, args, is_train=True)
#     w_kept = torch.randn(agent.critic.reward_size)
#     preference = (torch.abs(w_kept) / torch.norm(w_kept, p=1)).type(FloatTensor)
#     global_agent = MetaAgent(critic, actor, args, is_train=True)
#     global_client = client_local(id=1000, preference=preference, agent=global_agent)
#     # global_client=client_local(id=1000,preference=preference)
#
#     # 建立100个用户，随机定义用户的偏好，并初始化和环境交互的agent
#     client_list=[]
#     for id in range(100):
#         w_kept = torch.randn(agent.critic.reward_size)
#         preference = (torch.abs(w_kept) / torch.norm(w_kept, p=1)).type(FloatTensor)
#         agent_ = MetaAgent(copy.deepcopy(critic), copy.deepcopy(actor), args, is_train=True)
#         client_ = client_local(id=id, preference=preference, agent=agent_)
#         client_list.append(client_)
#     return global_client, client_list

#############针对特定的client_进行本地训练，并输出本地更新后的client
# def local_train(client_,rank,size):
#     # client_.agent.critic.load_state_dict(torch.load('global_model'))
#     ##########在本地训练多个episode
#     for num_eps in range(args.eps_per_round):
#         terminal = False
#         local_env=copy.deepcopy(env)
#         local_env.reset()
#         loss = 0
#         cnt = 0
#         tot_reward = 0
#         probe = None
#         if args.env_name == "dst":
#             probe = FloatTensor([0.8, 0.2])
#         elif args.env_name in ['ft', 'ft5', 'ft7']:
#             probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
#         while not terminal:
#             state = local_env.observe()
#             action = client_.agent.act(state, client_.preference)
#             next_state, reward, terminal = local_env.step(action)
#             client_.agent.memorize(state, action, next_state, reward, terminal)
#             client_.agent.actor, client_.agent.critic, critic_loss = client_.agent.learn()
#             loss += critic_loss
#             if cnt > 100:
#                 terminal = True
#                 client_.agent.reset()
#             tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
#             cnt = cnt + 1
#     # print('client',client_.id,'is training !',  'His preference:', client_.preference, ', length of trans mem:',
#     #       len(client_.agent.trans_mem),', example of trans:',[
#     #           (client_.agent.trans_mem[i].s, client_.agent.trans_mem[i].a, client_.agent.trans_mem[i].s_,
#     #       client_.agent.trans_mem[i].r, client_.agent.trans_mem[i].d) for i in range(5)])
#     print('This client is training:', client_.id, 'with preference:', client_.preference, 'length of trans mem:',
#           len(client_.agent.trans_mem))
#     return client_

#########利用torch.dist实现的分布式联邦训练
# group = dist.new_group([i for i in range(500)])
#
# op=dist
#
# reduce(tensor, dst, op, group)#####reduce的dst为rank=0;
#
# broadcast(tensor, src, group)#####broadcast的dst为rank=group;


# def aggregate(global_client):
#     size = float(dist.get_world_size())
#     fed_state_dict = collections.OrderedDict()
#     key_list=list(global_client.agent.critic.state_dict().keys())
#     for key in key_list:
#         dist.all_reduce()
#
#     for param in global_client.agent.critic.parameters():
#         dist.all_reduce(param, op=dist.reduce_op.SUM, group=0)
#         param /= size
#
#
#     worker_state_dict = [x.state_dict() for x in model_list]
#     weight_keys = list(worker_state_dict[0].keys())
#     fed_state_dict = collections.OrderedDict()
#     for key in weight_keys:
#         key_sum = 0
#         for i in range(len(model_list)):
#             key_sum = key_sum + worker_state_dict[i][key]
#         fed_state_dict[key] = key_sum / len(model_list)
def model_aggregation(model_list):
    worker_state_dict = [x.state_dict() for x in model_list]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(model_list)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(model_list)
    return fed_state_dict

##############联邦训练流程:选取client, 每个client进行本地更新，全局聚合，重复直至收敛
def fed_train(args,global_client,client_list):
    act1 = []
    act2 = []
    for round_num in range(args.round):
        print([len(client_list[i].agent.trans_mem) for i in range(args.client_num)])
        ##根据设定的采样比例sample_ratio进行client的随机采样，如每次选取10%的用户参与本轮训练
        # id_list = np.random.choice([i for i in range(args.client_num)], size=int(args.client_num * args.sample_ratio),replace=False)
        id_list = np.random.choice([i for i in range(args.client_num)], size=10,
                                   replace=False)
        print('###############################The following clients will participate this round of training:', id_list)
        Client_List=[client_list[id] for id in id_list]
        # print('0', Client_List[0].agent.actor_.forward(FloatTensor([0, 1]), FloatTensor([0.8, 0.2])))
        # print('1', Client_List[1].agent.actor_.forward(FloatTensor([0, 1]), FloatTensor([0.8, 0.2])))
        # print('2', Client_List[2].agent.actor_.forward(FloatTensor([0, 1]), FloatTensor([0.8, 0.2])))
        ##所有参与本轮训练的client，critic模型都与全局模型一致
        for client_ in Client_List:
            client_.agent.critic.load_state_dict(global_client.agent.critic.state_dict())
            client_.agent.critic_.load_state_dict(global_client.agent.critic_.state_dict())
        ##############基于for循环实现，多个client依次进行本地训练，再进行模型聚合，训练过程可以正常进行，且多个client有自己的trans_mem支撑训练
        # output_list=[]
        # for client in Client_List:
        #     output_list.append(local_train(client))
        ##############调用 thread_map 和 process_map分别实现多线程并行以及多进程并行，多线程并行可以正常运行，但多进程导致多个不同client的trans_mem一模一样
        # torch.multiprocessing.set_sharing_strategy('file_system')
        # output_list = process_map(local_train, Client_List)
        output_list = thread_map(local_train, Client_List)
        ###############调用multiprocessing.pool针对每个client进行并行化的本地模型训练，结果仍然存在多个client trans_mem一致的情况
        # torch.multiprocessing.set_sharing_strategy('file_system')
        # pool = torch.multiprocessing.Pool(processes = 10)
        # output_list = pool.map(local_train, Client_List)
        # pool.close()
        # pool.join()
        ###############同样调用multiprocessing库，但是采用pool.apply_async函数实现并行化
        # torch.multiprocessing.set_sharing_strategy('file_system')
        # pool = torch.multiprocessing.Pool(processes=10)
        # result_list = []
        # for client_ in Client_List:
        #     result_list.append(pool.apply_async(local_train, (client_,)))
        # pool.close()
        # pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
        # output_list = []
        # for res in result_list:
        #     temp_list = res.get()
        #     output_list.append(temp_list)
        #############基于pytorch.dist实现的多进程
        # torch.multiprocessing.set_sharing_strategy('file_system')
        # pool = torch.multiprocessing.Pool(processes = 10)
        # output_list = pool.map(local_train, Client_List)
        # pool.close()
        # pool.join()
        #############尝试在多个进程中加锁，防止内存读写的混乱，不奏效
        # q = multiprocessing.Queue()
        # process_lock = multiprocessing.Lock()
        # jobs = []
        # for i_arges in Client_List:
        #     p = multiprocessing.Process(target=local_train, args=(i_arges, q))
        #     jobs.append(p)
        #     p.start()
        # for i_p in jobs:
        #     i_p.join()
        # output_list1 = [q.get() for j in jobs]
        # output_list = [pickle.loads(obj_j) for obj_j in output_list1]
        # print('1111111111111111111111111111111111',output_list)
            # end_time_list.append(time_list[1])
            # log_time_list.append(time_list[2])
            # log_time_list.append(time_list[2])
        for clien in output_list:
            client_list[clien.id]=copy.deepcopy(clien)
        #     print(len(clien.agent.trans_mem))
        # print()
        critic_model_temp_list =[client_temp.agent.critic_ for client_temp in output_list]
        critic_model_list = [copy.deepcopy(model) for model in critic_model_temp_list]
        criti_model_temp_list = [client_temp.agent.critic for client_temp in output_list]
        criti_model_list = [copy.deepcopy(model) for model in criti_model_temp_list]
        ############计算模型更新并添加差分隐私噪声
        if args.dp:
            ###撰写差分隐私模块
            # pass
            print('model is aggregating with DP noise!')
            worker_state_dict = [x.state_dict() for x in critic_model_list]
            weight_keys = list(worker_state_dict[0].keys())
            fed_state_dict = collections.OrderedDict()
            for key in weight_keys:
                key_sum = 0
                for i in range(len(critic_model_list)):
                    key_sum = key_sum + worker_state_dict[i][key]
                fed_state_dict[key] = key_sum / len(critic_model_list)
            #### update fed weights to fl model
            global_client.agent.critic_.load_state_dict(fed_state_dict)



            worker_state_dict = [x.state_dict() for x in criti_model_list]
            weight_keys = list(worker_state_dict[0].keys())
            fed_state_dict = collections.OrderedDict()
            for key in weight_keys:
                key_sum = 0
                for i in range(len(criti_model_list)):
                    key_sum = key_sum + worker_state_dict[i][key]
                fed_state_dict[key] = key_sum / len(criti_model_list)
            #### update fed weights to fl model
            global_client.agent.critic.load_state_dict(fed_state_dict)
            # weight_keys = list(model_temp.keys())
            # update_state_dict = collections.OrderedDict()
            # for key in weight_keys:
            #     update_state_dict[key]=model_temp[key]-last_global[key]
            # model_temp.load_state_dict
            # #######添加高斯噪声
            # key_update+=noise
            # for i in range(len(model_list)):
            #     key_sum = key_sum + worker_state_dict[i][key]
            # param_list.append(critic.state_dict())
        ###########进行模型聚合
        else:
            print('model is aggregating!')
            worker_state_dict = [x.state_dict() for x in critic_model_list]
            weight_keys = list(worker_state_dict[0].keys())
            fed_state_dict = collections.OrderedDict()
            for key in weight_keys:
                key_sum = 0
                for i in range(len(critic_model_list)):
                    key_sum = key_sum + worker_state_dict[i][key]
                fed_state_dict[key] = key_sum / len(critic_model_list)
            #### update fed weights to fl model
            global_client.agent.critic_.load_state_dict(fed_state_dict)

            worker_state_dict = [x.state_dict() for x in criti_model_list]
            weight_keys = list(worker_state_dict[0].keys())
            fed_state_dict = collections.OrderedDict()
            for key in weight_keys:
                key_sum = 0
                for i in range(len(criti_model_list)):
                    key_sum = key_sum + worker_state_dict[i][key]
                fed_state_dict[key] = key_sum / len(criti_model_list)
            #### update fed weights to fl model
            global_client.agent.critic.load_state_dict(fed_state_dict)
        # torch.save(global_client.agent.critic_.state_dict(),'global_model')
        #############计算模型的评估指标
        print('model is evaluating!')
        probe = None
        if args.env_name == "dst":
            probe = FloatTensor([0.8, 0.2])
        elif args.env_name in ['ft', 'ft5', 'ft7']:
            probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
        _, q = global_client.agent.predict(probe)
        if args.env_name == "dst":
            act_1 = q[0, 3]
            act_2 = q[0, 1]
        elif args.env_name in ['ft', 'ft5', 'ft7']:
            act_1 = q[0, 1]
            act_2 = q[0, 0]
        if args.method == "crl-naive":
            act_1 = act_1.data.cpu()
            act_2 = act_2.data.cpu()
        elif args.method == "crl-envelope":
            act_1 = probe.dot(act_1.data)
            act_2 = probe.dot(act_2.data)
        elif args.method == "crl-energy":
            act_1 = probe.dot(act_1.data)
            act_2 = probe.dot(act_2.data)
        act1.append(act_1.item())
        act2.append(act_2.item())
        # reward_list.append(tot_reward)
        if (round_num + 1) % 100 == 0:
            with open('copy_multi_fed_train_pmorl_dst.txt', 'a') as f:  # 设置文件对象
                f.write(
                    'fed_pmorl_dst_sample_ratio#######################################################################################')
                f.write('\n')
                f.write('client_num=')
                f.write(str(args.client_num))
                # f.write(',sample_ratio=')
                # f.write(str(args.sample_ratio))
                f.write(',eps_per_round=')
                f.write(str(args.eps_per_round))
                f.write(',batch_size_per_client=')
                f.write(str(args.batch_size))
                f.write(',dp=')
                f.write(str(args.dp))
                f.write(':\n')
                f.write(str(act1))
                f.write('\n')
                f.write(str(act2))
                f.write('\n')
                # f.write('return: \t')
                # f.write(str(return_list))
                # f.write('\n')
        print("end of round %d, the Q is %0.2f | %0.2f" % (
            round_num,
            act_1,
            act_2,
        ))

        if (round_num+1)%100==0:
            # global_client.agent.save(actor, args.save,
            #                          "m.{}_e.{}_n.{}_r.{}".format(args.model, args.env_name, args.actor_name,round_num))
            global_client.agent.save(global_client.agent.critic, args.save,
                                     "m.{}_e.{}_n.{}_r.{}".format(args.model, args.env_name, args.critic_name,round_num))
        # monitor.update(round_num,1,
        #                act_1,
        #                act_2)
        # # if num_eps+1 % 100 == 0:
        # 	agent.save(args.save, args.model+args.name+"_tmp_{}".format(number))

    # global_client.agent.save(actor, args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.actor_name))
    global_client.agent.save(global_client.agent.critic, args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.critic_name))

# def main():
if __name__ == '__main__':

    args = parser.parse_args()
    # setup the environment
    env = MultiObjectiveEnv(args.env_name)
    # env.state_spec
    # get state / action / reward sizes
    state_size = len(env.state_spec)
    action_size = env.action_spec[2][1] - env.action_spec[2][0]
    reward_size = len(env.reward_spec)

    # generate an agent for initial training
    agent = None
    if args.method == 'crl-naive':
        from crl.naive.meta import MetaAgent
        from crl.naive.models import get_new_model
    elif args.method == 'crl-envelope':
        from crl.envelope.fed_pmorl import MetaAgent
        from crl.envelope.models import get_new_model_1
    elif args.method == 'crl-energy':
        from crl.energy.meta import MetaAgent
        from crl.energy.models import get_new_model

    if args.serialize:
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    else:
        critic, actor = get_new_model_1(args.model, state_size, action_size, reward_size)

    # client_id_list = [
    #     i for i in range((args.rank - 1) * 10, (args.rank - 1) * 10 + 10)
    # ]
    preference_list = []
    for i in range(args.client_num):
        w_kept = torch.randn(reward_size)
        preference = (torch.abs(w_kept) / torch.norm(w_kept, p=1)).type(FloatTensor)
        preference_list.append(preference)
    # client_list = [
    #     client_local(id=i,preference=preference_list[i],env=env) for i in range((args.rank - 1) * 10, (args.rank - 1) * 10 + 10)
    # ]
    # client_list = [
    #     client_local(id=i, preference=preference_list[i], env=env) for i in
    #     range(args.client_num)
    # ]
    ##定义本地的client
    # client_local = namedtuple('client_local', ['id', 'preference', 'agent'])
    #初始化全局和本地模型
    global_client, client_list=Initialize(args,env)
    ##联邦训练
    fed_train(args, global_client, client_list)
    #多进程
    # size = int(args.client_num)
    # # size=args.client_num
    # for round_num in range(args.round):
    #     sampled_id_list = np.random.choice([i for i in range(args.client_num)],
    #                                        size=int(args.client_num * args.sample_ratio), replace=False)
    #     sampled_client_list = [client_list[i] for i in sampled_id_list]
    #     processes = []
    #     for rank in range(size):
    #         p = torch.multiprocessing.Process(target=init_processes, args=(rank, size, run))
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()
    #         p.close()



