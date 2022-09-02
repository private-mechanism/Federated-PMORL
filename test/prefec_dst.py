from __future__ import absolute_import, division, print_function
import argparse
import visdom
import torch
import numpy as np
# from sklearn.manifold import TSNE

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from envs.mo_env import MultiObjectiveEnv

parser = argparse.ArgumentParser(description='MORL-PLOT')
# CONFIG
parser.add_argument('--env-name', default='dst', metavar='ENVNAME',
                    help='environment to train on (default: tf): ft | ft5 | ft7')
parser.add_argument('--method', default='crl-envelope', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy|crl-pmorl')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')

# LOG & SAVING
# parser.add_argument('--save', default='D://github/MORL-master/synthetic/fed_model/naive/saved/', metavar='SAVE',
#                     help='address for saving trained models')
# parser.add_argument('--name', default='fed_naive_dst_sample_01_V2', metavar='name',
#                     help='specify a name for saving the model')
parser.add_argument('--save', default='/home/zfy/workspace/MORL-master/synthetic/fed_model/pmorl/saved/', metavar='SAVE',
                    help='address for saving trained models')
parser.add_argument('--name', default='428_naive_dst', metavar='name',
                    help='specify a name for saving the model')
# parser.add_argument('--actor_name', default='actor_pmorl_ft_001_v3', metavar='actor_name',
#                     help='specify a name for saving the actor_model')
parser.add_argument('--critic_name', default='0506_fed_pmorl_dst_dp_8_client_1000_V2', metavar='critic_name',
                    help='specify a name for saving the critic_model')
# Useless but I am too laze to delete them
parser.add_argument('--mem-size', type=int, default=10000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.3, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=30, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=32, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=True, action='store_true',
                    help='use homotopy optimization method')

args = parser.parse_args()

vis = visdom.Visdom()

# assert vis.check_connection()
use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
FloatTensor =  torch.FloatTensor
LongTensor =  torch.LongTensor
ByteTensor =  torch.ByteTensor
Tensor = FloatTensor


def matrix2lists(MATRIX):
    X, Y = [], []
    for x in list(MATRIX[:, 0]):
        X.append(float(x))
    for y in list(MATRIX[:, 1]):
        Y.append(float(y))
    return X, Y


def make_train_data(num_step, reward):
    discounted_return = np.empty([num_step])
    # Discounted Return
    running_add = 0
    for t in range(num_step - 1, -1, -1):
        running_add += reward[t]
        discounted_return[t] = running_add
    return discounted_return[0]


def generate_w(num_prefence, pref_param, fixed_w=None):
    if fixed_w is not None and num_prefence>1:
        sigmas = torch.Tensor([0.01]*len(pref_param))
        w = torch.distributions.normal.Normal(torch.Tensor(pref_param), sigmas)
        w = w.sample(torch.Size((num_prefence-1,))).numpy()
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence-1, 1)
        return np.concatenate(([fixed_w], w))
    elif fixed_w is not None and num_prefence==1:
        return np.array([fixed_w])
    else:
        sigmas = torch.Tensor([0.001]*len(pref_param))
        w = torch.distributions.normal.Normal(torch.Tensor(pref_param), sigmas)
        w = w.sample(torch.Size((num_prefence,))).numpy()
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
        return w
    return w

def renew_w(preferences, dim, pref_param):
    sigmas = torch.Tensor([0.01]*len(pref_param))
    w = torch.distributions.normal.Normal(torch.Tensor(pref_param), sigmas)
    w = w.sample(torch.Size((1,))).numpy()
    w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1)
    preferences[dim] = w
    return preferences

distance=[]
mean_list=[]
for round in [99,199,299,399,499,599,699,799,899,999]:
# for round in [1099,1199,1299,1399,1499,1599,1699,1799,1899,1999]:
    # setup the environment
    env = MultiObjectiveEnv(args.env_name)
    # generate an agent for plotting
    agent = None
    if args.method == 'crl-naive':
        from crl.naive.meta import MetaAgent
        model = torch.load("{}{}.pkl".format(args.save,
                                              "m.{}_e.{}_n.{}_r.{}".format(args.model, args.env_name, args.critic_name,round)))
        agent = MetaAgent(model, args, is_train=False)
    elif args.method == 'crl-envelope':
        from crl.envelope.meta import MetaAgent
        model = torch.load("{}{}.pkl".format(args.save,
                                              "m.{}_e.{}_n.{}_r.{}".format(args.model, args.env_name, args.critic_name,round)))
        agent = MetaAgent(model, args, is_train=False)

    # elif args.method == 'crl-pmorl':
    #     from crl.envelope.pmorl import MetaAgent
    #     actor=torch.load("{}{}.pkl".format(args.save,
    #                                          "m.{}_e.{}_n.{}_r.{}".format(args.model, args.env_name, args.actor_name,round)))
    #     critic=torch.load("{}{}.pkl".format(args.save,
    #                                          "m.{}_e.{}_n.{}_r.{}".format(args.model, args.env_name, args.critic_name,round)))
    #     agent = MetaAgent(critic, actor,args, is_train=False)
        # agent = MetaAgent(critic, args, is_train=False)

    elif args.method == 'crl-energy':
        from crl.energy.meta import MetaAgent
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
        agent = MetaAgent(model, args, is_train=False)


    REPEAT = 1
    # unknown pref
    # preferences=[[1.0,0.0,0.0,0.0,0.0,0.0]
    #              ,[0.0,1.0,0.0,0.0,0.0,0.0]
    #              ,[0.0,0.0,1.0,0.0,0.0,0.0]
    #              ,[0.0,0.0,0.0,1.0,0.0,0.0]
    #              ,[0.0,0.0,0.0,0.0,1.0,0.0]
    #              ,[0.0,0.0,0.0,0.0,0.0,1.0]]
    preferences = [[1.0,0.0],[0.0,1.0]]
    # with open('findpreference_ft_.txt', 'a') as f:  # 设置文件对象
    #     f.write('pmorl#######################################################################################')
    # dis_list=[]
    # dis=0
    dis_temp=0
    for i,w in enumerate(preferences):
        # print('w:', w)
        unknown_w = w
        # List=[]
        # for j in range(5):
        bbbest_param = []
        for iii in range(10):
            pref_param = np.array([1.0 / 2.0, 1.0 / 2.0])
            # pref_param = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0,
            #                        1.0/6.0, 1.0/6.0, 1.0/6.0])
            explore_w = generate_w(1, pref_param)
            # print('explore_w:',explore_w)
            max_target = -5
            sample_episode = 0
            best_param = 0
            while True:
                # w = np.random.randn(6)
                # w[2], w[3], w[4], w[5] = 0, 0, 0, 0
                # w = np.abs(w) / np.linalg.norm(w, ord=1)
                # w = np.random.dirichlet(np.ones(2))
                # w_e = w / np.linalg.norm(w, ord=2)
                acc_target = 0
                rep_target = []
                rep_explore_w = []
                for _ in range(REPEAT):
                    sample_episode += 1
                    # ttrw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    ttrw = np.array([0.0, 0.0])#ttrw：total reward
                    terminal = False
                    env.reset()
                    cnt = 0
                    while not terminal:
                        state = env.observe()
                        # next_state, reward, terminal = env.step(action)
                        # action = agent.act(state)
                        # print(state,np.sum(explore_w.squeeze()),explore_w.squeeze())
                        preference=torch.from_numpy(explore_w.squeeze()).type(FloatTensor)
                        # print(preference)
                        action = agent.act(state, preference)
                        next_state, reward, terminal = env.step(action)
                        if cnt > 100:
                            terminal = True
                        ttrw = ttrw + reward * np.power(args.gamma, cnt)
                        cnt += 1
                    target = np.sum(ttrw * unknown_w)
                    acc_target += target
                    rep_target.append(target)
                    rep_explore_w.append(explore_w)
                acc_target = acc_target/REPEAT
                # print(acc_target)
                # print("avg performance", acc_target)
                if acc_target >= max_target:
                    max_target = acc_target
                    best_param = pref_param
                # print('best_param:', best_param)
                pref_param = agent.find_preference(
                        np.stack(rep_explore_w),
                        np.hstack(rep_target),
                        pref_param)
                # print('pref_param:',pref_param)
                explore_w = renew_w(explore_w, 0, pref_param)
                if sample_episode >= args.episode_num:
                    # print("the best param:", best_param)
                    break
            bbbest_param.append(best_param)
        bbbest_param = np.array(bbbest_param)
        # print('bbbest_param',bbbest_param)
        mean=bbbest_param.mean(axis=0)
        std=bbbest_param.std(axis=0)
        List=bbbest_param.tolist()
        print("mean", mean)
        print("std", std)
        dis_temp+=np.square(np.linalg.norm(mean-np.array(w)))
        # mean_list.append(mean)
        print('dis of round:',round,dis_temp)
    distance.append(np.sqrt(dis_temp))
print(distance)
    # List.append(mean.tolist())
    # average=np.sum(List, axis = 0)/len(List)
    # average=average.tolist()
    # with open('findpreference_ft_.txt', 'a') as f:  # 设置文件对象
    #     #     f.write('\n')
    #     #     f.write(str(i))
    #     #     f.write('\t:')
    #     #     f.write(str(List))
    #     #     f.write('\n')
    #     #     f.write('average:')
    #     #     f.write(str(mean))
    #     #     f.write('\n')
############ft环境
#naive:
# [2.2345227557168332, 2.079633770025548, 2.4874565583941153, 2.3141576480949397, 1.811867521782746, 2.1699559975233607, 1.9422345121649391, 2.541170701178487, 1.6866485954129362, 1.230411480546512]
#pmorl
#[2.458632234428387, 2.3762709101614914, 1.501691072802744, 1.9399914381837733, 1.6842247307235885, 1.293309312522601, 0.5103910558903536, 0.7846351532197703, 0.7353664933912921, 0.6602796024707367]
#morl
#[2.4693946166676564, 2.7220649394104166, 1.4707334477692755, 1.7417500202908118, 1.9307118428546857, 2.032292413612096, 1.7718020348484926, 1.9771130807581152, 1.8016173143291427, 1.0207798941824513]

############dst环境
#naive
#[1.224744871391589, 1.0704715494886459, 1.4210946775171038, 1.1246338581589685, 1.224744871391589, 1.224744871391589, 1.2097566419058046, 1.1785928845991984, 1.224744871391589, 1.20034602555386]
#morl
#[1.261658825975257, 1.5213650526322833, 1.2187515264942377, 1.46471583801223, 1.224744871391589, 1.224744871391589, 1.224744871391589, 1.224744871391589, 1.229082213815288, 1.222838560140405]
#pmorl
#[1.0, 1.224744871391589, 1.224744871391589, 1.224744871391589, 1.2153172316581324, 1.2137332352614674, 1.2003807592741462, 1.224744871391589, 1.1999633154082125, 1.224744871391589]

#morl
#[0.9528666110885228, 1.093654273147449, 0.9338577829021885, 0.8463147043145143, 0.9273618495495706, 0.9289251543360106, 0.9273618495495706, 0.9273618495495706, 0.8879745119883357, 0.906526583116254]
#pmorl
#[0.8288640256865886, 0.9273618495495706, 0.9273618495495706, 0.9282493098384581, 0.9099797426632289, 0.9273618495495706, 0.9257969951171345, 0.9196038353214457, 0.9018459627713985, 0.9223651770918391]
#naive
#[1.1661903789690604, 0.8353679467756165, 0.8705701764831584, 0.9337051908804019, 0.9273618495495706, 0.9020159799958629, 0.9273618495495706, 0.9182671053703957, 0.9209305720163713, 0.9273618495495706]
#[1.1661903789690604, 0.8351952545643275, 0.8293888242419429, 1.1404082454248095, 0.9258679499811315, 0.9273618495495706, 0.9000180401453893, 0.9047258417492546, 0.8874121445181467, 0.9012888837765959]


#pmorl：[0.8535356598558126, 0.9273618456931699, 0.7052067150960029, 0.9359428487629982, 0.812403828724677, 0.7645400137253691, 0.8653716814527571, 0.8767765712847372, 0.812403828724677, 0.7615773105863909]
#morl:[1.0842027438547965, 1.0920804054557052, 1.0624766297327761, 1.119689295783376, 0.8197532487664919, 0.7469844249854395, 0.8660254037844387, 0.7179104618176613, 0.9763769803219797, 0.9306451652386728]
#naive:[1.224744871391589, 1.1627523794556607, 0.9752597576013893, 0.8610322622095864, 0.771412791351172, 0.8308717628168475, 0.8162103690945662, 0.9074405528968477, 0.8213765915497488, 0.7681145794427793]


#0505
#pmorl:[0.812403828724677, 0.9273618456931699, 1.0592970537048112, 0.7970978101320669, 0.8660254037844387, 0.8660254037844387, 0.812403828724677, 0.8621968526625048, 0.7673152786270304, 0.8156606274589753,0.7942547954220786, 0.9316497803329199, 0.7157089420514076, 0.6092877303592653, 0.3999999761581421, 0.7303800106106976, 0.30000001192092896, 0.5108660336985618, 0.5288625914316373, 0.441221397583214]
#morl:[1.7203799348095123, 0.7457302883727946, 0.7681145794427793, 0.9273618456931699, 0.7681145794427793, 0.994987428719931, 0.8660254037844387, 0.8316606091730127, 0.8246211251235323, 0.7593567338660656,0.9552005278543702, 0.8496273025724332, 0.30000001192092896, 0.501423167342651, 0.8002353007290007, 0.7617300657607027, 0.643652204859181, 0.699999988079071, 0.4134248267515403, 0.5999999940395355]
#naive:[1.224744871391589, 1.1627523794556607, 0.9752597576013893, 0.8610322622095864, 0.771412791351172, 0.8308717628168475, 0.8162103690945662, 0.9074405528968477, 0.8213765915497488, 0.7681145794427793,0.994987428719931, 0.8660254037844387, 0.8316606091730127, 0.8246211251235323, 0.7593567338660656,0.9552005278543702, 0.8496273025724332, 0.30000001192092896, 0.501423167342651, 0.8002353007290007]

#dp_fed
#1000
# [1.1798408097529836, 0.39977104515115314, 0.28415645447311216, 0.7583794462962419, 0.7450261406902308, 0.8717573225355252, 0.7486356950262675, 0.7796446847872065, 0.7642254751420706, 0.8142081925545851]
#500
#[1.04649887149581, 0.48367465412389116, 0.3004420399280853, 0.8660254037844387, 0.812403828724677, 0.8803785095377701, 0.8545339644570427, 0.812403828724677, 1.224744871391589, 1.0053954854201361]


