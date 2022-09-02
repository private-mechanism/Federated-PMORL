from __future__ import absolute_import, division, print_function
import argparse
import visdom
import torch
from torch.autograd import Variable
import time as Timer
import math
import numpy as np

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from envs.mo_env import MultiObjectiveEnv

parser = argparse.ArgumentParser(description='MORL-PLOT')
# CONFIG
parser.add_argument('--env-name', default='dst', metavar='ENVNAME',
                    help='environment to train on (default: dst)')
parser.add_argument('--method', default='crl-envelope', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# PLOT
parser.add_argument('--pltmap', default=False, action='store_true',
                    help='plot deep sea treasure map')
parser.add_argument('--pltpareto', default=True, action='store_true',
                    help='plot pareto frontier')
parser.add_argument('--pltcontrol', default=False, action='store_true',
                    help='plot control curve')
parser.add_argument('--pltdemo', default=False, action='store_true',
                    help='plot demo')
# LOG & SAVING
# parser.add_argument('--save', default='D://github/MORL-master/synthetic/crl/envelope/saved/', metavar='SAVE',
#                     help='address for saving trained models')
parser.add_argument('--save', default='/home/zfy/workspace/MORL-master/synthetic/fed_model/pmorl/saved/', metavar='SAVE',
                    help='address for saving trained models')
parser.add_argument('--name', default='0506_fed_pmorl_dst_dp_8_client_1000_V2', metavar='name',
                    help='specify a name for saving the model')
# parser.add_argument('--round', default='599', metavar='Round',
#                     help='the model after X round of fed training')
# Useless but I am too laze to delete them
parser.add_argument('--mem-size', type=int, default=10000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.3, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=True, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=100, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=32, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.98, metavar='BETA',
                    help='beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=True, action='store_true',
                    help='use homotopy optimization method')

args = parser.parse_args()

vis = visdom.Visdom()

# print(vis.check_connection())
# assert vis.check_connection()

# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
FloatTensor =  torch.FloatTensor
LongTensor =  torch.LongTensor
ByteTensor =  torch.ByteTensor
Tensor = FloatTensor

# Add data
gamma = args.gamma
args = parser.parse_args()
time = [-1, -3, -5, -7, -8, -9, -13, -14, -17, -19]
# treasure = [0.1, 2.8, 5.2, 7.3, 8.2, 9.0, 11.5, 12.1, 13.5, 14.2]
treasure = [0.7, 8.2, 11.5, 14., 15.1, 16.1, 19.6, 20.3, 22.4, 23.7]
# time	 = [ -1, -3, -5, -7, -8, -9]
# treasure = [0.1, 2.8, 5.2, 7.3, 8.2, 9.0]

# apply gamma
dis_time = (-(1 - np.power(gamma, -np.asarray(time))) / (1 - gamma)).tolist()
dis_treasure = (np.power(gamma, -np.asarray(time) - 1) * np.asarray(treasure)).tolist()

def find_in(A, B, base=0):
    # base = 0: tolerance w.r.t. A
    # base = 1: tolerance w.r.t. B
    # base = 2: no tolerance
    cnt = 0.0
    for a in A:
        for b in B:
            if base == 0:
              if np.linalg.norm(a - b, ord=1) < 0.20*np.linalg.norm(a):
                  cnt += 1.0
                  break
            elif base == 1:
              if np.linalg.norm(a - b, ord=1) < 0.20*np.linalg.norm(b):
                  cnt += 1.0
                  break
            elif base == 2:
              if np.linalg.norm(a - b, ord=1) < 0.3:
                  cnt += 1.0
                  break
    return cnt / len(A)

F1_list=[]
act_F1_list=[]
#
for round in [99,199,299,399,499,599,699,799,899,999]:
# for eps in [99,199,299,399,499,599,699,799,899,999,1099,1199,1299,1399,1499,1599,1699,1799,1899,1999]:
################# Control Frontier #################

    if args.pltcontrol:

        # setup the environment
        env = MultiObjectiveEnv(args.env_name)

        # generate an agent for plotting
        agent = None
        if args.method == 'crl-naive':
            from crl.naive.meta import MetaAgent
        elif args.method == 'crl-envelope':
            from crl.envelope.meta import MetaAgent
        elif args.method == 'crl-energy':
            from crl.energy.meta import MetaAgent
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}_r.{}".format(args.model, args.env_name, args.name,round)))
        agent = MetaAgent(model, args, is_train=False)

        # compute opt
        opt_x = []
        opt_y = []
        q_x = []
        q_y = []
        act_x = []
        act_y = []
        real_sol = np.stack((dis_treasure, dis_time))

        policy_loss = np.inf
        predict_loss = np.inf

        for i in range(2000):
            w = np.random.randn(2)
            w = np.abs(w) / np.linalg.norm(w, ord=1)
            # w = np.random.dirichlet(np.ones(2))
            w_e = w / np.linalg.norm(w, ord=2)
            if args.method == 'crl-naive' or args.method == 'crl-envelope':
                hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
            elif args.method == 'crl-energy':
                hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor), alpha=1e-5)
            realc = w.dot(real_sol).max() * w_e
            qc = w_e
            if args.method == 'crl-naive':
                qc = hq.data[0] * w_e
            elif args.method == 'crl-envelope':
                qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
            elif args.method == 'crl-energy':
                qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
            ttrw = np.array([0, 0])
            terminal = False
            env.reset()
            cnt = 0
            while not terminal:
                state = env.observe()
                action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
                next_state, reward, terminal = env.step(action)
                if cnt > 30:
                    terminal = True
                ttrw = ttrw + reward * np.power(args.gamma, cnt)
                cnt += 1
            ttrw_w = w.dot(ttrw) * w_e
            opt_x.append(realc[0])
            opt_y.append(realc[1])
            q_x.append(qc[0])
            q_y.append(qc[1])
            act_x.append(ttrw_w[0])
            act_y.append(ttrw_w[1])

        trace_opt = dict(x=opt_x,
                         y=opt_y,
                         mode="markers",
                         type='custom',
                         marker=dict(
                             symbol="circle",
                             size=1),
                         name='real')

        q_opt = dict(x=q_x,
                     y=q_y,
                     mode="markers",
                     type='custom',
                     marker=dict(
                         symbol="circle",
                         size=1),
                     name='predicted')

        act_opt = dict(x=act_x,
                       y=act_y,
                       mode="markers",
                       type='custom',
                       marker=dict(
                           symbol="circle",
                           size=1),
                       name='policy')
          ## quantitative evaluation
        policy_loss = 0.0
        predict_loss = 0.0
        TEST_N = 5000.0

        for i in range(int(TEST_N)):
            w = np.random.randn(2)
            w = np.abs(w) / np.linalg.norm(w, ord=1)
            # w = np.random.dirichlet(np.ones(2))
            w_e = w / np.linalg.norm(w, ord=2)
            if args.method == 'crl-naive' or args.method == 'crl-envelope':
                hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
            elif args.method == 'crl-energy':
                hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor), alpha=1e-5)
            realc = w.dot(real_sol).max() * w_e
            qc = w_e
            if args.method == 'crl-naive':
                qc = hq.data[0] * w_e
            elif args.method == 'crl-envelope':
                qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
            elif args.method == 'crl-energy':
                qc = w.dot(hq.data.cpu().numpy().squeeze()) * w_e
            ttrw = np.array([0, 0])
            terminal = False
            env.reset()
            cnt = 0
            while not terminal:
                state = env.observe()
                action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
                next_state, reward, terminal = env.step(action)
                if cnt > 30:
                    terminal = True
                ttrw = ttrw + reward * np.power(args.gamma, cnt)
                cnt += 1
            ttrw_w = w.dot(ttrw) * w_e

            base = np.linalg.norm(realc, ord=2)
            policy_loss += np.linalg.norm(realc - ttrw_w, ord=2)/base
            if args.method != 'crl-naive':
                predict_loss = np.linalg.norm(realc - qc, ord=1)
            else:
                predict_loss += np.linalg.norm(realc - qc, ord=2)/base

        policy_loss /= TEST_N / 100
        predict_loss /= TEST_N / 100

        # policy_loss = np.linalg.norm(realc - ttrw_w, ord=1)
        if args.method != 'crl-naive':
           predict_loss = np.linalg.norm(realc - qc, ord=1)

        print("discrepancies: policy-{}|predict-{}".format(policy_loss, predict_loss))

        layout_opt = dict(title="DST Control Frontier - {}_n:{}({:.3f}|{:.3f})".format(
            args.method, args.name, policy_loss, predict_loss),
            xaxis=dict(title='teasure value'),
            yaxis=dict(title='time penalty'))
        # print(trace_opt,q_opt,act_opt)

        vis._send({'data': [trace_opt, q_opt, act_opt], 'layout': layout_opt})

    ################# Pareto Frontier #################

    if args.pltpareto:

        # setup the environment
        env = MultiObjectiveEnv(args.env_name)

        # generate an agent for plotting
        agent = None
        if args.method == 'crl-naive':
            from crl.naive.meta import MetaAgent
        elif args.method == 'crl-envelope':
            from crl.envelope.meta import MetaAgent
        elif args.method == 'crl-energy':
            from crl.energy.meta import MetaAgent
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}_r.{}".format(args.model, args.env_name, args.name, round)))
        agent = MetaAgent(model, args, is_train=False)

        # compute recovered Pareto
        act_x = []
        act_y = []

        # predicted solution
        pred_x = []
        pred_y = []
        pred = []
        for i in range(2000):
            w = np.random.randn(2)
            w = np.abs(w) / np.linalg.norm(w, ord=1)
            # w = np.random.dirichlet(np.ones(2))
            ttrw = np.array([0, 0])
            terminal = False
            env.reset()
            cnt = 0
            # if args.method == "crl-naive":
            #     # hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
            #     hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
            #     # print(hq.data.cpu().numpy())
            #     pred_x.append(hq.data.cpu().numpy().squeeze()[0] * 1.0)
            #     pred_y.append(hq.data.cpu().numpy().squeeze()[1] * 1.0)
            if args.method == "crl-envelope":
                hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))
                # print(hq.data.cpu().numpy(),hq.data.cpu().numpy().squeeze())
                pred_x.append(hq.data.cpu().numpy().squeeze()[0] * 1.0)
                pred_y.append(hq.data.cpu().numpy().squeeze()[1] * 1.0)
            elif args.method == "crl-energy":
                hq, _ = agent.predict(torch.from_numpy(w).type(FloatTensor), alpha=1e-5)
                pred_x.append(hq.data.cpu().numpy().squeeze()[0] * 1.0)
                pred_y.append(hq.data.cpu().numpy().squeeze()[1] * 1.0)
            while not terminal:
                state = env.observe()
                action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
                next_state, reward, terminal = env.step(action)
                if cnt > 50:
                    terminal = True
                ttrw = ttrw + reward * np.power(args.gamma, cnt)
                cnt += 1

            act_x.append(ttrw[0])
            act_y.append(ttrw[1])


        act = np.vstack((act_x,act_y))
        act = act.transpose()
        obj = np.vstack((dis_treasure,dis_time))
        obj = obj.transpose()
        act_precition = find_in(act, obj, 2)
        act_recall = find_in(obj, act, 2)
        if act_precition==0 and act_recall==0:
            act_f1=0
        else:
            act_f1 = 2 * act_precition * act_recall / (act_precition + act_recall)
        pred_f1 = 0.0
        if args.method != 'crl-naive':
            pred = np.vstack((pred_x,pred_y))
            pred = pred.transpose()
            pred_precition = find_in(pred, obj, 1)
            pred_recall = find_in(obj, pred, 0)
            if pred_precition > 1e-8 and pred_recall > 1e-8:
                pred_f1 = 2 * pred_precition * pred_recall / (pred_precition + pred_recall)


        # Create and style traces(())
        trace_pareto = dict(x=dis_treasure,
                            y=dis_time,
                            mode="markers+lines",
                            type='custom',
                            marker=dict(
                                symbol="circle",
                                size=10),
                            line=dict(
                                width=1,
                                dash='dash'),
                            name='Pareto')

        act_pareto = dict(x=act_x,
                          y=act_y,
                          mode="markers",
                          type='custom',
                          marker=dict(
                              symbol="circle",
                              size=10),
                          line=dict(
                              width=1,
                              dash='dash'),
                          name='Recovered')

        pred_pareto = dict(x=pred_x,
                           y=pred_y,
                           mode="markers",
                           type='custom',
                           marker=dict(
                               symbol="circle",
                               size=3),
                           line=dict(
                               width=1,
                               dash='dash'),
                           name='Predicted')

        layout = dict(title="DST Pareto Frontier - {}:{}({:.3f}|{:.3f})".format(args.method, args.name,act_f1,pred_f1),
                      xaxis=dict(title='teasure value',
                                 zeroline=False),
                      yaxis=dict(title='time penalty',
                                 zeroline=False))
        act_F1_list.append(act_f1)
        F1_list.append(pred_f1)
        print("F1: policy-{}|prediction-{}".format(act_f1, pred_f1))
        # send to visdom
        if args.method == "crl-naive":
            vis._send({'data': [trace_pareto, act_pareto], 'layout': layout})
        elif args.method == "crl-envelope":
            vis._send({'data': [trace_pareto, act_pareto, pred_pareto], 'layout': layout})
        elif args.method == "crl-energy":
            vis._send({'data': [trace_pareto, act_pareto, pred_pareto], 'layout': layout})

    ################# HEATMAP #################

    if args.pltmap:
        see_map = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]]
        )[::-1]

        vis.heatmap(X=see_map,
                    opts=dict(
                        title="DST Map",
                        xmin=-10,
                        xmax=16.6))

    if args.pltdemo:
        see_map = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]]
        )[::-1]

        # setup the environment
        env = MultiObjectiveEnv(args.env_name)

        # generate an agent for plotting
        agent = None
        if args.method == 'crl-naive':
            from crl.naive.meta import MetaAgent
        elif args.method == 'crl-envelope':
            from crl.envelope.meta import MetaAgent
        elif args.method == 'crl-energy':
            from crl.energy.meta import MetaAgent
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
        agent = MetaAgent(model, args, is_train=False)
        new_episode = True

        while new_episode:

            dy_map = np.copy(see_map)
            dy_map[10 - 0, 0] = -3

            win = vis.heatmap(X=dy_map,
                              opts=dict(
                                  title="DST Map",
                                  xmin=-10,
                                  xmax=16.6))

            w1 = float(input("treasure weight: "))
            w2 = float(input("time weight: "))
            w = np.array([w1, w2])
            w = np.abs(w) / np.linalg.norm(w, ord=1)
            # w = np.random.dirichlet(np.ones(2))
            ttrw = np.array([0, 0])
            terminal = False
            env.reset()
            cnt = 0
            while not terminal:
                state = env.observe()
                action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
                next_state, reward, terminal = env.step(action)
                dy_map[10 - next_state[0], next_state[1]] = -3
                vis.heatmap(X=dy_map,
                            win=win,
                            opts=dict(
                                title="DST Map",
                                xmin=-10,
                                xmax=14.5))
                Timer.sleep(.5)
                if cnt > 50:
                    terminal = True
                ttrw = ttrw + reward * np.power(args.gamma, cnt)
                cnt += 1
            print("final reward: treasure %0.2f, time %0.2f, tot %0.2f" % (ttrw[0], ttrw[1], w.dot(ttrw)))
            new_episode = int(input("try again? 1: Yes | 0: No "))

print('policy',act_F1_list)
print('predict',F1_list)
###################3
# morl：discrepancies: policy-13.0453658082694|predict-11.442096971082469;F1: policy-0.8497069167643613|prediction-0.8876529477196886
# morl：discrepancies: policy-13.504214736330733|predict-11.590733752712609;F1: policy-0.8448377581120945|prediction-0.8745075970737197
# pmorl:discrepancies: policy-12.374887912748811|predict-48.62562220372557;
# pmorl:discrepancies: policy-13.070232999291528|predict-40.3768836746978;
# pmorl:discrepancies: policy-13.09875975033651|predict-39.25116480255222; F1: policy-0.7865414710485134|prediction-0.898981557941095
# pmorl:discrepancies: policy-13.470989337346502|predict-30.91869611124023; F1: policy-0.7880736809241335|prediction-0.9032081162599396
# naive:discrepancies: policy-682.3081038769055|predict-41.92960220801671
# naive:discrepancies: policy-760.0874885081008|predict-45.751592027426106



####################dst环境F1-score
######fed——pmorl
# policy [0.33333333333333337, 0.852908506284712, 0.7413978494623655, 0.7238877481177276, 0.6444444444444445, 0.5237807509710832, 0.5124887690925427, 0.6770195499815566, 0.7031055900621118, 0.8310003003905078]
# predict [0.1457259158751696, 0.604117181314331, 0.7384615384615384, 0.9096455070074195, 0.967741935483871, 0.8916597395400387, 0.8767200224655995, 0.8252569750367107, 0.8031119090365051, 0.8252569750367107]
######fed_morl
# policy [0.166832504145937, 0.17513983840894964, 0.17868939797549283, 0.18181818181818182, 0.5904995904995904, 0.6076511723570547, 0.7120393120393119, 0.7400673400673401, 0.8711009174311926, 0.735593220338983]
# predict [0.0, 0.0, 0.0, 0.0, 0.5375973759737598, 0.7088444157520982, 0.7973541791942274, 0.8091693956534682, 0.811998811998812, 0.8462647822324777]
######fed_naive
# policy [0, 0.18066698888351862, 0.18181818181818182, 0.18181818181818182, 0.5833333333333334, 0.9154013015184382, 0.9629245527612134, 0.9526053940822204, 0.9666752777060191, 0.9795918367346939]
# predict [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#0428-pmorl
# policy [0.7499999999999999, 0.5271028037383177, 0.721564555209415, 0.7438127090301002, 0.7906417957635156, 0.8726960503720664, 0.8777113578138344, 0.8949720670391061, 0.959417273673257, 0.9009988901220867, 0.9247974068071313, 0.9664082687338501, 0.7834539163258886, 0.8834512022630835, 0.9209983722192079, 0.9847715736040609, 0.99849774661993, 0.9952273298166291, 0.9967394030599448, 1.0]
# predict [0.3034981905910736, 0.8536443148688048, 0.7480438184663537, 0.8038277511961722, 0.7722529158993248, 0.8368711834835708, 0.8314344142565001, 0.8639591025276909, 0.8633134413185564, 0.8415870257746887, 0.8814317673378076, 0.9495798319327732, 0.8792378817595965, 0.8854834215658958, 0.9212513484358144, 0.9238633306429916, 0.9232839838492598, 0.9748846745258841, 0.9738327347357619, 0.9656064132402379]
#0428-naive
# policy [0.178494623655914, 0.7200000000000001, 0.5528089887640449, 0.973305954825462, 0.817360285374554, 0.8390011890606421, 0.931367292225201, 0.6106322996375352, 1.0, 0.9473684210526316, 0.8784667418263811, 0.8235294117647058, 0.9473684210526316, 1.0, 0.9537012817159299, 0.888888888888889, 0.988621997471555, 1.0, 0.9896438494569336, 0.8686979016958897]
# predict [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#0428-morl
# policy [0.18181818181818182, 0.7730006397952655, 0.8235294117647058, 0.9108671789242592, 0.8792838874680308, 0.888888888888889, 0.8723138146674247, 0.8624603860558917, 0.9473684210526316, 0.9473684210526316, 1.0, 0.9672088820036148, 0.888888888888889, 1.0, 0.9473684210526316, 1.0, 0.8727272727272728, 1.0, 0.9473684210526316, 0.9257420399352402]
# # predict [0.0, 0.7058339052848319, 0.8542907180385289, 0.9558861915948839, 0.8462172505151604, 0.9197947610045908, 0.9919354838709677, 0.9924433249370278, 0.9904088844018173, 0.9929506545820745, 0.9919354838709677, 0.9909182643794148, 0.98989898989899, 0.9904088844018173, 0.993963782696177, 0.9942167462911742, 0.9949748743718593, 0.9967394030599448, 0.9962358845671266, 0.9934574735782586]

#505-morl-V1
# policy [0.33324989570296204, 0.6598639455782312, 0.6051905920519058, 0.6962298025134649, 0.8600522193211487, 0.822829386763813, 0.7895982559950171, 0.9658738366080661, 0.9243243243243242, 0.9209983722192079, 0.9269199676637024, 0.9106231128191052, 0.9435368754956384, 0.9364605543710022, 0.940127388535032, 0.9801121876593575, 0.991681371313335, 0.9473684210526316, 0.9962358845671266, 0.9473684210526316]
# predict [0.18085208233604597, 0.17907949790794983, 0.6138674884437597, 0.7154784842646115, 0.7745098039215685, 0.8023952095808384, 0.8310929281122151, 0.8492520138089759, 0.8716502115655853, 0.8962472406181016, 0.9389920424403183, 0.9588755856324831, 0.9743589743589743, 0.9860583016476552, 0.9924433249370278, 0.9821882951653944, 0.9891331817033107, 0.9926970536388819, 0.9451476793248946, 0.9703989703989705]
#0428-pmorl
# policy [0.7499999999999999, 0.5271028037383177, 0.721564555209415, 0.7438127090301002, 0.7906417957635156, 0.8726960503720664, 0.8777113578138344, 0.8949720670391061, 0.959417273673257, 0.9009988901220867, 0.9247974068071313, 0.9664082687338501, 0.7834539163258886, 0.8834512022630835, 0.9209983722192079, 0.9847715736040609, 0.99849774661993, 0.9952273298166291, 0.9967394030599448, 1.0]
# predict [0.3034981905910736, 0.8536443148688048, 0.7480438184663537, 0.8038277511961722, 0.7722529158993248, 0.8368711834835708, 0.8314344142565001, 0.8639591025276909, 0.8633134413185564, 0.8415870257746887, 0.8814317673378076, 0.9495798319327732, 0.8792378817595965, 0.8854834215658958, 0.9212513484358144, 0.9238633306429916, 0.9232839838492598, 0.9748846745258841, 0.9738327347357619, 0.9656064132402379]
#0428-naive
# policy [0.178494623655914, 0.7200000000000001, 0.5528089887640449, 0.973305954825462, 0.817360285374554, 0.8390011890606421, 0.931367292225201, 0.6106322996375352, 1.0, 0.9473684210526316, 0.8784667418263811, 0.8235294117647058, 0.9473684210526316, 1.0, 0.9537012817159299, 0.888888888888889, 0.988621997471555, 1.0, 0.9896438494569336, 0.8686979016958897]
# predict [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#0506-morl
# policy [0.1814643188137164, 0.18046875, 0.1775280898876405, 0.17859818084537188, 0.6666666666666666, 0.825946580569416, 0.8826815642458101, 0.9561586638830899, 0.9754098360655737, 0.9904088844018173]
# predict [0.0, 0.0, 0.0, 0.0, 0.17808219178082194, 0.6118811881188119, 0.7499999999999999, 0.6968335035750766, 0.8324576765907763, 0.9085948158253752]
#0506-pmorl
# policy [0.18181818181818182, 0.7408163265306122, 0.5448589626933577, 0.5689745836985101, 0.4788760662318114, 0.7588688946015424, 0.734257693608387, 0.8692190096377533, 0.9665514261019879, 0.95926493108729]
# predict [0.08505747126436783, 0.3060481503229595, 0.6986072423398328, 0.8914189568143579, 0.8710132655941292, 0.7393633785061456, 0.707175177763413, 0.7273305758829145, 0.7585350713842334, 0.8144635447540012]


#client=500
# policy [0.18181818181818182, 0.5714285714285715, 0.6666666666666666, 0.37026164645820037, 0.3604790419161676, 0.3806215722120658, 0.28652482269503543, 0.3009966777408638, 0.02905982905982906, 0.13788819875776398]
# predict [0.0, 0.0, 0.41318111053450957, 0.7327913279132793, 0.8287769784172662, 0.737175598247388, 0.8607234406152092, 0.424203821656051, 0.13344887348353554, 0.06182237600922723]
#client=1000
# policy [0.18181818181818182, 0.7499999999999999, 0.7499999999999999, 0.5655951346655083, 0.7934762348555452, 0.6292508917954817, 0.6628807822489657, 0.825946580569416, 0.8826815642458101, 0.9561586638830899]