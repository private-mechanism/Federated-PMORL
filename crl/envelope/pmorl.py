from __future__ import absolute_import, division, print_function
import random
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import collections
import torch.nn.functional as F
from collections import namedtuple
from collections import deque
from torch.distributions import MultivariateNormal, Categorical
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import minimize
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
# Tensor = FloatTensor
FloatTensor =torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
BoolTensor = torch.BoolTensor
Tensor = FloatTensor

dual_constraint = 0.1
kl_mean_constraint = 0.01
kl_var_constraint = 0.0001
kl_constraint = 0.01
discount_factor = 0.99
alpha_mean_scale = 1.0
# alpha_var_scale = 100.0
alpha_scale = 0.1
alpha_var_scale = 50.0
alpha_mean_max = 0.1
alpha_var_max = 10.0
alpha_max = 1.0
sample_episode_num = 30
sample_episode_maxstep = 200
sample_action_num = 64
mstep_iteration_num = 5
# evaluate_period = 10
# evaluate_episode_num = 10
# evaluate_episode_maxstep = 100


def categorical_kl(p1, p2):
    """
    calculates KL between two Categorical distributions
    :param p1: (B, D)
    :param p2: (B, D)
    """
    p1 = torch.clamp_min(p1, 0.0001)  # actually no need to clamp
    p2 = torch.clamp_min(p2, 0.0001)  # avoid zero division
    return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))

class MetaAgent(object):
    '''
    (1) act: how to sample an action to examine the learning
        outcomes or explore the environment;
    (2) memorize: how to store observed observations in order to
        help learing or establishing the empirical model of the
        enviroment;
    (3) learn: how the agent learns from the observations via
        explicitor implicit inference, how to optimize the policy
        model.
    '''

    def __init__(self, critic,actor, args, is_train=False):
        self.actor_ = actor
        self.actor=copy.deepcopy(actor)
        self.critic_ = critic
        # self.critic_2 = critic
        self.critic = copy.deepcopy(critic)
        self.is_train = is_train
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_delta = (args.epsilon - 0.05) / args.episode_num

        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num
        # self.probe=probe
        self.beta            = args.beta
        self.beta_init       = args.beta
        self.homotopy        = True
        self.beta_uplim      = 1.00
        self.tau             = 1000.
        self.beta_expbase    = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./args.episode_num))
        self.beta_delta      = self.beta_expbase / self.tau

        self.??_dual = dual_constraint
        self.??_kl_?? = kl_mean_constraint
        self.??_kl_?? = kl_var_constraint
        self.??_kl = kl_constraint
        self.?? = discount_factor
        self.??_??_scale = alpha_mean_scale
        self.??_??_scale = alpha_var_scale
        self.??_scale = alpha_scale
        self.??_??_max = alpha_mean_max
        self.??_??_max = alpha_var_max
        self.??_max = alpha_max
        # self.norm_loss_q = torch.nn.SmoothL1Loss()
        self.?? = np.random.rand()
        self.??_?? = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.??_?? = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.?? = 0.0  # lagrangian multiplier for discrete action space in the M-step
        self.sample_episode_num = sample_episode_num
        self.sample_episode_maxstep = sample_episode_maxstep
        self.sample_action_num = sample_action_num
        # self.batch_size = batch_size
        self.mstep_iteration_num = mstep_iteration_num
        # self.evaluate_period = evaluate_period
        # self.evaluate_episode_num = evaluate_episode_num
        # self.evaluate_episode_maxstep = evaluate_episode_maxstep
        self.trans_mem = deque()
        self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        self.priority_mem = deque()

        self.A_eye = torch.eye(self.critic.action_size)

        if args.optimizer == 'Adam':
            self.critic_optimizer = optim.Adam(self.critic_.parameters(), lr=args.lr)
            self.actor_optimizer = optim.Adam(self.actor_.parameters(), lr=args.lr)
            self.critic_optimizer_1 = optim.Adam(self.critic.parameters(), lr=args.lr)
            self.actor_optimizer_1 = optim.Adam(self.actor.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.critic_optimizer= optim.RMSprop(self.critic_.parameters(), lr=args.lr)
            self.actor_optimizer = optim.RMSprop(self.actor_.parameters(), lr=args.lr)
        for target_param, param in zip(self.actor_.parameters(), self.actor.parameters()):
            target_param.requires_grad = True
            param.requires_grad = False
        for target_param,param in zip(self.critic_.parameters(),self.critic.parameters()):
            target_param.requires_grad = True
            param.requires_grad = False

        self.w_kept = None
        self.update_count = 0
        self.update_freq = args.update_freq

        if self.is_train:
            self.critic.train()
        # if use_cuda:
        #     self.critic.cuda()
        #     self.critic_.cuda()

    def act(self, state, preference=None):
        # random pick a preference if it is not specified
        if preference is None:
            if self.w_kept is None:
                self.w_kept = torch.randn(self.critic.reward_size)
                self.w_kept = (torch.abs(self.w_kept) / \
                               torch.norm(self.w_kept, p=1)).type(FloatTensor)
            preference = self.w_kept
        #
        state = torch.from_numpy(state).type(FloatTensor)
        action=self.actor.action(Variable(state.unsqueeze(0), requires_grad=False),
            Variable(preference.unsqueeze(0), requires_grad=False))
        if self.is_train and (len(self.trans_mem) < self.batch_size or \
                              torch.rand(1)[0] < self.epsilon):
            action = np.random.choice(self.critic.action_size, 1)[0]
        return int(action)

    # def memorize(self, state, action, next_state, reward, terminal):
    #     self.trans_mem.append(self.trans(
    #         torch.from_numpy(state).type(FloatTensor),  # state
    #         action,  # action
    #         torch.from_numpy(next_state).type(FloatTensor),  # next state
    #         torch.from_numpy(reward).type(FloatTensor),  # reward
    #         terminal))  # terminal
    #     #######?????????????????????????????????priority#####################################################
    #     # randomly produce a preference for calculating priority
    #     # preference = self.w_kept
    #     preference = torch.randn(self.critic.reward_size)
    #     preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor)
    #     state = torch.from_numpy(state).type(FloatTensor)
    #     _, q = self.critic_(Variable(state.unsqueeze(0), requires_grad=False),
    #                        Variable(preference.unsqueeze(0), requires_grad=False))
    #     q = q[0, action].data
    #     wq = preference.dot(q)
    #     wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
    #     if not terminal:
    #         next_state = torch.from_numpy(next_state).type(FloatTensor)
    #         hq, _ = self.critic_(Variable(next_state.unsqueeze(0), requires_grad=False),
    #                             Variable(preference.unsqueeze(0), requires_grad=False))
    #         hq = hq.data[0]
    #         whq = preference.dot(hq)
    #         p = abs(wr + self.gamma * whq - wq)
    #     else:
    #         self.w_kept = None
    #         if self.epsilon_decay:
    #             self.epsilon -= self.epsilon_delta
    #         if self.homotopy:
    #             self.beta += self.beta_delta
    #             self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta
    #         p = abs(wr - wq)
    #     p += 1e-5
    #     self.priority_mem.append(
    #         p
    #     )
    #     if len(self.trans_mem) > self.mem_size:
    #         self.trans_mem.popleft()
    #         self.priority_mem.popleft()
    def memorize(self, state, action, next_state, reward, terminal):
        self.trans_mem.append(self.trans(
            torch.from_numpy(state).type(FloatTensor),  # state
            action,  # action
            torch.from_numpy(next_state).type(FloatTensor),  # next state
            torch.from_numpy(reward).type(FloatTensor),  # reward
            terminal))  # terminal
        #######?????????????????????????????????priority#####################################################
        # randomly produce a preference for calculating priority
        # preference = self.w_kept
        preference = torch.randn(self.critic.reward_size)
        preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor)
        state = torch.from_numpy(state).type(FloatTensor)
        _, q,_,q1= self.critic_(Variable(state.unsqueeze(0), requires_grad=False),
                           Variable(preference.unsqueeze(0), requires_grad=False))
        q = q[0, action].data
        wq = preference.dot(q)
        wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
        if not terminal:
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            hq, _,hq1, _= self.critic_(Variable(next_state.unsqueeze(0), requires_grad=False),
                                Variable(preference.unsqueeze(0), requires_grad=False))
            hq = hq.data[0]
            whq = preference.dot(hq)
            p = abs(wr + self.gamma * whq - wq)
        else:
            self.w_kept = None
            if self.epsilon_decay:
                self.epsilon -= self.epsilon_delta
            if self.homotopy:
                self.beta += self.beta_delta
                self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta
            p = abs(wr - wq)
        p += 1e-5
        self.priority_mem.append(
            p
        )
        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()

    def sample(self, pop, pri, k):
        pri = np.array(pri).astype(np.float)
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum())
        return [pop[i] for i in inds]

    def actmsk(self, num_dim, index):
        mask = BoolTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

################non-terminal indexes
    def nontmlinds(self, terminal_batch):
        mask = BoolTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def learn(self):
        # print('#######################,really',len(trans_mem))
        if len(self.trans_mem) > self.batch_size:
            # print('#######################,Check')
            self.update_count += 1
            action_size = self.critic.action_size
            minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)
            batchify = lambda x: list(x) * self.weight_num
            state_batch = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
            action_batch = batchify(map(lambda x: LongTensor([x.a]), minibatch))
            reward_batch = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
            next_state_batch = batchify(map(lambda x: x.s_.unsqueeze(0), minibatch))
            terminal_batch = batchify(map(lambda x: x.d, minibatch))
            w_batch = np.random.randn(self.weight_num, self.critic.reward_size)
            w_batch = np.abs(w_batch) / \
                      np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
            w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)
            __, Q,_,Q1 = self.critic_(Variable(torch.cat(state_batch, dim=0)),
                                    Variable(w_batch), w_num=self.weight_num)
            _, DQ,_,DQ1 = self.critic(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                               Variable(w_batch, requires_grad=False))
            ###############??????DQN???action??????
            # w_ext = w_batch.unsqueeze(2).repeat(1, action_size, 1)
            # w_ext = w_ext.view(-1, self.critic.reward_size)
            # _, tmpQ = self.critic_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
            #                       Variable(w_batch, requires_grad=False))
            # tmpQ = tmpQ.view(-1, self.critic.reward_size)
            # act = torch.bmm(Variable(w_ext.unsqueeze(1), requires_grad=False),
            #                 tmpQ.unsqueeze(2)).view(-1, action_size).max(1)[1]
            ###############??????Double_DQN???action??????
            w_ext = w_batch.unsqueeze(2).repeat(1, action_size, 1)
            w_ext = w_ext.view(-1, self.critic.reward_size)
            _, tmpQ,_,tmpQ1 = self.critic_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                   Variable(w_batch, requires_grad=False))
            tmpQ = tmpQ.view(-1, self.critic.reward_size)
            tmpQ1 = tmpQ1.view(-1, self.critic.reward_size)
            act = (torch.bmm(Variable(w_ext.unsqueeze(1), requires_grad=False),
                            tmpQ.unsqueeze(2)).view(-1, action_size)+torch.bmm(Variable(w_ext.unsqueeze(1), requires_grad=False),
                            tmpQ1.unsqueeze(2)).view(-1, action_size)).max(1)[1]
            ###############??????????????????action??????????????????action??????
            # act = self.actor_.forward(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
            #                                Variable(w_batch, requires_grad=False)).max(1)[1]
            # act = self.actor_.action(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
            #                            Variable(w_batch, requires_grad=False))
            #############??????????????????action?????????action??????
            # act_prob = self.actor_.forward(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
            #                           Variable(w_batch, requires_grad=False))
            # prob = Categorical(probs=act_prob)
            # act = prob.sample()
            ##############??????
            HQ = DQ.gather(1, act.view(-1, 1, 1).expand(DQ.size(0), 1, DQ.size(2))).squeeze()
            nontmlmask = self.nontmlinds(terminal_batch)
            with torch.no_grad():
                Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num,
                                             self.critic.reward_size).type(FloatTensor))
                Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                Tau_Q += Variable(torch.cat(reward_batch, dim=0))
            actions = Variable(torch.cat(action_batch, dim=0))

            Q = Q.gather(1, actions.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))
                         ).view(-1, self.critic.reward_size)
            Tau_Q = Tau_Q.view(-1, self.critic.reward_size)
            wQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                           Q.unsqueeze(2)).squeeze()
            wTQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                            Tau_Q.unsqueeze(2)).squeeze()

            HQ1 = DQ1.gather(1, act.view(-1, 1, 1).expand(DQ1.size(0), 1, DQ1.size(2))).squeeze()
            nontmlmask = self.nontmlinds(terminal_batch)
            with torch.no_grad():
                Tau_Q1 = Variable(torch.zeros(self.batch_size * self.weight_num,
                                             self.critic.reward_size).type(FloatTensor))
                Tau_Q1[nontmlmask] = self.gamma * HQ1[nontmlmask]
                Tau_Q1 += Variable(torch.cat(reward_batch, dim=0))
            actions = Variable(torch.cat(action_batch, dim=0))

            Q1 = Q1.gather(1, actions.view(-1, 1, 1).expand(Q1.size(0), 1, Q1.size(2))
                         ).view(-1, self.critic.reward_size)
            Tau_Q1 = Tau_Q1.view(-1, self.critic.reward_size)
            # Tau_Q_ = (Tau_Q.view(-1, self.critic.reward_size)+Tau_Q1.view(-1, self.critic.reward_size))/2
            # Tau_Q_ =(Tau_Q+Tau_Q1)/2
            Tau_Q_ = 0.8 * torch.minimum(Tau_Q, Tau_Q1) + 0.2 * torch.maximum(Tau_Q, Tau_Q1)
            wQ1 = torch.bmm(Variable(w_batch.unsqueeze(1)),
                           Q1.unsqueeze(2)).squeeze()
            wTQ1 = torch.bmm(Variable(w_batch.unsqueeze(1)),
                            Tau_Q1.unsqueeze(2)).squeeze()
            # wTQ_ = torch.minimum(wTQ1,wTQ)
            # wTQ_ = (wTQ1+wTQ)/2
            wTQ_ = 0.8*torch.minimum(wTQ1,wTQ)+0.2*torch.maximum(wTQ1,wTQ)
            critic_loss = (self.beta * F.mse_loss(wQ.view(-1), wTQ_.view(-1))+self.beta * F.mse_loss(wQ1.view(-1), wTQ_.view(-1)))*0.5
            critic_loss += ((1-self.beta) * F.mse_loss(Q.view(-1), Tau_Q_.view(-1))+(1-self.beta) * F.mse_loss(Q1.view(-1), Tau_Q_.view(-1)))*0.5
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            for name,param in self.critic_.named_parameters():
                param.grad.data.clamp_(-1, 1)
                # print('#######################,updation')
            self.critic_optimizer.step()
        ########??????MPO??????actor_network
            # if self.update_count % 10 == 0:
                # for i in range(3):
                #     minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)
                #     batchify = lambda x: list(x) * self.weight_num
                #     state_batch = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
                #     w_batch = np.random.randn(self.weight_num, self.critic.reward_size)
                #     w_batch = np.abs(w_batch) / \
                #               np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
                #     w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)
            #     self.actor.load_state_dict(self.actor_.state_dict())
            with torch.no_grad():
                actions = torch.arange(action_size)[..., None].expand(action_size, self.batch_size*self.weight_num)  # (da, K)
                b_p = self.actor.forward(torch.cat(state_batch,0), w_batch)  # (K, da)
                b = Categorical(probs=b_p)  # (K,)
                b_prob = b.expand((self.critic.action_size, self.batch_size*self.weight_num)).log_prob(actions).exp()  # (da, K)
                _, target_q_ ,_,_= self.critic.forward(
                    Variable(torch.cat(state_batch, 0), requires_grad=False),  # (K , ds)
                    Variable(w_batch, requires_grad=False)  # (K, da)
                )
                # _, target_q_=self.critic.forward(
                #         Variable(torch.cat(state_batch,0), requires_grad=False),  # (K , ds)
                #         Variable(w_batch, requires_grad=False)  # (K, da)
                #     )  # (K, da,dr)
                w_ext_ = w_batch.unsqueeze(2).repeat(1, action_size, 1)
                w_ext_ = w_ext_.view(-1, self.critic.reward_size)
                target_q_ = target_q_.view(-1, self.critic.reward_size)
                target_q=(torch.bmm(Variable(w_ext_.unsqueeze(1), requires_grad=False),
                          target_q_.unsqueeze(2)).view(-1, action_size)).transpose(0, 1)
                b_prob_np = b_prob.transpose(0, 1).numpy()  # (K, da)
                target_q_np = target_q.transpose(0, 1).numpy()  # (K, da)

            def dual(??):
                """
                dual function of the non-parametric variational
                g(??) = ??*?? + ??*mean(log(sum(??(a|s)*exp(Q(s, a)/??))))
                We have to multiply ?? by exp because this is expectation.
                This equation is correspond to last equation of the [2] p.15
                For numerical stabilization, this can be modified to
                Qj = max(Q(s, a), along=a)
                g(??) = ??*?? + mean(Qj, along=j) + ??*mean(log(sum(??(a|s)*(exp(Q(s, a)-Qj)/??))))
                """
                # print(target_q_np[0])
                max_q = np.max(target_q_np, 1)
                # print('max_q',max_q)
                return ?? * self.??_dual + np.mean(max_q) \
                        + ?? * np.mean(np.log(np.sum(
                        b_prob_np * np.exp((target_q_np - max_q[:, None]) / ??), axis=1)))
            bounds = [(1e-6, None)]
            res = minimize(dual, np.array([self.??]), method='SLSQP', bounds=bounds)
            self.?? = res.x[0]
            qij = torch.softmax(target_q / self.??, dim=0) # (N, K) or (da, K)z
            for _ in range(self.mstep_iteration_num):
                ??_p = self.actor_.forward(Variable(torch.cat(state_batch,0), requires_grad=True),Variable(w_batch,requires_grad=True)) # (K, da)
                # First term of last eq of [2] p.5
                ?? = Categorical(probs=??_p) # (K,)
                loss_p = torch.mean(
                    qij * ??.expand((action_size, self.batch_size*self.weight_num)).log_prob(actions)
                )
                kl = categorical_kl(p1=??_p, p2=b_p)
                if np.isnan(kl.item()):  # This should not happen
                    raise RuntimeError('kl is nan')
                # Update lagrange multipliers by gradient descent
                # this equation is derived from last eq of [2] p.5,
                # just differentiate with respect to ??
                # and update ?? so that the equation is to be minimized.
                self.?? -= self.??_scale * (self.??_kl - kl).detach().item()
                self.?? = np.clip(self.??, 0.0, self.??_max)
                loss_l = -(loss_p + self.?? * (self.??_kl - kl))
                self.actor_optimizer.zero_grad()
                # last eq of [2] p.5
                loss_l.backward()
                # clip_grad_norm_(self.actor_.parameters(), 0.3)
                for name, param in self.actor_.named_parameters():
                    param.grad.data.clamp_(-1, 1)
                self.actor_optimizer.step()
            # if self.update_count % 50 == 0:
            #     tau = 0.7
            #     for param, target_param in zip(self.actor_.parameters(), self.actor.parameters()):
            #         target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            #         # self.critic.load_state_dict(0.3*self.critic.state_dict()+0.7*self.critic_.state_dict())
            if self.update_count % 10 == 0:
                self.actor.load_state_dict(self.actor_.state_dict())
                # self.critic.load_state_dict(self.critic_.state_dict())
            if self.update_count % self.update_freq == 0:
                # self.actor.load_state_dict(self.actor_.state_dict())
                self.critic.load_state_dict(self.critic_.state_dict())
            #     print('update')
            #     tau=0.8
            #     for param, target_param in zip(self.actor_.parameters(), self.actor.parameters()):
            #             target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            #     for param, target_param in zip(self.critic_.parameters(), self.critic.parameters()):
            #         target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            return self.actor, self.critic, critic_loss
        return self.actor, self.critic, 0.0

    def reset(self):
        self.w_kept = None
        if self.epsilon_decay:
            self.epsilon -= self.epsilon_delta
        if self.homotopy:
            self.beta += self.beta_delta
            self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta

    def predict(self, probe):
        return self.critic(Variable(FloatTensor([0, 0]).unsqueeze(0), requires_grad=False),
                          Variable(probe.unsqueeze(0), requires_grad=False))

    def save(self, model, save_path, model_name):
        torch.save(model, "{}{}.pkl".format(save_path, model_name))


    def find_preference(
            self,
            w_batch,
            target_batch,
            pref_param):
        with torch.no_grad():
            w_batch = FloatTensor(w_batch)
            target_batch = FloatTensor(target_batch)
        # print('eeeeeeeeeeeeeeeeeee')
        pref_param = FloatTensor(pref_param)
        pref_param.requires_grad = True
        sigmas = FloatTensor([0.001]*len(pref_param))
        dist = torch.distributions.normal.Normal(pref_param, sigmas)
        pref_loss = dist.log_prob(w_batch).sum(dim=1) * target_batch
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        # Total loss
        loss = pref_loss.mean()
        loss.backward()
        eta = 1e-3
        pref_param = pref_param + eta * pref_param.grad
        pref_param = simplex_proj(pref_param.detach().cpu().numpy())
        # print("update prefreence parameters to", pref_param)
        return pref_param


# projection to simplex
def simplex_proj(x):
    y = -np.sort(-x)
    sum = 0
    ind = []
    for j in range(len(x)):
        sum = sum + y[j]
        if y[j] + (1 - sum) / (j + 1) > 0:
            ind.append(j)
        else:
            ind.append(0)
    rho = np.argmax(ind)
    delta = (1 - (y[:rho+1]).sum())/(rho+1)
    return np.clip(x + delta, 0, 1)