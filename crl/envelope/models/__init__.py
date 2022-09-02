from __future__ import absolute_import, division, print_function
from .linear import EnvelopeLinearCQN,MO_ActorDiscrete


def get_new_model(name, state_size, action_size, reward_size):
    # if name == 'linear':
    #     return EnvelopeLinearCQN(state_size, action_size, reward_size), MO_ActorDiscrete(state_size, action_size, reward_size)
    if name == 'linear':
        return EnvelopeLinearCQN(state_size, action_size, reward_size)
    else:
        print("model %s doesn't exist." % (name))
        return None

def get_new_model_1(name, state_size, action_size, reward_size):
    if name == 'linear':
        return EnvelopeLinearCQN(state_size, action_size, reward_size), MO_ActorDiscrete(state_size, action_size, reward_size)
    #     return EnvelopeLinearCQN(state_size, action_size, reward_size)
    else:
        print("model %s doesn't exist." % (name))
        return None
