import matplotlib.pyplot as plt
import numpy as np

def state_action_obs_plot(trajectory):
    # rough RNN state display
    policy_states = trajectory['states'][:,0,0,:]
    fig, axs = plt.subplots(2)
    axs[0].plot(trajectory['observations'])
    axs[1].imshow(np.transpose(policy_states))
    plt.show()