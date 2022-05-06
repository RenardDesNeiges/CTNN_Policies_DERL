import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def state_action_obs_plot(trajectory, title='Observation, actions and neuron activations for an LTC network solving InvertedPendulum-v2'):
    # rough RNN state display
    
    policy_states = trajectory['states'][:,0,0,:]
    fig, axs = plt.subplots(3,1)
    axs[0].plot(trajectory['observations'])
    axs[0].set_xlim([0, trajectory['observations'].shape[0]-1])
    axs[0].set_title("Observations")
    axs[1].plot(trajectory['actions'])
    axs[1].set_xlim([0, trajectory['observations'].shape[0]-1])
    axs[1].set_title("Actions")
    activations = axs[2].imshow(np.transpose(policy_states),aspect="auto", cmap='RdBu',interpolation = 'nearest')
    axs[2].set_title("Activations")

    fig.suptitle(title)
    _ = fig.colorbar(activations, ax=axs[2],orientation="horizontal", extend='both')

    fig.tight_layout()
    plt.show()

def plot_ltc_cell(obs,act,states, policy_object, node_size=400, _cmap = 'RdBu'):
    _ccode = plt.get_cmap(_cmap)

    def network_to_nx(inputs,hidden,outputs):
        G = nx.DiGraph(directed=True)
        time_constant = policy_object.model.policy.weights[6].numpy()[0,:]
        
        in_weights = policy_object.model.policy.weights[0].numpy()
        ltc_weights = policy_object.model.policy.weights[2].numpy()
        out_weights = policy_object.model.policy.weights[4].numpy()
        
        for i in range(hidden):
            n = 'h{}'.format(i)
            G.add_node(n, weight=time_constant[i])
        for i in range(hidden):
            for j in range(hidden):
                G.add_edge('h{}'.format(i),'h{}'.format(j), weight = ltc_weights[i,j])  
        
        pos = nx.shell_layout(G)
        
        for i in range(inputs):
            n = 'in{}'.format(i)
            G.add_node(n, weight=0)
            pos[n] = np.array([-2,  i/((inputs+2)/2) - 0.5 ])
        for i in range(outputs):
            n = 'out{}'.format(i)
            G.add_node(n, weight=0)
            pos[n] = np.array([2,  i/(outputs/2) ])
        
        for i in range(inputs):
            for j in range(hidden):
                G.add_edge('in{}'.format(i),'h{}'.format(j), weight = in_weights[i,j])  
        for i in range(hidden):
            for j in range(outputs):
                G.add_edge('h{}'.format(i),'out{}'.format(j), weight = out_weights[i,j])  
                
        return G, pos

    G, pos = network_to_nx(obs.shape[0], policy_object.model.policy.hidden_units,act.shape[0])
    weights = nx.get_edge_attributes(G,'weight').values()
    edge_colors = [_ccode(e) for e in nx.get_edge_attributes(G,'weight').values()]
    node_colors = [_ccode(e) for e in nx.get_node_attributes(G,'weight').values()]
        
    options = {
        'node_color': list(node_colors),
        'node_size': node_size,
        'width': np.array(list(weights))*2.5,
        'edge_color': edge_colors,
        'arrowstyle': '-|>',
        'arrowsize': 12,
    }

    nx.draw_networkx(G,pos, arrows=True, **options,connectionstyle='arc3, rad = 0.05')
    plt.show()