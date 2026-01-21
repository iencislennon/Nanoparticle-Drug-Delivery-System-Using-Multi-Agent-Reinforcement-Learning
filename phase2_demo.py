from rl.environment import NanoparticleEnv
from rl.hierarchical_agent import Hierarchical_agent
from rl.communication import CommunicationProtocol
import numpy as np

def train(n_episodes = 500):
    env = NanoparticleEnv(n_agents=5) # create 5 agents from environment
    agents = [Hierarchical_agent() for _ in range(5)] # they start their ecxisting
    comm = CommunicationProtocol() # object for communication of agents
    
    episode_rewards = []
    for episode in range(n_episodes): #Each episode is a new attempt by the agents to live life within the confines of nanoparticles, to adapt, to find a good uplift.
        obs = env.reset() #before the every episode environment is reset
        episode_reward = 0 
        """The obs variable stores observations—what each agent "sees" around itself.
            And episode_reward will accumulate the total reward for the entire group for one episode."""
        for step in range(1000):
            actions = []
            for i,agent in enumerate(agents):
                action = agent.act(obs[i])
                actions.append(action)
            next_obs, rewards, done, info = env.step(actions)
            """next_obs — a new set of experiences;

            rewards — rewards for selected actions;

            done — a signal indicating whether the episode is complete;

            info — additional internal environment information."""
            for i in range(len(agents)):
                position = info['positions']
                position_i = position[i]
                zone = env.tumor.get_zone(position_i)
                if zone == 'hypoxic' and rewards[i] > 50:
                    comm.broadcast(sender_id=i,position=position_i,message_type="tumor_found",strength=1.0,time=step)
                    messages = comm.receive(position,step)
            
        
            # learning 
            for i,agent in enumerate(agents):
                agent.learn(obs[i], actions[i],next_obs[i], rewards[i], done) #This fragment is the heart of the training.
                #Each agent gains experience: what they saw, what they did, what they received, and what the results were.

            episode_reward += rewards.sum()
            obs = next_obs

            if done:
                break

        episode_rewards.append(episode_reward)

        if episode % 10 ==0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}: reward = {avg:.2f}")
    return episode_rewards

if __name__ == "__main__":
    rewards = train(n_episodes=500)
    
    # Plot learning curve
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.savefig('outputs/phase2_training_final.png')
    print("Saved training curve!")
    
        


