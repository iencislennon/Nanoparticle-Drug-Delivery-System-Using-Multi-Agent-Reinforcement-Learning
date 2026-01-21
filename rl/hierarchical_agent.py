import numpy as np
class High_level_policy:
    """
    High-level strategic policy
    """
    def __init__(self):
        self.strategies = [
            'explore_randomly', 
            'target_necrotic',
            'target_hypoxic', 
            'follow_gradient'
        ]
        
        # q-table
        self.q_table = {}
        
        self.epsilon = 0.05  # exploration rate
        self.lr = 0.01      # learning rate
        self.gamma = 0.9    # discount factor
    
    def _get_state_key(self, state):
        """Convert observation to discrete state"""
        # discretize position to 1mm bins
        pos_discrete = tuple(np.round(state[:3] * 1000).astype(int))
        # discretize oxygen to 10% bins
        oxygen_discrete = int(state[6] * 10)
        return pos_discrete + (oxygen_discrete,)
    
    def select_strategy(self, state):
        """
        Arguments:
            state: observation array (7D)
        
        Returns:
            strategy name (string)
        """
        state_key = self._get_state_key(state)
        
        # epsilon-greedy
        if np.random.random() < self.epsilon:
            # random strategy
            return np.random.choice(self.strategies)
        else:
            # best strategy
            if state_key not in self.q_table:
                # initialize q-values as dictionary
                self.q_table[state_key] = {s: 0.0 for s in self.strategies}
            
            # choose the strategy with the biggest q value
            q_values = self.q_table[state_key]
            best_strategy = max(q_values, key=q_values.get)
            return best_strategy
    
    def update(self, state, strategy, reward, next_state, done):
        """
        Q(s,a) = Q(s,a) + aplha * r + gamma * max Q(s',a') - Q(s,a)]
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # initialize if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {s: 0.0 for s in self.strategies}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {s: 0.0 for s in self.strategies}
        
        # current q-value
        q_curr = self.q_table[state_key][strategy]
        
        # max next q-value
        if done:
            next_max_q = 0
        else:
            next_max_q = max(self.q_table[next_state_key].values())
        
        # q-learning update
        q_new = q_curr + self.lr * (reward + self.gamma * next_max_q - q_curr)
        
        # update q-table
        self.q_table[state_key][strategy] = q_new
class Low_level_policy():
    """
    Low-level tactical decision making
    Decides HOW to do it (actions)
    """
    
    def __init__(self, n_actions=8):
        self.n_actions = n_actions
        
        # Q-table: (state, strategy) -> Q-values for each action
        self.q_table = {}
        
        self.epsilon = 0.05
        self.lr = 0.01
        self.gamma = 0.9
    
    def _get_state_key(self, state, strategy):
        """
        Convert state and strategy to hashable key
        """
        # Discretize position
        pos_discrete = tuple(np.round(state[:3] * 10).astype(int))
        # Add strategy
        return pos_discrete + (strategy,)
    def select_action(self, state, strategy):
        """
        arguments:
            state: observation array
            strategy: current high-level strategy
        
        Returns:
            action (int 0-7)
        """
        state_key = self._get_state_key(state, strategy)
        
        # epsilon-greedy
        if np.random.random() < self.epsilon:
            # random action
            return np.random.randint(0, self.n_actions)
        else:
            # best action
            if state_key not in self.q_table:
                # initialize q-values
                self.q_table[state_key] = np.zeros(self.n_actions)
            
            # choose best action
            return np.argmax(self.q_table[state_key])
    def update(self, state, strategy, action, reward, next_state, next_strategy, done):
        """
        Update Q-values using Q-learning
        """
        state_key = self._get_state_key(state, strategy)
        next_state_key = self._get_state_key(next_state, next_strategy)
        
        # initialize if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_actions)
        
        # q-learning update
        current_q = self.q_table[state_key][action]
        
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q 

class Hierarchical_agent:
    """
        Agents with doubled hierarchy
    """
    def __init__(self, strategy_duration=10):
        self.high_level = High_level_policy()
        self.low_level = Low_level_policy()

        self.current_strategy = None
        self.strategy_timer = 0

    def act(self, observation):
        # update strategy after every 60 steps
        if self.strategy_timer == 0:
            self.current_strategy = self.high_level.select_strategy(observation)

        action = self.low_level.select_action(observation, self.current_strategy)

        self.strategy_timer = (self.strategy_timer + 1)% 60
        return action
    
    def learn(self, obs, action, next_obs, reward, done):
        """
            arguments:
                obs: current obsevation
                action: action taken
                reward: reward receive
                done: is episod finished
        """
        next_strategy = self.high_level.select_strategy(next_obs)
        # update for high level
        self.high_level.update(
            obs,
            self.current_strategy,
            reward,
            next_obs,
            done
        )

        self.low_level.update(
            obs,
            self.current_strategy,
            action,
            reward,
            next_obs,
            next_strategy,
            done
        )
            
