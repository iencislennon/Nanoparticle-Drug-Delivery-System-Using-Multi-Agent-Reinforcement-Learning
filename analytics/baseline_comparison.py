"""
Baseline Comparison

Compare personalized RL system against multiple baselines:
1. Random Agent
2. Greedy Agent
3. Non-Personalized Agent
4. Flat RL (no hierarchy)
5. No Communication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats

import sys
sys.path.append('.')

from rl.environment import NanoparticleEnv
from rl.hierarchical_agent import Hierarchical_agent
from models.genomic_vit import GenomicViT
from rl.personalized_env import PersonalizedNanoparticleEnv



class RandomAgent:
    """Baseline 1: Random action selection"""
    
    def __init__(self):
        self.action_space = [0, 1, 2, 3, 4]  # 5 actions
    
    def act(self, observation):
        """Random action"""
        return np.random.choice(self.action_space)
    
    def learn(self, *args):
        """No learning"""
        pass


class GreedyAgent:
    """Baseline 2: Always move toward nearest tumor cell"""
    
    def __init__(self, tumor_center=None):
        self.tumor_center = tumor_center or np.array([500, 500])
    
    def act(self, observation):
        """Move toward tumor center"""
        # observation contains position
        if len(observation) >= 2:
            agent_pos = observation[:2]
            
            # Direction to tumor
            direction = self.tumor_center - agent_pos
            
            # Choose action based on direction
            angle = np.arctan2(direction[1], direction[0])
            
            # Map angle to action (0-4)
            # 0: right, 1: up, 2: left, 3: down, 4: stay
            if abs(angle) < np.pi/4:
                return 0  # right
            elif angle > 3*np.pi/4 or angle < -3*np.pi/4:
                return 2  # left
            elif angle > 0:
                return 1  # up
            else:
                return 3  # down
        
        return 4  # stay if can't determine
    
    def learn(self, *args):
        """No learning"""
        pass


class FlatAgent:
    """Baseline 4: Flat Q-learning (no hierarchy)"""
    
    def __init__(self, learning_rate=0.01, epsilon=0.1):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.q_values = {}  # State-action values
        self.action_space = [0, 1, 2, 3, 4]
    
    def _state_to_key(self, state):
        """Convert state to hashable key"""
        return tuple(np.round(state, 1))
    
    def act(self, observation):
        """Epsilon-greedy action"""
        state_key = self._state_to_key(observation)
        
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        # Get Q-values for this state
        q_vals = [self.q_values.get((state_key, a), 0) 
                  for a in self.action_space]
        
        return self.action_space[np.argmax(q_vals)]
    
    def learn(self, state, action, next_state, reward, done):
        """Q-learning update"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_values.get((state_key, action), 0)
        
        # Max Q-value for next state
        next_q_vals = [self.q_values.get((next_state_key, a), 0) 
                       for a in self.action_space]
        max_next_q = max(next_q_vals)
        
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + 0.95 * max_next_q
        
        new_q = current_q + self.lr * (target - current_q)
        self.q_values[(state_key, action)] = new_q

def train_baseline(agent, env, n_episodes=100, verbose=False):
    """
    Train baseline agent on environment
    
    arguments:
        agent: Agent to train
        env: Environment
        n_episodes: Number of episodes
        verbose: Print progress
    
    returns:
        episode_rewards: List of rewards per episode
    """
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            # Get actions for all agents
            if hasattr(env, 'n_agents'):
                actions = [agent.act(obs[i]) for i in range(env.n_agents)]
            else:
                actions = [agent.act(obs)]
            
            # Step environment
            next_obs, rewards, done, info = env.step(actions)
            
            # Learning
            if hasattr(env, 'n_agents'):
                for i in range(env.n_agents):
                    agent.learn(obs[i], actions[i], next_obs[i], rewards[i], done)
            else:
                agent.learn(obs, actions[0], next_obs, rewards, done)
            
            episode_reward += rewards.sum() if hasattr(rewards, 'sum') else rewards
            obs = next_obs
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if verbose and (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.2f}")
    
    return episode_rewards


def train_personalized(genomic_model, patient_sequence, n_agents=3, n_episodes=100):
    """
    Train personalized agent
    
    arguments:
        genomic_model: Trained genomic model
        patient_sequence: Patient DNA sequence
        n_agents: Number of agents
        n_episodes: Training episodes
    
    returns:
        episode_rewards: List of rewards
    """
    # Create personalized environment
    env = PersonalizedNanoparticleEnv(
        genomic_model=genomic_model,
        patient_sequence=patient_sequence,
        n_agents=n_agents
    )
    
    # Create hierarchical agent
    agent = Hierarchical_agent()
    
    # Train
    episode_rewards = train_baseline(agent, env, n_episodes=n_episodes, verbose=False)
    
    return episode_rewards



def compare_all_baselines(
    genomic_model,
    patient_sequences,
    n_episodes=100,
    n_agents=3
):
    """
    Compare personalized system against all baselines
    
    arguments:
        genomic_model: Trained genomic model
        patient_sequences: List of patient sequences
        n_episodes: Episodes per patient
        n_agents: Number of agents
    
    returns:
        results: Dict with all results
    """
    results = {
        'personalized': [],
        'random': [],
        'greedy': [],
        'non_personalized': [],
        'flat': []
    }
    
    n_patients = len(patient_sequences)
    
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    
    for i, patient_seq in enumerate(patient_sequences):
        print(f"\nPatient {i+1}/{n_patients}")
        print("-" * 70)
        
        # 1. Personalized (our system)
        print("  Training: Personalized (Hierarchical + Genomics)...")
        pers_rewards = train_personalized(
            genomic_model, patient_seq, n_agents, n_episodes
        )
        pers_performance = np.mean(pers_rewards[-10:])
        results['personalized'].append(pers_performance)
        print(f"    Final performance: {pers_performance:.2f}")
        
        # 2. Random baseline
        print("  Training: Random Agent...")
        env = NanoparticleEnv(n_agents=n_agents)
        random_agent = RandomAgent()
        random_rewards = train_baseline(random_agent, env, n_episodes=n_episodes)
        random_performance = np.mean(random_rewards[-10:])
        results['random'].append(random_performance)
        print(f"    Final performance: {random_performance:.2f}")
        
        # 3. Greedy baseline
        print("  Training: Greedy Agent...")
        env = NanoparticleEnv(n_agents=n_agents)
        greedy_agent = GreedyAgent()
        greedy_rewards = train_baseline(greedy_agent, env, n_episodes=n_episodes)
        greedy_performance = np.mean(greedy_rewards[-10:])
        results['greedy'].append(greedy_performance)
        print(f"    Final performance: {greedy_performance:.2f}")
        
        # 4. Non-personalized (average drug sensitivity)
        print("  Training: Non-Personalized Agent...")
        # Use base environment without genomic modulation
        env = NanoparticleEnv(n_agents=n_agents)
        agent = Hierarchical_agent()
        nonpers_rewards = train_baseline(agent, env, n_episodes=n_episodes)
        nonpers_performance = np.mean(nonpers_rewards[-10:])
        results['non_personalized'].append(nonpers_performance)
        print(f"    Final performance: {nonpers_performance:.2f}")
        
        # 5. Flat RL (no hierarchy)
        print("  Training: Flat RL Agent...")
        env = NanoparticleEnv(n_agents=n_agents)
        flat_agent = FlatAgent()
        flat_rewards = train_baseline(flat_agent, env, n_episodes=n_episodes)
        flat_performance = np.mean(flat_rewards[-10:])
        results['flat'].append(flat_performance)
        print(f"    Final performance: {flat_performance:.2f}")
        
        # Summary for this patient
        print(f"\n  Summary:")
        print(f"    Personalized: {pers_performance:.2f}")
        print(f"    vs Random: {(pers_performance - random_performance):.2f} improvement")
        print(f"    vs Greedy: {(pers_performance - greedy_performance):.2f} improvement")
        print(f"    vs Non-personalized: {(pers_performance - nonpers_performance):.2f} improvement")
        print(f"    vs Flat: {(pers_performance - flat_performance):.2f} improvement")
    
    return results


def compute_statistics(results):
    """
    Compute statistical comparisons
    
    arguments:
        results: Dict with baseline results
    
    returns:
        stats_df: DataFrame with statistics
    """
    personalized = np.array(results['personalized'])
    
    stats_rows = []
    
    for baseline_name, baseline_results in results.items():
        if baseline_name == 'personalized':
            continue
        
        baseline_vals = np.array(baseline_results)
        
        # Compute improvement
        improvements = personalized - baseline_vals
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        # Relative improvement
        rel_improvement = (mean_improvement / np.abs(np.mean(baseline_vals))) * 100
        
        # Statistical test (paired t-test)
        t_stat, p_value = stats.ttest_rel(personalized, baseline_vals)
        
        # Effect size (Cohen's d)
        cohens_d = mean_improvement / np.std(improvements)
        
        stats_rows.append({
            'baseline': baseline_name,
            'baseline_mean': np.mean(baseline_vals),
            'personalized_mean': np.mean(personalized),
            'absolute_improvement': mean_improvement,
            'relative_improvement_pct': rel_improvement,
            'std_improvement': std_improvement,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    return pd.DataFrame(stats_rows)

def visualize_comparison(results, stats_df, output_file='outputs/baseline_comparison.png'):
    """
    Create visualization of baseline comparison
    
    arguments:
        results: Dict with baseline results
        stats_df: Statistics DataFrame
        output_file: Output filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Bar chart of mean performance
    ax = axes[0, 0]
    
    baseline_names = list(results.keys())
    mean_performances = [np.mean(results[name]) for name in baseline_names]
    colors = ['green', 'red', 'orange', 'blue', 'purple']
    
    bars = ax.bar(range(len(baseline_names)), mean_performances, color=colors, alpha=0.7)
    ax.set_xticks(range(len(baseline_names)))
    ax.set_xticklabels(baseline_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Performance')
    ax.set_title('Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_performances)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Improvement over baselines
    ax = axes[0, 1]
    
    improvements = []
    baseline_labels = []
    
    for _, row in stats_df.iterrows():
        improvements.append(row['relative_improvement_pct'])
        baseline_labels.append(row['baseline'])
    
    bars = ax.barh(range(len(baseline_labels)), improvements, color='green', alpha=0.7)
    ax.set_yticks(range(len(baseline_labels)))
    ax.set_yticklabels(baseline_labels)
    ax.set_xlabel('Relative Improvement (%)')
    ax.set_title('Personalized vs Baselines')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', ha='left', va='center', fontsize=10)
    
    # Plot 3: Box plot of distributions
    ax = axes[1, 0]
    
    data_to_plot = [results[name] for name in baseline_names]
    bp = ax.boxplot(data_to_plot, labels=baseline_names, patch_artist=True)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(baseline_names, rotation=45, ha='right')
    ax.set_ylabel('Performance')
    ax.set_title('Performance Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Statistical significance
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create table
    table_data = []
    table_data.append(['Baseline', 'Improvement', 'p-value', 'Significant'])
    
    for _, row in stats_df.iterrows():
        table_data.append([
            row['baseline'],
            f"{row['relative_improvement_pct']:+.1f}%",
            f"{row['p_value']:.4f}",
            row['significant']
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color significant rows
    for i in range(1, len(table_data)):
        if table_data[i][3] == 'Yes':
            for j in range(4):
                table[(i, j)].set_facecolor('#90EE90')
    
    ax.set_title('Statistical Significance', fontsize=12, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    plt.close()



def main():
    print("=" * 70)
    print("BASELINE COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Load genomic model
    print("\n1. Loading genomic model...")
    genomic_model = GenomicViT(
        seq_length=1024,
        patch_size=16,
        in_channels=4,
        embed_dim=256,
        num_layers=8,
        num_heads=8,
        mlp_dim=1024,
        num_classes=1,
        dropout=0.1
    )
    
    try:
        genomic_model.load_state_dict(
            torch.load('data/genomic/processed/best_model.pth')
        )
        print("Trained model loaded!")
    except:
        print("No trained model, using random weights")
    
    genomic_model.eval()
    
    # Load patient sequences
    print("\n2. Loading patient data...")
    try:
        sequences = np.load('data/genomic/processed/sequences_onehot.npy')
        
        # Select 3 diverse patients (for speed)
        n_test_patients = 3
        patient_indices = np.linspace(0, len(sequences)-1, n_test_patients, dtype=int)
        test_sequences = [sequences[i] for i in patient_indices]
        
        print(f"Selected {n_test_patients} patients for testing")
    except:
        print(" Creating synthetic patients")
        n_test_patients = 3
        test_sequences = [np.random.rand(1024, 4) for _ in range(n_test_patients)]
        test_sequences = [s / s.sum(axis=1, keepdims=True) for s in test_sequences]
    
    # Run comparison
    print("\n3. Running baseline comparison...")
    print(f"   Training episodes: 50 (reduced for speed)")
    print(f"   Number of agents: 3")
    print()
    
    results = compare_all_baselines(
        genomic_model=genomic_model,
        patient_sequences=test_sequences,
        n_episodes=50,  # Reduced for speed
        n_agents=3
    )
    
    # Compute statistics
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    stats_df = compute_statistics(results)
    print("\n" + stats_df.to_string(index=False))
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save statistics
    stats_df.to_csv('outputs/baseline_statistics.csv', index=False)
    print("✓ Statistics saved to: outputs/baseline_statistics.csv")
    
    # Save raw results
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/baseline_raw_results.csv', index=False)
    print("✓ Raw results saved to: outputs/baseline_raw_results.csv")
    
    # Visualize
    visualize_comparison(results, stats_df)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nPersonalized System Performance: {np.mean(results['personalized']):.2f}")
    print("\nImprovements over baselines:")
    
    for _, row in stats_df.iterrows():
        significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  vs {row['baseline']:20s}: {row['relative_improvement_pct']:+6.1f}% {significance}")
    
    print("\n* p < 0.05, ** p < 0.01, *** p < 0.001")
    
    print("\n" + "=" * 70)
    print("✅ BASELINE COMPARISON COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()