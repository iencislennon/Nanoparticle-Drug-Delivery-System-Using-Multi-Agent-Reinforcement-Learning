"""
Ablation Studies

Test which components contribute most to performance:
1. Remove Genomic Personalization
2. Remove Hierarchical Structure
3. Remove Communication (Pheromones)
4. Simplify Tumor Model
5. Reduce Agent Count
6. Remove Uncertainty Estimation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import sys
sys.path.append('.')

from models.genomic_vit import GenomicViT
from rl.environment import NanoparticleEnv
from rl.personalized_env import PersonalizedNanoparticleEnv
from rl.hierarchical_agent import Hierarchical_agent
from data.genomic.dataset import GenomicDataset


# ablation configuration

class AblationConfig:
    """Configuration for ablation experiments"""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.use_genomic = True
        self.use_hierarchy = True
        self.use_communication = True
        self.use_complex_tumor = True
        self.n_agents = 5
        self.use_uncertainty = True
    
    def __repr__(self):
        return f"AblationConfig({self.name})"


def get_ablation_configs():
    """Define all ablation configurations"""
    configs = []
    
    # Full model (baseline)
    full = AblationConfig("full", "Full model with all components")
    configs.append(full)
    
    # Ablation 1: No genomic personalization
    no_genomic = AblationConfig("no_genomic", "Without genomic personalization")
    no_genomic.use_genomic = False
    configs.append(no_genomic)
    
    # Ablation 2: No hierarchy (flat RL)
    no_hierarchy = AblationConfig("no_hierarchy", "Flat RL without hierarchy")
    no_hierarchy.use_hierarchy = False
    configs.append(no_hierarchy)
    
    # Ablation 3: No communication
    no_comm = AblationConfig("no_communication", "Without pheromone communication")
    no_comm.use_communication = False
    configs.append(no_comm)
    
    # Ablation 4: Simplified tumor
    simple_tumor = AblationConfig("simple_tumor", "Simplified tumor model")
    simple_tumor.use_complex_tumor = False
    configs.append(simple_tumor)
    
    # Ablation 5: Fewer agents
    fewer_agents = AblationConfig("fewer_agents", "Only 3 agents instead of 5")
    fewer_agents.n_agents = 3
    configs.append(fewer_agents)
    
    return configs


# agent classes

class FlatRLAgent:
    """Simple flat Q-learning agent (no hierarchy)"""
    
    def __init__(self):
        self.q_values = {}
        self.epsilon = 0.1
        self.lr = 0.01
        self.gamma = 0.95
        self.action_space = [0, 1, 2, 3, 4]
    
    def _state_key(self, obs):
        return tuple(np.round(obs, 1))
    
    def act(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        state_key = self._state_key(obs)
        q_vals = [self.q_values.get((state_key, a), 0) for a in self.action_space]
        return self.action_space[np.argmax(q_vals)]
    
    def learn(self, obs, action, next_obs, reward, done):
        state_key = self._state_key(obs)
        next_key = self._state_key(next_obs)
        
        current_q = self.q_values.get((state_key, action), 0)
        next_q_vals = [self.q_values.get((next_key, a), 0) for a in self.action_space]
        max_next_q = max(next_q_vals)
        
        target = reward + (0 if done else self.gamma * max_next_q)
        new_q = current_q + self.lr * (target - current_q)
        self.q_values[(state_key, action)] = new_q


class NoCommAgent(Hierarchical_agent):
    """Hierarchical agent without communication"""
    
    def __init__(self):
        super().__init__()
        self.use_pheromones = False


# training

def train_ablation(config, genomic_model, patient_sequence, n_episodes=50):
    """
    Train with specific ablation configuration
    
    arguments:
        config: AblationConfig
        genomic_model: Trained genomic model
        patient_sequence: Patient DNA sequence
        n_episodes: Training episodes
    
    returns:
        episode_rewards: List of rewards per episode
    """
    # Create environment
    if config.use_genomic:
        env = PersonalizedNanoparticleEnv(
            genomic_model=genomic_model,
            patient_sequence=patient_sequence,
            n_agents=config.n_agents
        )
    else:
        env = NanoparticleEnv(n_agents=config.n_agents)
    
    # Create agent
    if config.use_hierarchy:
        if config.use_communication:
            agent = Hierarchical_agent()
        else:
            agent = NoCommAgent()
    else:
        agent = FlatRLAgent()
    
    # Training loop
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            actions = [agent.act(obs[i]) for i in range(config.n_agents)]
            next_obs, rewards, done, info = env.step(actions)
            
            for i in range(config.n_agents):
                agent.learn(obs[i], actions[i], next_obs[i], rewards[i], done)
            
            episode_reward += rewards.sum()
            obs = next_obs
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
    
    return episode_rewards


#  ablation analyses

def run_ablation_study(genomic_model, patient_sequences, n_episodes=50, n_trials=3):
    """
    Run complete ablation study
    
    arguments:
        genomic_model: Trained model
        patient_sequences: List of patient sequences
        n_episodes: Episodes per trial
        n_trials: Number of trials per ablation
    
    returns:
        results: DataFrame with all results
    """
    configs = get_ablation_configs()
    
    results = []
    
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print(f"\nConfigurations: {len(configs)}")
    print(f"Patients: {len(patient_sequences)}")
    print(f"Trials per config: {n_trials}")
    print(f"Episodes per trial: {n_episodes}")
    print()
    
    for config_idx, config in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"ABLATION {config_idx+1}/{len(configs)}: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*70}")
        
        config_results = []
        
        for patient_idx, patient_seq in enumerate(patient_sequences):
            print(f"\n  Patient {patient_idx+1}/{len(patient_sequences)}")
            
            trial_performances = []
            
            for trial in range(n_trials):
                print(f"    Trial {trial+1}/{n_trials}...", end=" ")
                
                episode_rewards = train_ablation(
                    config, genomic_model, patient_seq, n_episodes
                )
                
                # Final performance (last 10 episodes)
                final_perf = np.mean(episode_rewards[-10:])
                trial_performances.append(final_perf)
                
                print(f"Performance: {final_perf:.2f}")
            
            # Average across trials
            mean_perf = np.mean(trial_performances)
            std_perf = np.std(trial_performances)
            
            config_results.append({
                'config': config.name,
                'description': config.description,
                'patient_id': patient_idx,
                'mean_performance': mean_perf,
                'std_performance': std_perf,
                'trials': trial_performances
            })
            
            print(f"    Patient {patient_idx+1} Mean: {mean_perf:.2f} ± {std_perf:.2f}")
        
        results.extend(config_results)
    
    return pd.DataFrame(results)


def compute_component_importance(results_df):
    """
    Compute importance of each component
    
    arguments:
        results_df: Results from ablation study
    
    returns:
        importance_df: Component importance ranking
    """
    # get full model performance
    full_perf = results_df[results_df['config'] == 'full']['mean_performance'].mean()
    
    importance = []
    
    for config in results_df['config'].unique():
        if config == 'full':
            continue
        
        ablation_perf = results_df[results_df['config'] == config]['mean_performance'].mean()
        
        # Performance drop when component removed
        drop = full_perf - ablation_perf
        drop_pct = (drop / abs(full_perf)) * 100
        
        importance.append({
            'component': config,
            'full_performance': full_perf,
            'ablation_performance': ablation_perf,
            'performance_drop': drop,
            'drop_percentage': drop_pct,
            'importance_rank': 0  # Will be filled later
        })
    
    importance_df = pd.DataFrame(importance)
    
    # Rank by absolute drop
    importance_df = importance_df.sort_values('performance_drop', ascending=False)
    importance_df['importance_rank'] = range(1, len(importance_df) + 1)
    
    return importance_df


# visualization

def visualize_ablations(results_df, importance_df, output_file='outputs/ablation_results.png'):
    """Create visualization of ablation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Performance by configuration
    ax = axes[0, 0]
    
    config_means = results_df.groupby('config')['mean_performance'].mean()
    config_stds = results_df.groupby('config')['mean_performance'].std()
    
    configs = config_means.index.tolist()
    means = config_means.values
    stds = config_stds.values
    
    colors = ['green' if c == 'full' else 'red' for c in configs]
    
    bars = ax.bar(range(len(configs)), means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Mean Performance')
    ax.set_title('Performance by Configuration')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight full model
    ax.axhline(y=config_means['full'], color='green', linestyle='--', linewidth=2, alpha=0.5, label='Full Model')
    ax.legend()
    
    # Plot 2: Component importance
    ax = axes[0, 1]
    
    components = importance_df['component'].values
    drops = importance_df['drop_percentage'].values
    
    bars = ax.barh(range(len(components)), drops, color='orange', alpha=0.7)
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components)
    ax.set_xlabel('Performance Drop When Removed (%)')
    ax.set_title('Component Importance')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, drops)):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=10)
    
    # Plot 3: Performance distribution
    ax = axes[1, 0]
    
    data_to_plot = [results_df[results_df['config'] == c]['mean_performance'].values 
                    for c in configs]
    
    bp = ax.boxplot(data_to_plot, labels=configs, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Performance')
    ax.set_title('Performance Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Ranking table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [['Rank', 'Component', 'Drop %']]
    
    for _, row in importance_df.iterrows():
        table_data.append([
            f"#{int(row['importance_rank'])}",
            row['component'],
            f"{row['drop_percentage']:.1f}%"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color by rank
    for i in range(1, len(table_data)):
        if i == 1:  # Most important
            for j in range(3):
                table[(i, j)].set_facecolor('#FFD700')  # Gold
        elif i == 2:
            for j in range(3):
                table[(i, j)].set_facecolor('#C0C0C0')  # Silver
        elif i == 3:
            for j in range(3):
                table[(i, j)].set_facecolor('#CD7F32')  # Bronze
    
    ax.set_title('Component Importance Ranking', fontsize=12, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n Visualization saved to: {output_file}")
    plt.close()


#main

def main():
    print("=" * 70)
    print("ABLATION STUDY ANALYSIS")
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
        
        # Select 2 patients for speed (ablations are expensive!)
        n_test_patients = 2
        patient_indices = [0, len(sequences)//2]  # First and middle patient
        test_sequences = [sequences[i] for i in patient_indices]
        
        print(f"Selected {n_test_patients} patients for testing")
    except:
        print("Creating synthetic patients")
        n_test_patients = 2
        test_sequences = [np.random.rand(1024, 4) for _ in range(n_test_patients)]
        test_sequences = [s / s.sum(axis=1, keepdims=True) for s in test_sequences]
    
    # Run ablation study
    print("\n3. Running ablation study...")
    print("   (This will take 15-20 minutes)")
    print()
    
    results_df = run_ablation_study(
        genomic_model=genomic_model,
        patient_sequences=test_sequences,
        n_episodes=50,  # Reduced for speed
        n_trials=2      # Reduced for speed
    )
    
    # Compute importance
    print("\n" + "=" * 70)
    print("COMPUTING COMPONENT IMPORTANCE")
    print("=" * 70)
    
    importance_df = compute_component_importance(results_df)
    
    print("\n" + importance_df.to_string(index=False))
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    results_df.to_csv('outputs/ablation_raw_results.csv', index=False)
    print("Raw results saved to: outputs/ablation_raw_results.csv")
    
    importance_df.to_csv('outputs/ablation_importance.csv', index=False)
    print("Importance saved to: outputs/ablation_importance.csv")
    
    # Visualize
    visualize_ablations(results_df, importance_df)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    full_perf = results_df[results_df['config'] == 'full']['mean_performance'].mean()
    print(f"\nFull Model Performance: {full_perf:.2f}")
    
    print("\nComponent Importance (by performance drop):")
    for _, row in importance_df.iterrows():
        print(f"  #{int(row['importance_rank'])} {row['component']:20s}: "
              f"{row['drop_percentage']:+6.1f}% drop")
    
    print("\n" + "=" * 70)
    print("TABLATION STUDY COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()