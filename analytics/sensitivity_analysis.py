"""
Parameter Sensitivity Analysis

Test robustness to parameter variations:
1. Nanoparticle Properties (size, coating, loading)
2. Drug Dosage
3. Tumor Characteristics
4. RL Hyperparameters
5. Agent Configuration
6. Genomic Model Architecture
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



def get_parameter_sweeps():

    sweeps = {
        'nanoparticle_size': {
            'values': [10, 50, 100, 200, 500],
            'units': 'nm',
            'description': 'Nanoparticle diameter'
        },
        'drug_loading': {
            'values': [0.1, 0.3, 0.5, 0.7, 0.9],
            'units': 'fraction',
            'description': 'Drug payload fraction'
        },
        'n_agents': {
            'values': [1, 3, 5, 10, 20],
            'units': 'count',
            'description': 'Number of nanoparticle agents'
        },
        'learning_rate': {
            'values': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            'units': 'lr',
            'description': 'Agent learning rate'
        },
        'epsilon': {
            'values': [0.05, 0.1, 0.2, 0.3, 0.4],
            'units': 'ε',
            'description': 'Exploration rate'
        },
        'discount_factor': {
            'values': [0.90, 0.95, 0.97, 0.99, 0.995],
            'units': 'γ',
            'description': 'Future reward discount'
        }
    }
    
    return sweeps



def train_with_parameter(
    param_name,
    param_value,
    genomic_model,
    patient_sequence,
    n_episodes=50
):
    """
    Train agent with specific parameter value
    
    arguments:
        param_name: Name of parameter to vary
        param_value: Value of parameter
        genomic_model: Trained genomic model
        patient_sequence: Patient DNA sequence
        n_episodes: Training episodes
    
    returns:
        final_performance: Mean reward of last 10 episodes
    """
    # creating environment
    if param_name == 'n_agents':
        n_agents = int(param_value)
    else:
        n_agents = 3  
    
    env = PersonalizedNanoparticleEnv(
        genomic_model=genomic_model,
        patient_sequence=patient_sequence,
        n_agents=n_agents
    )
    
    # create agent with parameter
    agent = Hierarchical_agent()
    
    if param_name == 'learning_rate':
        agent.learning_rate = param_value
    elif param_name == 'epsilon':
        agent.epsilon = param_value
    elif param_name == 'discount_factor':
        agent.gamma = param_value
    
    # training loop
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            actions = [agent.act(obs[i]) for i in range(n_agents)]
            next_obs, rewards, done, info = env.step(actions)
            
            for i in range(n_agents):
                agent.learn(obs[i], actions[i], next_obs[i], rewards[i], done)
            
            episode_reward += rewards.sum()
            obs = next_obs
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
    
    final_performance = np.mean(episode_rewards[-10:])
    
    return final_performance



def run_parameter_sweep(
    param_name,
    param_values,
    genomic_model,
    patient_sequences,
    n_episodes=50,
    n_trials=2
):
    """
    Sweep through parameter values
    
    arguments:
        param_name: Parameter to sweep
        param_values: List of values to test
        genomic_model: Trained model
        patient_sequences: List of patient sequences
        n_episodes: Episodes per trial
        n_trials: Trials per value
    
    returns:
        results: List of dicts with results
    """
    results = []
    
    print(f"\n{'='*70}")
    print(f"PARAMETER SWEEP: {param_name}")
    print(f"{'='*70}")
    print(f"Values: {param_values}")
    print()
    
    for value_idx, value in enumerate(param_values):
        print(f"  Value {value_idx+1}/{len(param_values)}: {value}")
        
        value_results = []
        
        for patient_idx, patient_seq in enumerate(patient_sequences):
            trial_performances = []
            
            for trial in range(n_trials):
                perf = train_with_parameter(
                    param_name, value, genomic_model, 
                    patient_seq, n_episodes
                )
                trial_performances.append(perf)
            
            mean_perf = np.mean(trial_performances)
            std_perf = np.std(trial_performances)
            
            value_results.append(mean_perf)
            
            print(f"    Patient {patient_idx+1}: {mean_perf:.2f} ± {std_perf:.2f}")
        
        # Average across patients
        overall_mean = np.mean(value_results)
        overall_std = np.std(value_results)
        
        results.append({
            'parameter': param_name,
            'value': value,
            'mean_performance': overall_mean,
            'std_performance': overall_std,
            'patient_performances': value_results
        })
        
        print(f"    Overall: {overall_mean:.2f} ± {overall_std:.2f}\n")
    
    return results



def compute_sensitivity(sweep_results):
    """
    Compute sensitivity metrics
    
    arguments:
        sweep_results: Results from parameter sweep
    
    returns:
        sensitivity_metrics: Dict with sensitivity info
    """
    param_name = sweep_results[0]['parameter']
    values = [r['value'] for r in sweep_results]
    performances = [r['mean_performance'] for r in sweep_results]
    
    # Sensitivity = variance / mean
    performance_variance = np.var(performances)
    performance_mean = np.mean(performances)
    sensitivity_score = performance_variance / abs(performance_mean)
    
    # Best value
    best_idx = np.argmax(performances)
    best_value = values[best_idx]
    best_performance = performances[best_idx]
    
    # Worst value
    worst_idx = np.argmin(performances)
    worst_value = values[worst_idx]
    worst_performance = performances[worst_idx]
    
    # Range
    performance_range = best_performance - worst_performance
    relative_range = (performance_range / abs(performance_mean)) * 100
    
    return {
        'parameter': param_name,
        'sensitivity_score': sensitivity_score,
        'best_value': best_value,
        'best_performance': best_performance,
        'worst_value': worst_value,
        'worst_performance': worst_performance,
        'performance_range': performance_range,
        'relative_range_pct': relative_range,
        'values': values,
        'performances': performances
    }


def visualize_sensitivity(all_sensitivity, output_file='outputs/sensitivity_analysis.png'):
    """
    create visualization of sensitivity analysis
    """
    n_params = len(all_sensitivity)
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, sens in enumerate(all_sensitivity):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        values = sens['values']
        performances = sens['performances']
        
        # Plot performance vs parameter value
        ax.plot(values, performances, 'o-', linewidth=2, markersize=8, color='blue', alpha=0.7)
        
        # Highlight best
        best_idx = np.argmax(performances)
        ax.plot(values[best_idx], performances[best_idx], 'o', 
                markersize=12, color='green', label='Best')
        
        # Highlight worst
        worst_idx = np.argmin(performances)
        ax.plot(values[worst_idx], performances[worst_idx], 'o', 
                markersize=12, color='red', label='Worst')
        
        ax.set_xlabel(f"{sens['parameter']}")
        ax.set_ylabel('Performance')
        ax.set_title(f"{sens['parameter']}\n(Range: {sens['relative_range_pct']:.1f}%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add optimal value text
        ax.text(0.05, 0.95, f"Optimal: {sens['best_value']}", 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Sensitivity visualization saved to: {output_file}")
    plt.close()


def visualize_sensitivity_comparison(all_sensitivity, output_file='outputs/sensitivity_comparison.png'):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Sensitivity scores
    ax = axes[0]
    
    param_names = [s['parameter'] for s in all_sensitivity]
    sensitivity_scores = [s['sensitivity_score'] for s in all_sensitivity]
    
    bars = ax.barh(range(len(param_names)), sensitivity_scores, color='orange', alpha=0.7)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Sensitivity Score (Variance/Mean)')
    ax.set_title('Parameter Sensitivity Comparison')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sensitivity_scores)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=10)
    
    # Plot 2: Performance ranges
    ax = axes[1]
    
    relative_ranges = [s['relative_range_pct'] for s in all_sensitivity]
    
    bars = ax.barh(range(len(param_names)), relative_ranges, color='purple', alpha=0.7)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Performance Range (%)')
    ax.set_title('Performance Variability')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, relative_ranges)):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to: {output_file}")
    plt.close()



def main():
    print("=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
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
        print("   ✓ Trained model loaded!")
    except:
        print("   ⚠️  No trained model, using random weights")
    
    genomic_model.eval()
    
    print("\n2. Loading patient data...")
    try:
        sequences = np.load('data/genomic/processed/sequences_onehot.npy')

        n_test_patients = 2
        patient_indices = [0, len(sequences)//2]
        test_sequences = [sequences[i] for i in patient_indices]
        
        print(f"   ✓ Selected {n_test_patients} patients for testing")
    except:
        print("   ⚠️  Creating synthetic patients")
        n_test_patients = 2
        test_sequences = [np.random.rand(1024, 4) for _ in range(n_test_patients)]
        test_sequences = [s / s.sum(axis=1, keepdims=True) for s in test_sequences]
    
    # Get parameter sweeps
    print("\n3. Loading parameter sweep configurations...")
    sweeps = get_parameter_sweeps()
    
    # Select subset for speed (full analysis would take hours)
    params_to_test = ['n_agents', 'learning_rate', 'epsilon']
    
    print(f"   Testing parameters: {params_to_test}")
    print("   (Reduced set for speed - full analysis available in config)")
    print()
    
    # Run sweeps
    print("=" * 70)
    print("RUNNING PARAMETER SWEEPS")
    print("=" * 70)
    print("(This will take 15-20 minutes)")
    
    all_sweep_results = []
    all_sensitivity = []
    
    for param_name in params_to_test:
        sweep_config = sweeps[param_name]
        
        sweep_results = run_parameter_sweep(
            param_name=param_name,
            param_values=sweep_config['values'],
            genomic_model=genomic_model,
            patient_sequences=test_sequences,
            n_episodes=30,  
            n_trials=2
        )
        
        all_sweep_results.extend(sweep_results)
        
        # Compute sensitivity
        sensitivity = compute_sensitivity(sweep_results)
        all_sensitivity.append(sensitivity)
    
    # Create summary DataFrame
    print("\n" + "=" * 70)
    print("SENSITIVITY SUMMARY")
    print("=" * 70)
    
    summary_data = []
    for sens in all_sensitivity:
        summary_data.append({
            'parameter': sens['parameter'],
            'sensitivity_score': sens['sensitivity_score'],
            'best_value': sens['best_value'],
            'best_performance': sens['best_performance'],
            'relative_range_pct': sens['relative_range_pct']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('sensitivity_score', ascending=False)
    
    print("\n" + summary_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    sweep_df = pd.DataFrame(all_sweep_results)
    sweep_df.to_csv('outputs/sensitivity_raw_results.csv', index=False)
    print("✓ Raw results saved to: outputs/sensitivity_raw_results.csv")

    summary_df.to_csv('outputs/sensitivity_summary.csv', index=False)
    print("✓ Summary saved to: outputs/sensitivity_summary.csv")

    visualize_sensitivity(all_sensitivity)
    visualize_sensitivity_comparison(all_sensitivity)

    print("\n" + "=" * 70)
    print("OPTIMAL PARAMETERS")
    print("=" * 70)
    
    for sens in all_sensitivity:
        print(f"\n{sens['parameter']}:")
        print(f"  Optimal value: {sens['best_value']}")
        print(f"  Performance at optimal: {sens['best_performance']:.2f}")
        print(f"  Performance range: {sens['relative_range_pct']:.1f}%")
        print(f"  Sensitivity score: {sens['sensitivity_score']:.4f}")

    print("\n" + "=" * 70)
    print("ROBUSTNESS ASSESSMENT")
    print("=" * 70)
    
    avg_sensitivity = np.mean([s['sensitivity_score'] for s in all_sensitivity])
    avg_range = np.mean([s['relative_range_pct'] for s in all_sensitivity])
    
    print(f"\nAverage sensitivity score: {avg_sensitivity:.4f}")
    print(f"Average performance range: {avg_range:.1f}%")
    
    if avg_range < 10:
        robustness = "HIGH"
    elif avg_range < 20:
        robustness = "MEDIUM"
    else:
        robustness = "LOW"
    
    print(f"\nOverall robustness: {robustness}")
    print(f"  (Based on average {avg_range:.1f}% performance variation)")
    
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()