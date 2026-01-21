"""
Train Personalized RL Agents

train nanoparticle agents on multiple patients with different
genomic profiles to demonstrate personalization.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

from models.genomic_vit import GenomicViT
from rl.personalized_env import PersonalizedNanoparticleEnv
from rl.hierarchical_agent import Hierarchical_agent


def train_agent_for_patient(
    env,
    agent,
    n_episodes=50,
    verbose=False
):
    """
    Train agent on personalized environment
    
    arguments:
        env: PersonalizedNanoparticleEnv
        agent: Hierarchical_agent
        n_episodes: Number of training episodes
        verbose: Print progress
    
    returns:
        episode_rewards: List of rewards per episode
    """
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            # actions
            actions = [agent.act(obs[i]) for i in range(len(obs))]
            
            # step environment
            next_obs, rewards, done, info = env.step(actions)
            
            # learning
            for i in range(len(obs)):
                agent.learn(obs[i], actions[i], next_obs[i], rewards[i], done)
            
            episode_reward += rewards.sum()
            obs = next_obs
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"   Episode {episode+1:3d}: Avg Reward = {avg_reward:.2f}")
    
    return episode_rewards


def main():
    print("=" * 70)
    print("PERSONALIZED MULTI-PATIENT RL TRAINING")
    print("=" * 70)
    
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
    except Exception as e:
        print(f"   ⚠️  Could not load model: {e}")
        print("   Using random weights for demonstration")
    
    genomic_model.eval()
    
    # ===== LOAD PATIENT DATA =====
    print("\n2. Loading patient genomic data...")
    try:
        sequences = np.load('data/genomic/processed/sequences_onehot.npy')
        features = pd.read_csv('data/genomic/processed/patient_features.csv')
        
        # Select 5 diverse patients
        n_patients = min(5, len(sequences))
        patient_indices = np.linspace(0, len(sequences)-1, n_patients, dtype=int)
        
        print(f"   ✓ Loaded data for {n_patients} patients")
    except Exception as e:
        print(f"   ⚠️  Could not load data: {e}")
        print("   Generating synthetic patients")
        
        n_patients = 5
        patient_indices = range(n_patients)
        sequences = [np.random.rand(1024, 4) for _ in range(n_patients)]
        # Normalize
        sequences = [s / s.sum(axis=1, keepdims=True) for s in sequences]
    
    # ===== TRAIN AGENTS FOR EACH PATIENT =====
    print("\n3. Training personalized agents...")
    print("-" * 70)
    
    results = []
    
    for i, patient_idx in enumerate(patient_indices):
        patient_sequence = sequences[patient_idx]
        
        print(f"\n📊 Patient {i+1}/{n_patients} (ID: {patient_idx})")
        print("-" * 70)
        
        # Create personalized environment
        env = PersonalizedNanoparticleEnv(
            genomic_model=genomic_model,
            patient_sequence=patient_sequence,
            n_agents=3  # Reduced for speed
        )
        
        # Get patient info
        patient_info = env.get_patient_info()
        print(f"Responder Type: {patient_info['responder_type']}")
        print(f"Strategy: {patient_info['recommended_strategy']}")
        
        # Create fresh agent for this patient
        agent = Hierarchical_agent()
        
        # Train
        print("\nTraining...")
        episode_rewards = train_agent_for_patient(
            env,
            agent,
            n_episodes=50,  # Reduced for speed
            verbose=True
        )
        
        # Compute improvement
        initial_performance = np.mean(episode_rewards[:10])
        final_performance = np.mean(episode_rewards[-10:])
        improvement = ((final_performance - initial_performance) / 
                      abs(initial_performance) * 100)
        
        print(f"\nResults:")
        print(f"   Initial: {initial_performance:.2f}")
        print(f"   Final: {final_performance:.2f}")
        print(f"   Improvement: {improvement:+.1f}%")
        
        # Store results
        results.append({
            'patient_id': patient_idx,
            'drug_sensitivity': patient_info['drug_sensitivity'],
            'responder_type': patient_info['responder_type'],
            'initial_reward': initial_performance,
            'final_reward': final_performance,
            'improvement_pct': improvement,
            'episode_rewards': episode_rewards
        })
    
    # ===== ANALYSIS =====
    print("\n" + "=" * 70)
    print("MULTI-PATIENT ANALYSIS")
    print("=" * 70)
    
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'episode_rewards'}
        for r in results
    ])
    
    print("\n" + str(results_df.to_string(index=False)))
    
    # Average improvement by responder type
    print("\n" + "-" * 70)
    print("AVERAGE IMPROVEMENT BY RESPONDER TYPE:")
    print("-" * 70)
    
    for responder_type in results_df['responder_type'].unique():
        subset = results_df[results_df['responder_type'] == responder_type]
        avg_improvement = subset['improvement_pct'].mean()
        avg_final = subset['final_reward'].mean()
        
        print(f"{responder_type:20s}: {avg_improvement:+6.1f}% improvement, "
              f"Final Reward: {avg_final:7.2f}")
    
    # ===== VISUALIZATION =====
    print("\n4. Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Learning curves by responder type
    ax1 = axes[0]
    colors = {'High Responder': 'green', 'Medium Responder': 'orange', 
              'Poor Responder': 'red'}
    
    for result in results:
        responder = result['responder_type']
        ax1.plot(
            result['episode_rewards'],
            label=f"Patient {result['patient_id']} ({responder})",
            color=colors.get(responder, 'blue'),
            alpha=0.7
        )
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Curves by Patient Type')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final performance vs drug sensitivity
    ax2 = axes[1]
    sensitivities = [r['drug_sensitivity'] for r in results]
    final_rewards = [r['final_reward'] for r in results]
    responder_types = [r['responder_type'] for r in results]
    
    for sens, reward, resp in zip(sensitivities, final_rewards, responder_types):
        ax2.scatter(sens, reward, s=200, alpha=0.6, 
                   c=colors.get(resp, 'blue'),
                   label=resp if resp not in ax2.get_legend_handles_labels()[1] else "")
    
    ax2.set_xlabel('Drug Sensitivity (Genomic Prediction)')
    ax2.set_ylabel('Final RL Performance')
    ax2.set_title('Genomic Sensitivity vs RL Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/personalized_rl_results.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: outputs/personalized_rl_results.png")
    
    # ===== SAVE RESULTS =====
    results_df.to_csv('outputs/personalized_rl_results.csv', index=False)
    print("   ✓ Saved: outputs/personalized_rl_results.csv")
    
    print("\n" + "=" * 70)
    print("✅ PERSONALIZED MULTI-PATIENT TRAINING COMPLETE!")
    print("=" * 70)
    
    print("\n📊 KEY FINDINGS:")
    print(f"   - Trained {n_patients} personalized agents")
    print(f"   - Average improvement: {results_df['improvement_pct'].mean():+.1f}%")
    print(f"   - Best performer: {results_df.loc[results_df['final_reward'].idxmax(), 'responder_type']}")
    print(f"   - Correlation (sensitivity vs performance): "
          f"{np.corrcoef(sensitivities, final_rewards)[0,1]:.3f}")


if __name__ == "__main__":
    main()