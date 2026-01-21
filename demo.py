"""
End-to-End Interactive Demo

Demonstrates complete pipeline:
1. Patient Selection
2. Genomic Analysis
3. Personalized Environment Creation
4. RL Agent Training
5. Treatment Simulation
6. Results Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import sys
import time

sys.path.append('.')

from models.genomic_vit import GenomicViT
from rl.personalized_env import PersonalizedNanoparticleEnv
from rl.hierarchical_agent import Hierarchical_agent


# utility functions

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def print_section(text):
    """Print formatted section"""
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


def progress_bar(current, total, bar_length=40):
    """Display progress bar"""
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r[{bar}] {percent*100:.1f}% ({current}/{total})", end='', flush=True)


# patient selection

def select_patient():
    """
    Interactive patient selection
    
    returns:
        patient_data: Dict with patient info
    """
    print_header("PATIENT SELECTION")
    
    # Load patient data
    try:
        features = pd.read_csv('data/genomic/processed/patient_features.csv')
        sequences = np.load('data/genomic/processed/sequences_onehot.npy')
        
        # Select diverse patients
        n_display = 5
        indices = np.linspace(0, len(features)-1, n_display, dtype=int)
        
        print("\nAvailable patients:")
        print()
        
        for i, idx in enumerate(indices):
            patient = features.iloc[idx]
            print(f"  {i+1}. Patient {idx}")
            print(f"     Age: {patient['age']:.0f} years")
            print(f"     Cancer Stage: {patient['stage']:.0f}")
            print(f"     Mutation Count: {patient['mutation_count']:.0f}")
            print(f"     Drug Response: {patient['drug_response']:.2f}")
            print()
        
        # User selection
        while True:
            try:
                choice = input("Select patient (1-5) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print("\nExiting demo...")
                    sys.exit(0)
                
                choice = int(choice)
                if 1 <= choice <= n_display:
                    selected_idx = indices[choice - 1]
                    break
                else:
                    print("Invalid choice. Please enter 1-5.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        patient_data = {
            'index': selected_idx,
            'features': features.iloc[selected_idx],
            'sequence': sequences[selected_idx],
            'age': features.iloc[selected_idx]['age'],
            'stage': features.iloc[selected_idx]['stage'],
            'mutations': features.iloc[selected_idx]['mutation_count'],
            'true_response': features.iloc[selected_idx]['drug_response']
        }
        
        print(f"\n✓ Selected Patient {selected_idx}")
        
    except Exception as e:
        print(f"\n⚠️  Could not load patient data: {e}")
        print("Using synthetic patient...")
        
        patient_data = {
            'index': 0,
            'sequence': np.random.rand(1024, 4),
            'age': 55,
            'stage': 2,
            'mutations': 150,
            'true_response': 0.65
        }
        
        # normalize sequence
        patient_data['sequence'] = patient_data['sequence'] / \
                                  patient_data['sequence'].sum(axis=1, keepdims=True)
    
    return patient_data

#genomic analyses

def analyze_genomics(patient_data, genomic_model):
    """
    Perform genomic analysis
    
    arguments:
        patient_data: Patient information
        genomic_model: Trained genomic model
    
    returns:
        genomic_results: Dict with predictions
    """
    print_section("GENOMIC ANALYSIS")
    
    print("\nAnalyzing patient DNA sequence...")
    time.sleep(1)  # Simulate processing
    
    # Prepare input
    sequence = torch.tensor(
        patient_data['sequence'],
        dtype=torch.float32
    ).unsqueeze(0)
    
    # Predict
    genomic_model.eval()
    with torch.no_grad():
        drug_response, uncertainty = genomic_model(sequence)
        
        sensitivity = drug_response.item()
        confidence = 1 - uncertainty.item()
    
    # Classify responder type
    if sensitivity > 0.7:
        responder_type = "High Responder"
        color = "🟢"
        prognosis = "Excellent"
        strategy = "Standard nanoparticle therapy"
    elif sensitivity > 0.5:
        responder_type = "Medium Responder"
        color = "🟡"
        prognosis = "Good"
        strategy = "Enhanced targeting with close monitoring"
    else:
        responder_type = "Poor Responder"
        color = "🔴"
        prognosis = "Guarded"
        strategy = "High-dose combination or alternative therapy"
    
    # Display results
    print("\n" + "="*50)
    print("GENOMIC ANALYSIS RESULTS")
    print("="*50)
    print(f"\n  {color} Drug Sensitivity:    {sensitivity:.3f}")
    print(f"Confidence:         {confidence:.3f}")
    print(f"Classification:     {responder_type}")
    print(f"Prognosis:          {prognosis}")
    print(f"Strategy:           {strategy}")
    print("\n" + "="*50)
    
    genomic_results = {
        'sensitivity': sensitivity,
        'confidence': confidence,
        'responder_type': responder_type,
        'strategy': strategy
    }
    
    return genomic_results


# rl training

def train_personalized_agent(
    genomic_model,
    patient_sequence,
    n_episodes=30,
    n_agents=3
):
    """
    Train personalized RL agent
    
    arguments:
        genomic_model: Trained model
        patient_sequence: Patient DNA
        n_episodes: Training episodes
        n_agents: Number of agents
    
    returns:
        agent: Trained agent
        env: Personalized environment
        episode_rewards: Training history
    """
    print_section("PERSONALIZED RL TRAINING")
    
    print(f"\nCreating personalized environment...")
    env = PersonalizedNanoparticleEnv(
        genomic_model=genomic_model,
        patient_sequence=patient_sequence,
        n_agents=n_agents
    )
    
    print(f"Initializing hierarchical agent...")
    agent = Hierarchical_agent()
    
    print(f"\nTraining for {n_episodes} episodes...")
    print("(This may take 1-2 minutes)\n")
    
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
        
        # Progress bar
        progress_bar(episode + 1, n_episodes)
    
    print("\n")
    
    # Results
    initial_perf = np.mean(episode_rewards[:10])
    final_perf = np.mean(episode_rewards[-10:])
    improvement = ((final_perf - initial_perf) / abs(initial_perf)) * 100
    
    print(f"\n  Initial Performance: {initial_perf:.2f}")
    print(f"Final Performance:  {final_perf:.2f}")
    print(f"Improvement:        {improvement:+.1f}%")
    
    return agent, env, episode_rewards


# treatment simulation

def simulate_treatment(agent, env, n_agents=3):
    """
    Simulate treatment with trained agent
    
    arguments:
        agent: Trained agent
        env: Environment
        n_agents: Number of agents
    
    returns:
        trajectory: List of states
        metrics: Performance metrics
    """
    print_section("TREATMENT SIMULATION")
    
    print("\nRunning treatment simulation...")
    
    obs = env.reset()
    trajectory = []
    
    total_reward = 0
    
    for step in range(1000):
        actions = [agent.act(obs[i]) for i in range(n_agents)]
        next_obs, rewards, done, info = env.step(actions)
        
        # Store trajectory
        trajectory.append({
            'step': step,
            'positions': obs.copy() if hasattr(obs, 'copy') else obs,
            'rewards': rewards,
            'actions': actions
        })
        
        total_reward += rewards.sum()
        obs = next_obs
        
        if done or step >= 500:  # Limit for demo
            break
    
    print(f"  Simulation completed: {len(trajectory)} steps")
    print(f"  Total reward: {total_reward:.2f}")
    
    metrics = {
        'total_reward': total_reward,
        'steps': len(trajectory),
        'final_positions': obs
    }
    
    return trajectory, metrics


# visualization

def visualize_results(
    patient_data,
    genomic_results,
    episode_rewards,
    trajectory,
    metrics
):
    """
    Create comprehensive visualization
    
    Args:
        patient_data: Patient info
        genomic_results: Genomic predictions
        episode_rewards: Training history
        trajectory: Treatment simulation
        metrics: Performance metrics
    """
    print_section("RESULTS VISUALIZATION")
    
    print("\nCreating visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Patient Profile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    profile_text = f"""
    PATIENT PROFILE
    ━━━━━━━━━━━━━━━━━━
    
    Patient ID: {patient_data['index']}
    Age: {patient_data['age']:.0f} years
    Stage: {patient_data['stage']:.0f}
    Mutations: {patient_data['mutations']:.0f}
    
    True Response: {patient_data['true_response']:.2f}
    """
    
    ax1.text(0.1, 0.5, profile_text, fontsize=11, family='monospace',
             verticalalignment='center')
    ax1.set_title('Patient Information', fontsize=12, weight='bold')
    
    # Plot 2: Genomic Prediction
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    genomic_text = f"""
    GENOMIC ANALYSIS
    ━━━━━━━━━━━━━━━━━━
    
    Predicted Sensitivity:
        {genomic_results['sensitivity']:.3f}
    
    Classification:
        {genomic_results['responder_type']}
    
    Confidence:
        {genomic_results['confidence']:.3f}
    
    Strategy:
        {genomic_results['strategy'][:30]}...
    """
    
    ax2.text(0.1, 0.5, genomic_text, fontsize=11, family='monospace',
             verticalalignment='center')
    ax2.set_title('Genomic Prediction', fontsize=12, weight='bold')
    
    # Plot 3: Treatment Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    metrics_text = f"""
    TREATMENT RESULTS
    ━━━━━━━━━━━━━━━━━━
    
    Total Reward:
        {metrics['total_reward']:.2f}
    
    Simulation Steps:
        {metrics['steps']}
    
    RL Performance:
        {episode_rewards[-1]:.2f}
    
    Improvement:
        {((episode_rewards[-1] - episode_rewards[0])/abs(episode_rewards[0])*100):+.1f}%
    """
    
    ax3.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center')
    ax3.set_title('Treatment Metrics', fontsize=12, weight='bold')
    
    # Plot 4: Learning Curve
    ax4 = fig.add_subplot(gs[1, :])
    
    ax4.plot(episode_rewards, linewidth=2, color='blue', alpha=0.7)
    ax4.axhline(y=np.mean(episode_rewards[:5]), color='red', 
                linestyle='--', label='Initial', alpha=0.5)
    ax4.axhline(y=np.mean(episode_rewards[-5:]), color='green',
                linestyle='--', label='Final', alpha=0.5)
    
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Total Reward', fontsize=11)
    ax4.set_title('RL Learning Progress', fontsize=12, weight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Sensitivity vs Response
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Comparison
    categories = ['Predicted\nSensitivity', 'True\nResponse', 'RL\nPerformance']
    values = [
        genomic_results['sensitivity'],
        patient_data['true_response'],
        (episode_rewards[-1] + 600) / 600  # Normalize to 0-1
    ]
    colors = ['blue', 'green', 'orange']
    
    bars = ax5.bar(categories, values, color=colors, alpha=0.7)
    ax5.set_ylabel('Score (0-1)', fontsize=11)
    ax5.set_title('Prediction vs Reality', fontsize=12, weight='bold')
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 6: Reward Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    
    rewards_list = [t['rewards'].sum() for t in trajectory]
    ax6.hist(rewards_list, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Step Reward', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Reward Distribution', fontsize=12, weight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # Determine success
    if genomic_results['sensitivity'] > 0.7:
        outcome = "Excellent Candidate"
        outcome_color = 'green'
    elif genomic_results['sensitivity'] > 0.5:
        outcome = "Good Candidate"
        outcome_color = 'orange'
    else:
        outcome = "Consider Alternatives"
        outcome_color = 'red'
    
    summary_text = f"""
    CLINICAL DECISION
    ━━━━━━━━━━━━━━━━━━
    
    {outcome}
    
    Recommended:
      {genomic_results['strategy'][:25]}
      {genomic_results['strategy'][25:50]}
    
    Expected Response:
      {genomic_results['sensitivity']*100:.0f}%
    
    Monitoring:
      Standard protocol
    """
    
    ax7.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', color=outcome_color)
    ax7.set_title('Clinical Summary', fontsize=12, weight='bold')
    
    plt.savefig('outputs/demo_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved: outputs/demo_results.png")
    
    plt.show()


# export results

def export_results(
    patient_data,
    genomic_results,
    metrics,
    output_file='outputs/demo_report.txt'
):
    """
    Export comprehensive report
    
    arguments:
        patient_data: Patient info
        genomic_results: Predictions
        metrics: Treatment metrics
        output_file: Output filename
    """
    print_section("EXPORTING RESULTS")
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PERSONALIZED NANOPARTICLE TREATMENT REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("PATIENT INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Patient ID: {patient_data['index']}\n")
        f.write(f"Age: {patient_data['age']:.0f} years\n")
        f.write(f"Cancer Stage: {patient_data['stage']:.0f}\n")
        f.write(f"Mutation Count: {patient_data['mutations']:.0f}\n\n")
        
        f.write("GENOMIC ANALYSIS\n")
        f.write("-"*70 + "\n")
        f.write(f"Predicted Drug Sensitivity: {genomic_results['sensitivity']:.3f}\n")
        f.write(f"Classification: {genomic_results['responder_type']}\n")
        f.write(f"Confidence: {genomic_results['confidence']:.3f}\n")
        f.write(f"Strategy: {genomic_results['strategy']}\n\n")
        
        f.write("TREATMENT SIMULATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Reward: {metrics['total_reward']:.2f}\n")
        f.write(f"Simulation Steps: {metrics['steps']}\n\n")
        
        f.write("CLINICAL RECOMMENDATION\n")
        f.write("-"*70 + "\n")
        
        if genomic_results['sensitivity'] > 0.7:
            f.write("RECOMMENDATION: Proceed with nanoparticle therapy\n")
            f.write("EXPECTED RESPONSE: 70-85%\n")
            f.write("MONITORING: Standard protocol (every 9 weeks)\n")
        elif genomic_results['sensitivity'] > 0.5:
            f.write("RECOMMENDATION: Nanoparticle therapy with enhanced monitoring\n")
            f.write("EXPECTED RESPONSE: 50-70%\n")
            f.write("MONITORING: Frequent imaging (every 6 weeks)\n")
        else:
            f.write("RECOMMENDATION: Consider alternative therapies\n")
            f.write("EXPECTED RESPONSE: 30-50%\n")
            f.write("MONITORING: Close follow-up, early switch if progression\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        
    
    print(f"Report saved: {output_file}")


#main demo

def main():
    """Main demo function"""
    
    print_header("PERSONALIZED NANOPARTICLE DRUG DELIVERY SYSTEM")
    print("\nAI-Driven Precision Medicine Demo")
    print("Combining Genomics + Reinforcement Learning + Multi-Agent Systems")
    
    # Step 1: Load genomic model
    print_section("LOADING SYSTEM")
    
    print("\nLoading genomic model...")
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
        print("Trained model loaded (AUROC: 0.71)")
    except:
        print("Using untrained model")
    
    genomic_model.eval()
    
    # Step 2: Patient selection
    patient_data = select_patient()
    
    # Step 3: Genomic analysis
    genomic_results = analyze_genomics(patient_data, genomic_model)
    
    # Step 4: RL training
    print("\n")
    proceed = input("Proceed with RL training? (y/n): ").strip().lower()
    
    if proceed != 'y':
        print("\nDemo terminated by user.")
        return
    
    agent, env, episode_rewards = train_personalized_agent(
        genomic_model,
        patient_data['sequence'],
        n_episodes=30,
        n_agents=3
    )
    
    # Step 5: Treatment simulation
    trajectory, metrics = simulate_treatment(agent, env, n_agents=3)
    
    # Step 6: Visualization
    visualize_results(
        patient_data,
        genomic_results,
        episode_rewards,
        trajectory,
        metrics
    )
    
    # Step 7: Export
    export_results(patient_data, genomic_results, metrics)
    
    # Final summary
    print_header("DEMO COMPLETE")
    
    print("\nGenomic analysis completed")
    print("Personalized RL agent trained")
    print("Treatment simulation executed")
    print("Results visualized and exported")
    
    print("\nOutput files:")
    print("outputs/demo_results.png")
    print("outputs/demo_report.txt")
    
    


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()