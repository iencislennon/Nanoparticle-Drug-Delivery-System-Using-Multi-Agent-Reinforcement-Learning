"""
Comprehensive Metrics System

Computes metrics across all project components:
1. Drug Delivery Efficiency
2. RL Performance
3. Genomic Prediction Quality
4. Personalization Impact
5. System Integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error
)
import torch
import sys
sys.path.append('.')

from models.genomic_vit import GenomicViT
from rl.personalized_env import PersonalizedNanoparticleEnv
from rl.hierarchical_agent import Hierarchical_agent


class MetricsComputer:
    """
    Comprehensive metrics computation for all project components
    """
    
    def __init__(self):
        self.metrics = {}
        self.detailed_results = {}
    
     #1: DRUG DELIVERY EFFICIENCY 
    
    def compute_delivery_metrics(self, trajectory, env):
        """
        Compute physics-based delivery metrics
        
        arguments:
            trajectory: List of environment states during episode
            env: Environment instance
        
        returns:
            dict with delivery metrics
        """
        metrics = {}
        
        # 1. Target Hit Rate
        # % of agents that reached tumor core
        final_positions = trajectory[-1]['positions']
        tumor_center = env.tumor_center
        tumor_radius = env.tumor_radius
        
        distances = np.linalg.norm(final_positions - tumor_center, axis=1)
        agents_in_tumor = np.sum(distances < tumor_radius)
        
        metrics['target_hit_rate'] = agents_in_tumor / len(final_positions)
        
        # 2. Drug Concentration at Tumor
        # Total drug delivered to tumor vs total released
        tumor_drug = trajectory[-1].get('tumor_drug', 0)
        total_released = trajectory[-1].get('released_drug', 1)
        
        metrics['drug_concentration_ratio'] = tumor_drug / max(total_released, 1e-6)
        metrics['tumor_drug_amount'] = tumor_drug
        metrics['total_drug_released'] = total_released
        
        # 3. Spatial Coverage
        # What % of tumor volume was covered by agents
        if hasattr(env, 'tumor_grid'):
            covered_cells = np.sum(env.tumor_grid > 0)
            total_cells = env.tumor_grid.size
            metrics['spatial_coverage'] = covered_cells / total_cells
        else:
            metrics['spatial_coverage'] = None
        
        # 4. Delivery Time
        # Average time for agents to reach tumor
        delivery_times = []
        for t, state in enumerate(trajectory):
            positions = state['positions']
            distances = np.linalg.norm(positions - tumor_center, axis=1)
            
            for i, dist in enumerate(distances):
                if dist < tumor_radius and i not in [d[0] for d in delivery_times]:
                    delivery_times.append((i, t))
        
        if delivery_times:
            avg_delivery_time = np.mean([t for _, t in delivery_times])
            metrics['delivery_time'] = avg_delivery_time
        else:
            metrics['delivery_time'] = len(trajectory)  # Never reached
        
        # 5. Off-Target Ratio
        # Drug at healthy tissue vs tumor
        healthy_drug = trajectory[-1].get('healthy_drug', 0)
        metrics['off_target_ratio'] = healthy_drug / max(tumor_drug, 1e-6)
        
        return metrics
    
    #2: RL PERFORMANCE
    
    def compute_rl_metrics(self, episode_rewards, training_history=None):
        """
        Compute RL performance metrics
        
        Args:
            episode_rewards: List of rewards per episode
            training_history: Optional detailed training data
        
        Returns:
            dict with RL metrics
        """
        metrics = {}
        
        # 1. Episode Reward Statistics
        metrics['mean_reward'] = np.mean(episode_rewards)
        metrics['std_reward'] = np.std(episode_rewards)
        metrics['min_reward'] = np.min(episode_rewards)
        metrics['max_reward'] = np.max(episode_rewards)
        metrics['median_reward'] = np.median(episode_rewards)
        
        # 2. Learning Progress
        initial_performance = np.mean(episode_rewards[:10])
        final_performance = np.mean(episode_rewards[-10:])
        
        metrics['initial_performance'] = initial_performance
        metrics['final_performance'] = final_performance
        metrics['absolute_improvement'] = final_performance - initial_performance
        metrics['relative_improvement'] = (
            (final_performance - initial_performance) / 
            abs(initial_performance) * 100
        )
        
        # 3. Success Rate
        # Define success as reward > threshold (e.g., median)
        threshold = np.median(episode_rewards)
        success_count = np.sum(np.array(episode_rewards) > threshold)
        metrics['success_rate'] = success_count / len(episode_rewards)
        
        # 4. Sample Efficiency
        # How many episodes to reach 90% of final performance
        target = 0.9 * final_performance
        
        for i, reward in enumerate(episode_rewards):
            if reward >= target:
                metrics['episodes_to_90pct'] = i + 1
                break
        else:
            metrics['episodes_to_90pct'] = len(episode_rewards)
        
        # 5. Stability
        # Variance in last 25% of training
        final_quarter = episode_rewards[int(0.75 * len(episode_rewards)):]
        metrics['final_stability'] = np.std(final_quarter)
        
        return metrics
    
    #3: GENOMIC PREDICTION QUALITY 
    
    def compute_genomic_metrics(self, predictions, targets, predictions_proba=None):
        """
        Compute genomic model metrics
        
        arguments:
            predictions: Binary predictions (0 or 1)
            targets: True binary labels (0 or 1)
            predictions_proba: Predicted probabilities (for AUROC)
        
        returns:
            dict with genomic metrics
        """
        metrics = {}
        
        # Ensure binary targets
        binary_targets = (targets > 0.5).astype(int)
        
        # 1. AUROC (primary metric)
        if predictions_proba is not None:
            try:
                metrics['auroc'] = roc_auc_score(binary_targets, predictions_proba)
            except:
                metrics['auroc'] = None
        else:
            metrics['auroc'] = None
        
        # 2. Accuracy
        metrics['accuracy'] = accuracy_score(binary_targets, predictions)
        
        # 3. Precision & Recall
        metrics['precision'] = precision_score(binary_targets, predictions, zero_division=0)
        metrics['recall'] = recall_score(binary_targets, predictions, zero_division=0)
        
        # 4. F1 Score
        metrics['f1_score'] = f1_score(binary_targets, predictions, zero_division=0)
        
        # 5. Confusion Matrix
        cm = confusion_matrix(binary_targets, predictions)
        metrics['confusion_matrix'] = cm
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            metrics['true_positives'] = tp
            
            # specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 6. Calibration Error (if probabilities available)
        if predictions_proba is not None:
            # Expected Calibration Error
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (predictions_proba >= bin_lower) & (predictions_proba < bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = binary_targets[in_bin].mean()
                    avg_confidence_in_bin = predictions_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            metrics['calibration_error'] = ece
        
        return metrics
    
    #4: PERSONALIZATION IMPACT
    
    def compute_personalization_metrics(self, personalized_results, baseline_results):
        """
        Compute personalization impact metrics
        
        arguments:
            personalized_results: dict with patient_id -> performance
            baseline_results: dict with patient_id -> baseline performance
        
        returns:
            dict with personalization metrics
        """
        metrics = {}
        
        # 1. Average Improvement
        improvements = []
        for patient_id in personalized_results.keys():
            if patient_id in baseline_results:
                pers = personalized_results[patient_id]
                base = baseline_results[patient_id]
                
                improvement = (pers - base) / abs(base) * 100
                improvements.append(improvement)
        
        if improvements:
            metrics['mean_improvement'] = np.mean(improvements)
            metrics['std_improvement'] = np.std(improvements)
            metrics['min_improvement'] = np.min(improvements)
            metrics['max_improvement'] = np.max(improvements)
            metrics['median_improvement'] = np.median(improvements)
        
        # 2. Patients Improved
        patients_improved = sum(1 for imp in improvements if imp > 0)
        metrics['patients_improved_count'] = patients_improved
        metrics['patients_improved_pct'] = patients_improved / len(improvements) * 100
        
        # 3. Stratification by Responder Type
        # (Requires responder_type info in results)
        # Placeholder for now
        metrics['high_responder_improvement'] = None
        metrics['medium_responder_improvement'] = None
        metrics['poor_responder_improvement'] = None
        
        return metrics
    
    #5: SYSTEM INTEGRATION 
    
    def compute_integration_metrics(self, end_to_end_results):
        """
        Compute system-level integration metrics
        
        arguments:
            end_to_end_results: Results from full pipeline
        
        returns:
            dict with integration metrics
        """
        metrics = {}
        
        # 1. End-to-End Success Rate
        # % of patients with successful outcome
        if 'outcomes' in end_to_end_results:
            outcomes = end_to_end_results['outcomes']
            metrics['e2e_success_rate'] = np.mean(outcomes)
        
        # 2. Pipeline Consistency
        # Correlation between genomic prediction and RL performance
        if 'genomic_predictions' in end_to_end_results and 'rl_performances' in end_to_end_results:
            gen_pred = end_to_end_results['genomic_predictions']
            rl_perf = end_to_end_results['rl_performances']
            
            metrics['prediction_performance_correlation'] = np.corrcoef(gen_pred, rl_perf)[0, 1]
        
        # 3. Robustness
        # Performance variance across patients
        if 'patient_performances' in end_to_end_results:
            perfs = end_to_end_results['patient_performances']
            metrics['performance_variance'] = np.var(perfs)
            metrics['performance_cv'] = np.std(perfs) / np.mean(perfs)  # Coefficient of variation
        
        return metrics
    
    
    def compute_all_metrics(
        self,
        delivery_data=None,
        rl_data=None,
        genomic_data=None,
        personalization_data=None,
        integration_data=None
    ):
        """
        Compute all metrics at once
        
        arguments:
            delivery_data: dict with trajectory, env
            rl_data: dict with episode_rewards
            genomic_data: dict with predictions, targets
            personalization_data: dict with personalized/baseline results
            integration_data: dict with e2e results
        
        returns:
            dict with all metrics
        """
        all_metrics = {}
        
        # Delivery metrics
        if delivery_data:
            print("Computing delivery metrics...")
            all_metrics['delivery'] = self.compute_delivery_metrics(
                delivery_data['trajectory'],
                delivery_data['env']
            )
        
        # RL metrics
        if rl_data:
            print("Computing RL metrics...")
            all_metrics['rl'] = self.compute_rl_metrics(
                rl_data['episode_rewards']
            )
        
        # Genomic metrics
        if genomic_data:
            print("Computing genomic metrics...")
            all_metrics['genomic'] = self.compute_genomic_metrics(
                genomic_data['predictions'],
                genomic_data['targets'],
                genomic_data.get('predictions_proba')
            )
        
        # Personalization metrics
        if personalization_data:
            print("Computing personalization metrics...")
            all_metrics['personalization'] = self.compute_personalization_metrics(
                personalization_data['personalized_results'],
                personalization_data['baseline_results']
            )
        
        # Integration metrics
        if integration_data:
            print("Computing integration metrics...")
            all_metrics['integration'] = self.compute_integration_metrics(
                integration_data
            )
        
        self.metrics = all_metrics
        return all_metrics
    
    #report
    
    def generate_report(self, output_file='outputs/metrics_report.txt'):
        """
            generate comprehensive metrics report
        """
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("COMPREHENSIVE METRICS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Delivery metrics
            if 'delivery' in self.metrics:
                f.write("DRUG DELIVERY EFFICIENCY\n")
                f.write("-" * 70 + "\n")
                for key, value in self.metrics['delivery'].items():
                    if value is not None:
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            # RL metrics
            if 'rl' in self.metrics:
                f.write("RL PERFORMANCE\n")
                f.write("-" * 70 + "\n")
                for key, value in self.metrics['rl'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            # Genomic metrics
            if 'genomic' in self.metrics:
                f.write("GENOMIC PREDICTION QUALITY\n")
                f.write("-" * 70 + "\n")
                for key, value in self.metrics['genomic'].items():
                    if key != 'confusion_matrix' and value is not None:
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            # Personalization metrics
            if 'personalization' in self.metrics:
                f.write("PERSONALIZATION IMPACT\n")
                f.write("-" * 70 + "\n")
                for key, value in self.metrics['personalization'].items():
                    if value is not None:
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            # Integration metrics
            if 'integration' in self.metrics:
                f.write("SYSTEM INTEGRATION\n")
                f.write("-" * 70 + "\n")
                for key, value in self.metrics['integration'].items():
                    if value is not None:
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")
        
        print(f"✓ Report saved to: {output_file}")
    
    def save_to_csv(self, output_file='outputs/metrics_detailed.csv'):
        """
            save all metrics to CSV for further analysis
        """
        rows = []
        for category, metrics_dict in self.metrics.items():
            for metric_name, value in metrics_dict.items():
                if isinstance(value, (int, float, np.number)):
                    rows.append({
                        'category': category,
                        'metric': metric_name,
                        'value': float(value)
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"✓ Detailed metrics saved to: {output_file}")
    
    def visualize_metrics(self, output_file='outputs/metrics_visualization.png'):
        """
        Create visualization of key metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: RL Learning Curve (if available)
        if 'rl' in self.metrics:
            ax = axes[0, 0]
            # Placeholder - would need episode data
            ax.set_title('RL Learning Progress')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
        
        # Plot 2: Genomic Metrics
        if 'genomic' in self.metrics:
            ax = axes[0, 1]
            metrics = self.metrics['genomic']
            
            metric_names = ['auroc', 'accuracy', 'precision', 'recall', 'f1_score']
            values = [metrics.get(m, 0) for m in metric_names]
            
            ax.bar(range(len(metric_names)), values)
            ax.set_xticks(range(len(metric_names)))
            ax.set_xticklabels(metric_names, rotation=45)
            ax.set_ylabel('Score')
            ax.set_title('Genomic Prediction Metrics')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Delivery Efficiency
        if 'delivery' in self.metrics:
            ax = axes[1, 0]
            metrics = self.metrics['delivery']
            
            metric_names = ['target_hit_rate', 'drug_concentration_ratio', 'spatial_coverage']
            values = [metrics.get(m, 0) for m in metric_names if metrics.get(m) is not None]
            names = [m for m in metric_names if metrics.get(m) is not None]
            
            if values:
                ax.bar(range(len(names)), values)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')
                ax.set_ylabel('Score')
                ax.set_title('Drug Delivery Efficiency')
                ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "KEY METRICS SUMMARY\n\n"
        
        if 'rl' in self.metrics:
            summary_text += f"RL Improvement: {self.metrics['rl'].get('relative_improvement', 0):.1f}%\n"
        
        if 'genomic' in self.metrics:
            summary_text += f"Genomic AUROC: {self.metrics['genomic'].get('auroc', 0):.3f}\n"
        
        if 'personalization' in self.metrics:
            summary_text += f"Personalization: {self.metrics['personalization'].get('mean_improvement', 0):.1f}%\n"
        
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_file}")
        plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("METRICS COMPUTATION EXAMPLE")
    print("=" * 70)
    
    # Create metrics computer
    computer = MetricsComputer()
    
    # Example: Load some results
    print("\nLoading personalized RL results...")
    try:
        rl_results = pd.read_csv('outputs/personalized_rl_results.csv')
        
        # Prepare RL data
        episode_rewards = []  # Would need actual episode data
        
        # For now, use final rewards as proxy
        rl_data = {
            'episode_rewards': rl_results['final_reward'].values.tolist()
        }
        
        print("Loaded RL results")
    except:
        print("No RL results found, using synthetic data")
        rl_data = {
            'episode_rewards': list(range(-500, -300, 4))  # Simulated learning
        }
    
    # Example: Genomic predictions
    print("\nLoading genomic predictions...")
    # Example: Genomic predictions
    print("\nLoading genomic predictions...")
    try:
        # Load trained model
        from models.genomic_vit import GenomicViT
        from data.genomic.dataset import GenomicDataset, genomic_collate_fn
        from torch.utils.data import DataLoader, random_split
        
        model = GenomicViT()
        model.load_state_dict(torch.load('data/genomic/processed/best_model.pth'))
        model.eval()
        
        # Load data
        features = pd.read_csv('data/genomic/processed/patient_features.csv')
        sequences = np.load('data/genomic/processed/sequences_onehot.npy')
        
        # Create dataset
        dataset = GenomicDataset(features, sequences)
        
        # Use test split
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        _, _, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=genomic_collate_fn)
        
        # Evaluate model (REAL predictions!)
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequence']
                labels = batch['label']
                
                drug_response, _ = model(sequences)
                
                all_preds.append(drug_response.cpu())
                all_targets.append(labels.cpu())
        
        predictions_proba = torch.cat(all_preds).squeeze().numpy()
        targets = torch.cat(all_targets).squeeze().numpy()
        
        # Binarize
        predictions = (predictions_proba > 0.5).astype(int)
        targets = (targets > 0.5).astype(int)
        
        genomic_data = {
            'predictions': predictions,
            'targets': targets,
            'predictions_proba': predictions_proba
        }
        
        print("Evaluated model on test set")
    except:
        print("No genomic data found")
        genomic_data = None
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("COMPUTING METRICS...")
    print("=" * 70)
    
    all_metrics = computer.compute_all_metrics(
        rl_data=rl_data,
        genomic_data=genomic_data
    )
    
    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORTS...")
    print("=" * 70)
    
    computer.generate_report()
    computer.save_to_csv()
    computer.visualize_metrics()
    
    print("\n" + "=" * 70)
    print("METRICS COMPUTATION COMPLETE!")
    print("=" * 70)
    
    # Print summary
    print("\nKEY FINDINGS:")
    if 'rl' in all_metrics:
        print(f"  RL Performance:")
        print(f"    - Mean reward: {all_metrics['rl']['mean_reward']:.2f}")
        print(f"    - Improvement: {all_metrics['rl']['relative_improvement']:.1f}%")
    
    if 'genomic' in all_metrics:
        print(f"  Genomic Prediction:")
        print(f"    - AUROC: {all_metrics['genomic'].get('auroc', 'N/A')}")
        print(f"    - Accuracy: {all_metrics['genomic']['accuracy']:.3f}")