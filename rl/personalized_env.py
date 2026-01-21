"""
Personalized Nanoparticle Environment

Integrates genomic predictions into rl environment to create
patient-specific drug delivery strategies.
"""

import numpy as np
import torch
from rl.environment import NanoparticleEnv

class PersonalizedNanoparticleEnv(NanoparticleEnv):
    """
    Genomic-aware RL environment
    
    Modulates rewards based on patient's predicted drug response
    from genomic model.
    """
    
    def __init__(
        self,
        genomic_model,
        patient_sequence,
        n_agents=5
    ):
        """
        Args:
            genomic_model: Trained GenomicViT model
            patient_sequence: One-hot encoded DNA (1024, 4)
            n_agents: Number of nanoparticle agents
        """
        super().__init__(n_agents=n_agents)
        
        self.genomic_model = genomic_model
        self.patient_sequence = patient_sequence
        
        self.drug_sensitivity = self._predict_sensitivity()
        
        print(f"Patient drug sensitivity: {self.drug_sensitivity:.3f}")
    
    def _predict_sensitivity(self):
        """
        Predict patient's drug response from genomic sequence
        
        returns:
            float: Drug sensitivity (0-1)
        """
        self.genomic_model.eval()
        
        with torch.no_grad():
            sequence = torch.tensor(
                self.patient_sequence,
                dtype=torch.float32
            ).unsqueeze(0)
            
            drug_response, uncertainty = self.genomic_model(sequence)
            
            sensitivity = drug_response.item()
        
        return sensitivity
    
    def step(self, actions):
        """
        Execute actions with genomic personalization
        
        arguments:
            actions: List of actions for each agent
        
        returns:
            next_obs: Next observations
            personalized_rewards: Rewards modulated by genomic sensitivity
            done: Episode termination flag
            info: Additional information
        """
        # Base environment step
        next_obs, base_rewards, done, info = super().step(actions)
        
        # Personalize rewards based on drug sensitivity
        # High sensitivity → higher rewards for successful delivery
        # Low sensitivity → lower rewards (harder to treat)
        personalized_rewards = self._personalize_rewards(base_rewards)
        
        # Add genomic info
        info['drug_sensitivity'] = self.drug_sensitivity
        info['base_rewards'] = base_rewards
        info['personalized_rewards'] = personalized_rewards
        
        return next_obs, personalized_rewards, done, info
    
    def _personalize_rewards(self, base_rewards):
        """
        Modulate rewards by drug sensitivity
        
        arguments:
            base_rewards: original rewards from environment
        
        returns:
            personalized rewards
        """
        # Strategy 1: Linear scaling
        # High responders (0.8-1.0) → 1.5x rewards
        # Medium responders (0.5-0.8) → 1.0x rewards
        # Poor responders (0-0.5) → 0.5x rewards
        
        if self.drug_sensitivity > 0.7:
            # High responder → easier to treat
            scaling_factor = 1.0 + (self.drug_sensitivity - 0.7) * 2
        elif self.drug_sensitivity > 0.5:
            # Medium responder → normal difficulty
            scaling_factor = 1.0
        else:
            # Poor responder → harder to treat
            scaling_factor = 0.5 + self.drug_sensitivity
        
        personalized_rewards = base_rewards * scaling_factor
        
        return personalized_rewards
    
    def get_patient_info(self):
        """
        get patient genomic information
        
        returns:
            dict with patient info
        """
        return {
            'drug_sensitivity': self.drug_sensitivity,
            'responder_type': self._classify_responder(),
            'recommended_strategy': self._recommend_strategy()
        }
    
    def _classify_responder(self):
        """
        classify patient as good/medium/poor responder
        """
        if self.drug_sensitivity > 0.7:
            return "High Responder"
        elif self.drug_sensitivity > 0.5:
            return "Medium Responder"
        else:
            return "Poor Responder"
    
    def _recommend_strategy(self):
        """
        recommend treatment strategy based on genomic profile
        """
        if self.drug_sensitivity > 0.7:
            return "Standard dose, aggressive delivery"
        elif self.drug_sensitivity > 0.5:
            return "Standard dose, targeted delivery"
        else:
            return "High dose, multiple delivery attempts"


