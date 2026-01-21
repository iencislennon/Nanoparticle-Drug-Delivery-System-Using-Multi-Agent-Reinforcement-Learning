from dataclasses import dataclass
import numpy as np
@dataclass
class PheromoneMessage:
    """
        chemical signal from agent
    """
    sender_id:int 
    position: np.ndarray #where nanopartical stayed
    strength: float # force of signal
    timestamp: int # when
    message_type: str #something like 'tumor_found'

class CommunicationProtocol:
    """
        manipulate phoromone signals
    """
    
    def __init__(self, comm_range=500e-6):
        self.comm_range = comm_range
        self.messages = [] # list of all messages
        self.decay_rate = 0.01 #how fast fades away

    def broadcast(self, sender_id, position, message_type, strength=1.0,time=0):
        """
            to send a message
        """
        msg = PheromoneMessage(
            sender_id=sender_id,
            position=position.copy(),
            strength=strength,
            timestamp=time,
            message_type=message_type
        )

        self.messages.append(msg)

    def receive(self, position, time):
        """
            returns:
                list of messages within comm_range
        """
        received = []
        for msg in self.messages:
            
            distance = np.linalg.norm(msg.position - position)

            if distance <= self.comm_range:
                age = time-msg.timestamp
                current_strength =  msg.strength * np.exp(-self.decay_rate * age)
                
                if current_strength > 0.01: #threshold
                    received.append(msg)
        return received
    
    def get_gradient(self, position, message_type, time):
        messages = self.receive(position, time)  
    
        messages = [m for m in messages if m.message_type == message_type]
        if len(messages) == 0:
            return np.zeros(3)
        
        gradient = np.zeros(3)
        total_weight = 0.0
        for msg in messages:
            direction = msg.position - position

            age = time - msg.timestamp
            current_strength = msg.strength * np.exp(-self.decay_rate * age)
            w = current_strength
            distance = np.linalg.norm(direction)

            if distance > 0:
                weight = current_strength/ distance+1e-6

                gradient += weight * direction
                total_weight += weight        
        if total_weight > 0:
            gradient = gradient / total_weight
        
        return gradient

