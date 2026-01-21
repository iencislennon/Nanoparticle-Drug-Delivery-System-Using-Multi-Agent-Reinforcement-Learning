import numpy as np
class Tumor_model:
    # 3D tumor model with 3 concrentic circles: necrotic, hypoxic, proliferative
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
        """
            arguments:
                center: (x,y,z) coordinates measures in meters
                radius: radius of tumor measures in meters
        """
        self.r_necrotic = radius * 0.5
        self.r_hypoxic = radius * 0.8
        # proliferative = radius, so we don't need to find it 

    def get_zone(self, position):
        """
            arguments:
                position (x,y,z) coordinates measures in meters
            reterns whether ceil is necrotic, hypoxic, proliferative or normal
        """    
        position = np.array(position)
        # distance from center to position 
        distance = np.linalg.norm(self.center - position)

        # define zone based on a distance
        if distance > self.radius:
            return 'normal'
        elif distance < self.r_necrotic:
            return 'necrotic'
        elif distance < self.r_hypoxic:
            return 'hypoxic'
        else:
            return 'proliferative'
        
    def properties(self, position):
        """
            get properties of tumor at this point
        """
        zone = self.get_zone(position)

        # return dict with parametrs for this zone
        
        if zone == 'necrotic':
            return  {
                'Oxygen (mmHg)': 0,
                'pH': 6.2,
                'Cell density': 'low',
                'is_tumor': True
            }
        elif zone == 'hypoxic':
            return  {
                'Oxygen (mmHg)': 5,
                'pH': 6.8,
                'Cell density': 'medium',
                'is_tumor': True
            }
        elif zone == 'proliferative':
            return  {
                'Oxygen (mmHg)': 40,
                'pH': 7.4,
                'Cell density': 'high',
                'is_tumor': True
            }        
        else:
            return  {
                'Oxygen (mmHg)': 60,
                'pH': 7.4,
                'Cell density': 'normal',
                'is_tumor': False
            }
    def oxygen_gradient(self, position):
        """
            returns:
                (gx, gy. gz) gradient vector
        """
        direction = position - self.center
        distance = np.linalg.norm(direction)
        if distance < 1e-10:
            return np.zeros(3)
        gradient = direction / distance
        return gradient


