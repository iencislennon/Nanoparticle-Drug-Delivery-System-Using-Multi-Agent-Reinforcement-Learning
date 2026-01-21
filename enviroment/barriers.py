import numpy as np
class Endothelial_Barrier:
    def __init__(self, is_tumor=False):
        self.is_tumor = is_tumor

        """
            arguments:
                is_tumor: if true, has EPR effect (leaky)
        """

        if is_tumor:
            self.pore_size = 400e-9 #nm
            self.permeability = 1e-7 #m/s 
            # self.permeability = 0.01 for demo phase 1
        else:
            self.pore_size = 5e-9 #nm
            self.permeability = 1e-10 #m/s
            # self.permeability = 0.001  for demo phase 1

        
    def allow_pass(self, particle_size):
        """
            arguments:
                particle_size: the size of particle measures in meters
            returns:
                true/false
        """
        # size must fit through pore
        if particle_size >= self.pore_size:
            return False
        # despite the size is OK , there are always random probability that it wouldn't go through
        random_chance = np.random.random()

        if random_chance < self.permeability:
            return True #Luck it passed
        else:
            return False


