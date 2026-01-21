import numpy as np
class Brownian_Motion:
    def __init__(self, particle_radius:float, viscosity:float, temperature:float = 310): # T =310 because we must to take temperature in a human's body as 37C in krlcin it is 370
        self.particle_radius = particle_radius
        self.viscosity = viscosity
        self.temperature = temperature
        # diffusion coefficient
        self.k = 1.38e-23 
        
        # first off we need to calculate a mass of nanopartial
        # m = ρV, the material of nanopartial is Fe3O4 
        density_of_iron_oxide = 5100 # m/m3
        volume = 4/3 * np.pi * np.power(self.particle_radius, 3) # nanopartial supposed to be a sphere
        self.mass = density_of_iron_oxide*volume
        # drag coeffient γ  = 6πηr
        self.drag_coeff = 6 * np.pi * viscosity * particle_radius
        # D = kT/ (6πηr) (diffusion coefficient) k=1.38e-23
        self.D = (self.k* temperature)/ (6 * np.pi * viscosity * particle_radius)

    def step(self, position, velocity, dt = 0.1):
        """
            argunments:
                position: current posirtion (x,y,z) measures in meters
                velocity: current velocity (vx, vy, vz) mesasures in m/s
                dt: time step measures in seconds
        """

        # Random thermal force (Gaussian noise)
        thermal_scale = np.sqrt(2 * self.D * dt)
        displacement = np.random.randn(3) * thermal_scale
        
        new_position = position + displacement
        
        new_velocity = displacement / dt
        return  new_position, new_velocity
        
    def trajectory_simulation(self, steps=1000, dt = 0.1):
        position = np.zeros(3) 
        velocity = np.zeros(3)
        step_results = []
        for i in range(steps):
            new_position, new_velocity = self.step(position, velocity, dt)
            step_results.append(new_position)
            # update the values
            position = new_position
            velocity = new_velocity
        return np.array(step_results)
        



            

            

