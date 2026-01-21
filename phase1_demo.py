import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import numpy as np
# import classes for the 1st phase: biological parametr's, brownian motion, tumor model, endothelial barrier
from config.biological_params import BiologicalParams
from physics.diffusion import Brownian_Motion
from enviroment.tumor import Tumor_model
from enviroment.barriers import Endothelial_Barrier

def diffusion_demo():
    print("*"*50+"Diffusion demo"+"*"*50)

    bm_obj = Brownian_Motion(
        particle_radius=50e-9,
        viscosity=0.0035
    ) # simulate the trjectory 
    steps = 1000
    dt = 0.1
    positions = bm_obj.trajectory_simulation(steps=steps, dt=dt)
    
    #calculate displacement

    displacement = np.linalg.norm(positions[-1] - positions[0])

    # graph plotting

    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111, projection='3d')

    #coordinates

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    ax.plot(x,y,z, 'b-', linewidth = 0.5, alpha = 0.7, label = 'Trajectory')

    #mark start and end

    ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label = 'start')
    ax.scatter(x[-1], y[-1], z[-1], c='orange', s=100, marker='*', label = 'end')

    ax.set_xlabel('x (µm)')
    ax.set_ylabel(' y (µm)') #µm
    ax.set_zlabel('z (µm)')
    ax.set_title('Brownian motion trajectory 100 nm nanoparticle in blood')
    ax.legend()

    #set limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e6:.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e6:.0f}'))
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e6:.0f}'))


    plt.savefig('outputs/phase1_brownian_motion.png', dpi=150, bbox_inches='tight')
    print("\n Saved: outputs/phase1_brownian_motion.png")
    
    plt.show()

def tumor_demo():
    print("*"*50+"tumor demo"+"*"*50)

    tumor_obj = Tumor_model(
        center = [0,0,0],
        radius= 5e-3
    )
    necrotic = tumor_obj.get_zone([0,0,0])
    hypoxic = tumor_obj.get_zone([3e-3,0,0])
    proliferative = tumor_obj.get_zone([4.5e-3,0,0])
    normal = tumor_obj.get_zone([10e-3,0,0])

    #visualization
    test_points = {
        'Center (0,0,0)': ([0, 0, 0], necrotic),
        'Hypoxic (3mm,0,0)': ([3e-3, 0, 0], hypoxic),
        'Proliferative (4.5mm,0,0)': ([4.5e-3, 0, 0], proliferative),
        'Outside (10mm,0,0)': ([10e-3, 0, 0], normal)       
    } 
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # sphere creating 
    def create_sphere(center, radius, num_points=30):
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, np.pi, num_points)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z
    # Necrotic zone (red)
    x, y, z = create_sphere([0, 0, 0], tumor_obj.r_necrotic)
    ax.plot_surface(x, y, z, color='red', alpha=0.3, label='Necrotic')
    
    # Hypoxic zone (yellow)
    x, y, z = create_sphere([0, 0, 0], tumor_obj.r_hypoxic)
    ax.plot_surface(x, y, z, color='yellow', alpha=0.2, label='Hypoxic')
    
    # Proliferative zone (green)
    x, y, z = create_sphere([0, 0, 0], tumor_obj.radius)
    ax.plot_surface(x, y, z, color='green', alpha=0.2, label='Proliferative')
    
    # testing points
    point_colors = {
        'necrotic': 'darkred',
        'hypoxic': 'orange',
        'proliferative': 'darkgreen',
        'normal': 'blue'
    }

    for name, (point, zone) in test_points.items():
        ax.scatter(point[0], point[1], point[2], 
                  c=point_colors[zone], s=100, marker='o',
                  edgecolors='black', linewidths=2)
    
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title('3D Tumor Model with Heterogeneous Zones', fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e3:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e3:.1f}'))
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1e3:.1f}'))

    #legends
    legend_elements = [
        Patch(facecolor='red', alpha=0.3, label='Necrotic (0-50%)'),
        Patch(facecolor='yellow', alpha=0.3, label='Hypoxic (50-80%)'),
        Patch(facecolor='green', alpha=0.3, label='Proliferative (80-100%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    max_range = tumor_obj.radius * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.savefig('outputs/phase1_tumor_zones.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: outputs/phase1_tumor_zones.png")
    
    plt.show()   

def barriers_demo():
    normal = Endothelial_Barrier(is_tumor=False)
    tumor = Endothelial_Barrier(is_tumor=True)

    partcile_sizes = [3e-9, 50e-9, 100e-9, 500e-9]
    n_tests = 1000000
    results = {
            'sizes': [],
            'normal': [],
            'tumor': []
        }
    for i in partcile_sizes:
        #normal vessel's testing
        normal_passes = 0
        for j in range(n_tests):
            if normal.allow_pass(i):
                normal_passes += 1
        tumor_passes = 0
        for k in range(n_tests):
            if tumor.allow_pass(i):
                tumor_passes += 1
    
        #percent calculation
        percent_of_normal = (normal_passes/n_tests) * 100
        percent_of_tumor = (tumor_passes/n_tests) * 100

        

        results['sizes'].append(i*1e9)
        results['normal'].append(percent_of_normal)
        results['tumor'].append(percent_of_tumor)   

    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results['sizes']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, results['normal'], width, label='Normal vessel', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, results['tumor'], width, label='Tumor vessel (EPR)', color='red', alpha=0.7)
    
    ax.set_xlabel('Particle Size (nm)', fontsize=12)
    ax.set_ylabel('Passage Rate (%)', fontsize=12)
    ax.set_title('Endothelial Barrier Permeability Test\n(1000 trials per size)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s:.0f}' for s in results['sizes']])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            # if height > 0.001:  
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}%',
                    ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    plt.savefig('outputs/phase1_barrier_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: outputs/phase1_barrier_test.png")
    
    plt.show()
        
if __name__ == "__main__":
    diffusion_demo()
    tumor_demo()
    barriers_demo()