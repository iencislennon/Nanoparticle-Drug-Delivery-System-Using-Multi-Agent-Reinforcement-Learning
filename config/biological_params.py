# Biological parameters for nanoparticle delivery simulation , values were picked from literature(you can find all of them in notes.md)

class BiologicalParams: 
    vessel_diameter = 8e-6 # 8 µm 
    blood_flow_velocity =  0.5e-3 #0.5 mm/sec
    blood_viscosity = 3.5 * 1e-3 #3.5 mPa·s
    nanoparticle_size = 100e-9 #100 nm
    tumor_ifp = 30 * 133.322 #30 mmHg 1mmhg ~= 133.322

def show_summary():
    print(f"vessel diameter: {BiologicalParams.vessel_diameter} µm")
    print(f"blood flow_velocity: {BiologicalParams.blood_flow_velocity} mm/sec")
    print(f"blood viscosity: {BiologicalParams.blood_viscosity} mPa·s")
    print(f"nanoparticle size: {BiologicalParams.nanoparticle_size} 100 nm")
    print(f"tumor ifp: {BiologicalParams.tumor_ifp} mmHg")

if __name__ == "__main__":
    show_summary()


        