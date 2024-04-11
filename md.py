from asap3 import *
from asap3.md.velocitydistribution import MaxwellBoltzmannDistribution
from asap3.analysis.rdf import RadialDistributionFunction
from ase.lattice.cubic import *
import matplotlib.pyplot as plt
from ase.io import read
from ase.md.langevin import Langevin
from ase.constraints import FixAtoms

# We want RDF out to a distance of 15 Angstrom, with 200 bins
rng = 10.0
bins = 200

# Load the structure
atoms = read("Al.xyz")

# Example indices of constrained atoms
fixed_indices = list(range(0, 720))

# Apply constraints
constraints = FixAtoms(indices=fixed_indices)
atoms.set_constraint(constraints)

# Select unconstrained atoms using boolean indexing
unconstrained_indices = [i for i in range(len(atoms)) if i not in fixed_indices]

atoms.calc = EMT()

# Set up temperature values to loop over
temperature_values = [300, 600, 900, 1200, 1500, 1800]

# Initialize lists to store RDF data for each temperature
rdf_data_list = []

# Loop over temperature values
for T in temperature_values:
    # Set temperature
    MaxwellBoltzmannDistribution(atoms, T * units.kB)

    # Set up Langevin dynamics
    dyn = Langevin(atoms, 5 * units.fs, T * units.kB, 0.002, logfile=f"log_T{T}.log", trajectory=f"traj_T{T}.traj")

    # Run the simulation
    dyn.run(200)

    # Make the RDF object, attach it to the dynamics
    RDFobj = RadialDistributionFunction(atoms[unconstrained_indices], rng, bins, verbose=True)

    # Attach the update_rdf function to the dynamics
    dyn.attach(RDFobj.update, interval=5)

    # Run the simulation
    dyn.run(2500)

    # Get the RDF data and store it in the list
    rdf = RDFobj.get_rdf()
    rdf_data_list.append((T, rdf))

    # Plot separate RDF graphs and save data to file
    plt.figure()
    x = np.arange(bins) * rng / bins
    plt.plot(x, rdf, label=f"T = {T} K")
    plt.xlabel("Distance (Å)")
    plt.ylabel("RDF")
    plt.legend()
    plt.savefig(f"rdf_{T}.png")
    plt.close()

    # Save RDF data to file
    np.savetxt(f"rdf_data_{T}.txt", np.column_stack((x, rdf)), header="Distance (Å)   RDF")

# Plot overlayed RDF data
plt.figure()
for T, rdf in rdf_data_list:
    x = np.arange(bins) * rng / bins
    plt.plot(x, rdf, label=f"T = {T} K")

# Plot labels and legend
plt.xlabel("Distance (Å)")
plt.ylabel("RDF")
plt.legend()
plt.savefig("rdf_overlay.png")
plt.show()
