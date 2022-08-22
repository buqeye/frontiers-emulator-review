from numpy import sqrt, pi

mass_proton = 938.2872046  # MeV
mass_neutron = 939.565379  # MeV
MN = mass_nucleon = 2 * (mass_neutron * mass_proton) / (mass_neutron + mass_proton)
mass_electron = 0.510998928
mass_alpha = 3727.379378  # MeV
mass_pion = 140.0

hbar_c = 197.3269718
alpha = 7.297352570930644e-3  # approx 1/137
fm_to_sqrt_mb = sqrt(10)  # 1 fm**2 == 10 mb
coulomb_constant = 14.3996 * 1e-6 * 1e5  # Convert ev.Angstrom.e^-2 ---> MeV.fm.e^-2
