module TriangularPlateaux

using LinearAlgebra
using Sunny
import Sunny: Vec3, SelfConsistentNLSWT, PerturbativeTheory, solve_self_consistent_nlswt!, calculate_real_space_cubic_vertices_dipole, calculate_real_space_quartic_vertices_dipole, excitations_scnlswt, excitations_nlsw, calculate_mean_field_values_lswt, dynamical_matrix!, swt_hamiltonian_dipole_nlsw!, bogoliubov!, norm2, update_mean_field_values!

include("Types.jl")
export UUDPlateau, UUUDPlateau

include("SWTs.jl")
export swts

using JLD2
include("SCF.jl")
export find_lb_ub_scf, find_J2_bound_scf, renormalized_single_magnon_energies_scf, magnetization_correction_scf, ground_state_energy_scf

include("OneLoopPrime.jl")
export magnetization_correction_one_loop_prime, ground_state_energy_one_loop_prime, calculate_mean_field_values_hc_one_loop_prime, renormalized_single_magnon_energies_one_loop_prime

include("OneLoop.jl")
export renormalized_single_magnon_energies_one_loop

import StaticArrays: SVector
import HCubature: hcubature

end
