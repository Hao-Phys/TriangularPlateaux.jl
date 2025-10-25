module TriangularPlateaux

using Sunny
import Sunny: Vec3, SelfConsistentNLSWT, solve_self_consistent_nlswt!, PerturbationTheory, calculate_real_space_cubic_vertices_dipole, calculate_real_space_quartic_vertices_dipole, excitations_scnlswt, excitations_nlsw, calculate_mean_field_values_lswt

include("Types.jl")
export UUDPlateau, UUUDPlateau

include("SWTs.jl")
export swts

using JLD2
include("SCF.jl")
export find_lb_scf, find_ub_scf, renormalized_single_magnon_energies_scf

end
