using TriangularPlateaux

hcubature_opts = (; atol=1e-4)
nlsolve_opts = (; ftol=1e-4, iterations=100, show_trace=true)

# One-loop
J2 = 0.1
uud_state = UUDPlateau(1.0, J2, :one_loop)
E_ol = renormalized_single_magnon_energies_one_loop(uud_state, 1.5, TriangularPlateaux.Vec3(1/3, 1/3, 0); hcubature_opts)
E_ol_prime = renormalized_single_magnon_energies_one_loop_prime(uud_state, 1.5, TriangularPlateaux.Vec3(1/3, 1/3, 0); hcubature_opts)