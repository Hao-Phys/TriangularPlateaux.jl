function renormalized_single_magnon_energies_one_loop_prime(state::AbstractTriangularPlateau, h, q::Vec3; hcubature_opts::NamedTuple=NamedTuple(;))
    (; nbands) = state
    swt, swt_ref = swts(state, h)

    real_space_cubic_vertices = calculate_real_space_cubic_vertices_dipole(swt.sys)
    real_space_quartic_vertices = calculate_real_space_quartic_vertices_dipole(swt.sys)

    ptt_ref = PerturbativeTheory(swt_ref; hcubature_opts...)

    ptt = PerturbativeTheory(swt, real_space_quartic_vertices, real_space_cubic_vertices, ptt_ref.mean_field_values)

    try
        E, _ = excitations_nlsw(ptt, q)
        E_renormalized = E[1:nbands]
        return E_renormalized
    catch _
        @warn "Unstable state encountered during renormalized energy calculation"
        return fill(NaN, nbands)
    end
end

function calculate_mean_field_values_hc_one_loop_prime(state::AbstractTriangularPlateau, result_path::String; hcubature_opts::NamedTuple=NamedTuple(;), save_mean_field_values::Bool=true)
    (; J₁, J₂, mode) = state

    if typeof(state) == UUDPlateau
        @assert J₂ ≤ 1/8 * J₁ "UUD Plateau state unstable for J₂/J₁ > 1/8"
    elseif typeof(state) == UUUDPlateau
        @assert J₂ ≥ 1/8 * J₁ "UUUD Plateau state unstable for J₂/J₁ < 1/8"
    else
        error("Unsupported state type: $(typeof(state))")
    end
    
    isdir(result_path) || mkpath(result_path)

    hc = calculate_hc(state)
    _, swt_ref = swts(state, hc)

    mean_field_values = calculate_mean_field_values_lswt(swt_ref; hcubature_opts...)

    if save_mean_field_values
        filename = joinpath(result_path, "mean_field_values_hc_$(typeof(state))$(mode)_J1_$(round(state.J₁, digits=4))_J2_$(round(state.J₂, digits=4)).jld2")
        jldsave(filename; mean_field_values=mean_field_values, hc=hc)
    end

    return mean_field_values
end

function magnetization_correction_one_loop_prime(state::AbstractTriangularPlateau, h; kwargs...)
    swt, swt_ref = swts(state, h)

    real_space_cubic_vertices = calculate_real_space_cubic_vertices_dipole(swt.sys)
    real_space_quartic_vertices = calculate_real_space_quartic_vertices_dipole(swt.sys)

    ptt_ref = PerturbativeTheory(swt_ref; kwargs...)
    ptt = PerturbativeTheory(swt, real_space_quartic_vertices, real_space_cubic_vertices, ptt_ref.mean_field_values)

    L = Sunny.nbands(swt)
    H1 = zeros(ComplexF64, 2L, 2L)
    H2 = zeros(ComplexF64, 2L, 2L)
    V = zeros(ComplexF64, 2L, 2L)
    δS = hcubature((0,0,0),(1,1,1); kwargs...) do q
        dynamical_matrix!(H1, swt, q)
        swt_hamiltonian_dipole_nlsw!(H2, ptt, q)
        @. H1 += H2
        bogoliubov!(V, H1)
        return SVector{L}(-norm2(view(V, L+i, 1:L)) for i in 1:L)
    end

    return δS[1]
end

function ground_state_energy_one_loop_prime(state::AbstractTriangularPlateau, h; kwargs...)
    swt, _ = swts(state, h)
    (; sys) = swt
    E_gs = 0.0
    # Classical energy
    E_cl = classical_energy(state, h)
    println("Classical energy per site: $E_cl")
    E_gs += E_cl

    swt, swt_ref = swts(state, h)

    ptt_ref = PerturbativeTheory(swt_ref; kwargs...)
    real_space_cubic_vertices = calculate_real_space_cubic_vertices_dipole(swt.sys)
    real_space_quartic_vertices = calculate_real_space_quartic_vertices_dipole(swt.sys)

    ptt = PerturbativeTheory(swt, real_space_quartic_vertices, real_space_cubic_vertices, ptt_ref.mean_field_values)

    L = Sunny.nbands(swt)
    H1 = zeros(ComplexF64, 2L, 2L)
    H2 = zeros(ComplexF64, 2L, 2L)
    V = zeros(ComplexF64, 2L, 2L)

    # The uniform correction (trace of the (1,1)-block of the dynamical matrix)
    dynamical_matrix!(H1, swt, zero(Vec3))
    swt_hamiltonian_dipole_nlsw!(H2, ptt, zero(Vec3))
    @. H1 += H2
    δE₁ = -real(tr(view(H1, 1:L, 1:L))) / (2Natoms)
    E_gs += δE₁

    # Integration over the Brillouin zone
    δE₂ = hcubature((0,0,0), (1,1,1); kwargs...) do q
        dynamical_matrix!(H1, swt, q)
        swt_hamiltonian_dipole_nlsw!(H2, ptt, q)
        @. H1 += H2
        ωs = bogoliubov!(V, H1)
        return sum(view(ωs, 1:L)) / (2Natoms)
    end
    # Discard the error bar in the integration
    E_gs += δE₂[1]

    println("Zero-point energy corrections: δE₁=$δE₁, δE₂=$(δE₂[1])")
    println("Energy per site up to quartic order before quartic correction: $E_gs")

    # Correction from the quartic terms
    δE₄ = 0.0
    index = 0
    (; real_space_quartic_vertices) = ptt
    for (i, int) in enumerate(sys.interactions_union)
        # Single-ion anisotropy
        (; c2, c4, c6) = swt.data.stevens_coefs[i]
        @assert iszero(c2) "Rank 2 Stevens operators not supported in :dipole non-perturbative calculations yet"
        @assert iszero(c4) "Rank 4 Stevens operators not supported in :dipole non-perturbative calculations yet"
        @assert iszero(c6) "Rank 6 Stevens operators not supported in :dipole non-perturbative calculations yet"
        for coupling in int.pair
            (; isculled) = coupling
            isculled && break
            index += 1

            Nii, Njj, Nij, Δii, Δjj, Δij = mean_field_values[6*(index-1)+1:6*(index-1)+6]
            (; V41, V42, V43) = real_space_quartic_vertices[index]
            if !iszero(coupling.bilin)
                Q = V41 * conj(Nij) + V42 * Δii + conj(V42) * conj(Δjj) + 2 * conj(V43) * (Nii + Njj)
                δE₄ += 2 * real(Q * Nij)

                Qi = V41 * Njj + 2 * V42 * Δij + 2 * conj(V42) * conj(Δij) + 2 * V43 * conj(Nij) + 2 * conj(V43) * Nij
                δE₄ += real(Qi * Nii)

                Qj = V41 * Nii + 2 * V42 * Δij + 2 * conj(V42) * conj(Δij) + 2 * V43 * conj(Nij) + 2 * conj(V43) * Nij
                δE₄ += real(Qj * Njj)

                P = V41 * conj(Δij) + 2 * V42 * (Nii + Njj) + V43 * conj(Δjj) + conj(V43) * conj(Δii)
                δE₄ += 2 * real(P * Δij)

                Pi = V42 * Nij + V43 * conj(Δij)
                δE₄ += 2 * real(Pi * Δii)

                Pj = V42 * conj(Nij) + conj(V43) * conj(Δij)
                δE₄ += 2 * real(Pj * Δjj)
            end

            # Biquadratic exchange
            if !iszero(coupling.biquad)
                @error "Biquadratic exchange not supported in :dipole perturbative calculations yet"
            end
        end
    end

    E_gs += δE₄
    println("Quartic energy correction: δE₄=$δE₄")
    println("Total ground state energy per site up to quartic order: $E_gs")

    return (E_gs, E_cl, δE₁, δE₂[1], δE₄)
end