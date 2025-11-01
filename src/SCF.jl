function renormalized_single_magnon_energies_scf(state::AbstractTriangularPlateau, mean_field_values::Vector{ComplexF64}, h, q::Vec3)
    (; nbands) = state
    swt, _ = swts(state, h)
    scnlswt = SelfConsistentNLSWT(swt)

    update_mean_field_values!(scnlswt, mean_field_values)
    try
        E, _ = excitations_scnlswt(scnlswt, q)
        E_renormalized = E[1:nbands]
        return E_renormalized
    catch _
        @warn "Unstable state encountered during renormalized energy calculation"
        return fill(NaN, nbands)
    end
end

function calculate_mean_field_values_hc_scf(state::AbstractTriangularPlateau, result_path::String; hcubature_opts::NamedTuple=NamedTuple(;), nlsolve_opts::NamedTuple=NamedTuple(;), save_mean_field_values::Bool=true)
    (; J₁, J₂) = state

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
    println("Calculating reference mean field values of swt_ref")
    scnlswt = SelfConsistentNLSWT(swt_ref)
    try
        solve_self_consistent_nlswt!(scnlswt; mean_field_values, hcubature_opts, nlsolve_opts)
        mean_field_values = copy(scnlswt.mean_field_values)
    catch _
        fill!(mean_field_values, NaN)
    end
    println("Reference mean field values calculated")

    if save_mean_field_values
        filename = joinpath(result_path, "mean_field_values_hc_$(typeof(state))_J1_$(round(state.J₁, digits=4))_J2_$(round(state.J₂, digits=4)).jld2")
        jldsave(filename; mean_field_values=mean_field_values, hc=hc)
    end

    return mean_field_values
end

function find_lb_ub_scf(state::AbstractTriangularPlateau, Δh, result_path::String; E_tol::Float64=1e-3, max_iter::Int=50, hcubature_opts::NamedTuple=NamedTuple(;), nlsolve_opts::NamedTuple=NamedTuple(;))
    (; J₂, nbands) = state
    isdir(result_path) || mkpath(result_path)

    iter = 1
    find_lb = false
    hc = calculate_hc(state)
    h_curr = hc - Δh
    h_lo = h_curr
    h_hi = hc

    mean_field_values = calculate_mean_field_values_hc_scf(state, result_path; hcubature_opts=hcubature_opts, nlsolve_opts=nlsolve_opts)

    if typeof(state) == UUDPlateau
        @assert J₂ ≤ 1/8 * state.J₁ "UUD Plateau state unstable for J₂/J₁ > 1/8. This code cannot be used to find lb/ub in that regime."
        q_ordering = Vec3(1/3, 1/3, 0)
    elseif typeof(state) == UUUDPlateau
        @assert J₂ ≥ 1/8 * state.J₁ "UUUD Plateau state unstable for J₂/J₁ < 1/8. This code cannot be used to find lb/ub in that regime."
        q_ordering = Vec3(1/2, 1/2, 0)
    else
        error("Unsupported state type: $(typeof(state))")
    end

    E = renormalized_single_magnon_energies_scf(state, mean_field_values, h_curr, q_ordering)[nbands]

    @warn !isnan(E) "Lowest magnon energy is $E at h=$h_curr, for J₂=$(round(J₂, digits=4)), consider using more aggressive guess for Δh"
    h_curr = (h_hi + h_lo) / 2

    println("Finding lower bound field h_lb")
    while !find_lb && iter ≤ max_iter
        println("--------------------------------")
        @show J₂, iter, h_curr
        E = renormalized_single_magnon_energies_scf(state, mean_field_values, h_curr, q_ordering)[nbands]
        println("Current lowest magnon energy E=$E")
        println("--------------------------------")
        if isnan(E)
            h_lo = h_curr
            h_curr = (h_lo + h_hi) / 2
        elseif E > E_tol
            h_hi = h_curr
            h_curr = (h_lo + h_hi) / 2
        else
            find_lb = true
        end
        iter += 1
    end
    h_lb, E_lb = h_curr, E

    iter = 1
    find_ub = false
    hc = calculate_hc(state)
    h_curr = hc + Δh
    h_lo = hc
    h_hi = h_curr

    E = renormalized_single_magnon_energies_scf(state, mean_field_values, h_curr, q_ordering)[nbands]
    @warn !isnan(E) "Lowest magnon energy is $E at h=$h_curr, for J₂=$(round(J₂, digits=4)), consider using more aggressive guess for Δh"
    h_curr = (h_hi + h_lo) / 2

    println("Finding upper bound field h_ub")
    while !find_ub && iter ≤ max_iter
        println("--------------------------------")
        @show J₂, iter, h_curr
        E = renormalized_single_magnon_energies_scf(state, mean_field_values, h_curr, q_ordering)[nbands]
        println("Current lowest magnon energy E=$E")
        println("--------------------------------")
        if isnan(E)
            h_hi = h_curr
            h_curr = (h_lo + h_hi) / 2
        elseif E > E_tol
            h_lo = h_curr
            h_curr = (h_lo + h_hi) / 2
        else
            find_ub = true
        end
        iter += 1
    end

    h_ub, E_ub = h_curr, E

    return h_lb, h_ub, E_lb, E_ub
end

function find_J2_bound_scf(state::UUDPlateau, h, ΔJ₂, result_path::String; E_tol::Float64=1e-3, max_iter::Int=50, hcubature_opts::NamedTuple=NamedTuple(;), nlsolve_opts::NamedTuple=NamedTuple(;))
    (; J₁, J₂, nbands) = state
    isdir(result_path) || mkpath(result_path)
    @assert J₂ == 1/8 * J₁ "Initial J₂ must be at the classical boundary value of 1/8 J₁"
    mean_field_values = calculate_mean_field_values_hc_scf(state, result_path; hcubature_opts=hcubature_opts, nlsolve_opts=nlsolve_opts, save_mean_field_values=false)

    iter = 1
    find_bound = false
    J2_lo = J₂
    J2_hi = J₂ + ΔJ₂
    J2_curr = J2_hi
    state_curr = UUDPlateau(J₁, J2_curr, :scf)

    # For UUD, J₂>1/8J₁, the minimum of magnon band shifts to M point
    q_ordering = Vec3(1/2, 1/2, 0)
    E = renormalized_single_magnon_energies_scf(state_curr, mean_field_values, h, q_ordering)[nbands]
    @warn !isnan(E) "Lowest magnon energy is $E at J2=$J2_curr, and h=$h, consider using more aggressive guess for ΔJ2"
    J2_curr = (J2_lo + J2_hi) / 2
    println("Finding J2 bound for UUD Plateau")
    while !find_bound && iter ≤ max_iter
        println("--------------------------------")
        @show J2_curr, iter
        state_curr = UUDPlateau(J₁, J2_curr, :scf)
        E = renormalized_single_magnon_energies_scf(state_curr, mean_field_values, h, q_ordering)[nbands]
        println("Current lowest magnon energy E=$E")
        println("--------------------------------")
        if isnan(E)
            J2_hi = J2_curr
            J2_curr = (J2_lo + J2_hi) / 2
        elseif E > E_tol
            J2_lo = J2_curr
            J2_curr = (J2_lo + J2_hi) / 2
        else
            find_bound = true
        end
        iter += 1
    end
    return J2_curr, E
end

function find_J2_bound_scf(state::UUUDPlateau, h, ΔJ2, result_path::String; E_tol::Float64=1e-3, max_iter::Int=50, hcubature_opts::NamedTuple=NamedTuple(;), nlsolve_opts::NamedTuple=NamedTuple(;))
    (; J₁, J₂, nbands) = state
    isdir(result_path) || mkpath(result_path)
    @assert J₂ == 1/8 * J₁ "Initial J₂ must be at the classical boundary value of 1/8 J₁"

    mean_field_values = calculate_mean_field_values_hc_scf(state, result_path; hcubature_opts=hcubature_opts, nlsolve_opts=nlsolve_opts, save_mean_field_values=false)

    iter = 1
    find_bound = false

    J2_hi = J₂
    J2_lo = J₂ - ΔJ2
    J2_curr = J2_lo
    state_curr = UUUDPlateau(J₁, J2_curr, :scf)

    # For the UUUD phase, J₂<1/8J₁, the minimum of magnon band can be at K or M point
    q_K = Vec3(1/3, 1/3, 0)
    q_M = Vec3(1/2, 1/2, 0)

    E_K = renormalized_single_magnon_energies_scf(state_curr, mean_field_values, h, q_K)[nbands]
    E_M = renormalized_single_magnon_energies_scf(state_curr, mean_field_values, h, q_M)[nbands]
    @warn !isnan(E_K) || !isnan(E_M) "Use more aggressive guess for ΔJ2, lowest magnon energies are E_K=$E_K, E_M=$E_M"

    if isnan(E_K) || isnan(E_M)
        E = NaN
    else
        E = min(E_K, E_M)
    end

    J2_curr = (J2_lo + J2_hi) / 2
    println("Finding J2 bound for UUUD Plateau")
    while !find_bound && iter ≤ max_iter
        println("--------------------------------")
        @show J2_curr, iter
        state_curr = UUUDPlateau(J₁, J2_curr, :scf)
        E_K = renormalized_single_magnon_energies_scf(state_curr, mean_field_values, h, q_K)[nbands]
        E_M = renormalized_single_magnon_energies_scf(state_curr, mean_field_values, h, q_M)[nbands]
        if isnan(E_K) || isnan(E_M)
            E = NaN
        else
            E = min(E_K, E_M)
        end
        println("Current lowest magnon energy E=$E")
        println("--------------------------------")
        if isnan(E)
            J2_lo = J2_curr
            J2_curr = (J2_lo + J2_hi) / 2
        elseif E > E_tol
            J2_hi = J2_curr
            J2_curr = (J2_lo + J2_hi) / 2
        else
            find_bound = true
        end
        iter += 1
    end
    return J2_curr, E
end

function magnetization_correction_scf(state::AbstractTriangularPlateau, h, mean_field_values; kwargs...)
    swt, _ = swts(state, h)
    scnlswt = SelfConsistentNLSWT(swt)
    @assert length(mean_field_values) == length(scnlswt.mean_field_values) "Mean field values length mismatch"
    for i in eachindex(mean_field_values)
        scnlswt.mean_field_values[i] = mean_field_values[i]
    end
    L = Sunny.nbands(swt)
    H1 = zeros(ComplexF64, 2L, 2L)
    H2 = zeros(ComplexF64, 2L, 2L)
    V = zeros(ComplexF64, 2L, 2L)
    δS = hcubature((0,0,0),(1,1,1); kwargs...) do q
        dynamical_matrix!(H1, swt, q)
        swt_hamiltonian_dipole_nlsw!(H2, scnlswt, q)
        @. H1 += H2
        bogoliubov!(V, H1)
        return SVector{L}(-norm2(view(V, L+i, 1:L)) for i in 1:L)
    end

    return δS[1]
end