function renormalized_single_magnon_energies_scf(state::AbstractTriangularPlateau, mean_field_values::Vector{ComplexF64}, h, result_path::String; hcubature_opts::NamedTuple=NamedTuple(;), nlsolve_opts::NamedTuple=NamedTuple(;))
    (; q_ordering, nbands) = state
    ftol = get(nlsolve_opts, :ftol, 1e-8)
    swt, _ = swts(state, h)
    scnlswt = SelfConsistentNLSWT(swt)
    try
       sol_ftol = solve_self_consistent_nlswt!(scnlswt; mean_field_values, hcubature_opts, nlsolve_opts)
       if sol_ftol > ftol
           @warn "SCF did not converge to desired tolerance ftol=$ftol, got sol_ftol=$sol_ftol. Returing to NaN energies"
           return fill(NaN, nbands)
       end
       try
            E, _ = excitations_scnlswt(scnlswt, q_ordering)
            E_renormalized = E[1:nbands]
            jldsave(joinpath(result_path, "scf_$(typeof(state))_J1_$(state.J₁)_J2_$(round(state.J₂, digits=4))_h_$(round(h, digits=4)).jld2"); E_renormalized=E_renormalized, mean_field_values=scnlswt.mean_field_values, h=h)
            return E_renormalized
       catch _
            @warn "Failed to compute renormalized energies after SCF, skipping renormalized energies calculation"
            return fill(NaN, nbands)
       end
    catch _
        @warn "Failed SCF, skipping renormalized energies calculation"
        return fill(NaN, nbands)
    end
end

function find_lb_scf(state::AbstractTriangularPlateau, Δh, result_path::String; E_tol::Float64=1e-4, max_iter::Int=50, hcubature_opts::NamedTuple=NamedTuple(;), nlsolve_opts::NamedTuple=NamedTuple(;))
    (; J₂, nbands) = state
    isdir(result_path) || mkpath(result_path)

    iter = 1
    find_lb = false
    hc = calculate_hc(state)
    h_curr = hc - Δh
    h_lo = h_curr
    h_hi = hc

    swt, swt_ref = swts(state, hc)
    mean_field_values = calculate_mean_field_values_lswt(swt_ref; hcubature_opts...)
    println("Calculating reference mean field values of swt_ref")
    scnlswt = SelfConsistentNLSWT(swt)
    try
         solve_self_consistent_nlswt!(scnlswt; mean_field_values, hcubature_opts, nlsolve_opts)
    catch _
        @warn "Failed SCF for reference state, cannot proceed. The corresponding state cannot be stable"
        return NaN, NaN
    end

    mean_field_values0 = copy(scnlswt.mean_field_values)
    println("Reference mean field values calculated")

    E = renormalized_single_magnon_energies_scf(state, mean_field_values0, h_curr, result_path; hcubature_opts, nlsolve_opts)[nbands]
    @assert isnan(E) "Use more aggressive guess for Δh"
    h_curr = (h_hi + h_lo) / 2

    while !find_lb && iter ≤ max_iter
        println("--------------------------------")
        @show J₂, iter, h_curr
        E = renormalized_single_magnon_energies_scf(state, mean_field_values0, h_curr, result_path; hcubature_opts, nlsolve_opts)[nbands]
        println("Lowest magnon energy E=$E")
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

    return h_curr, E
end

function find_ub_scf(state::AbstractTriangularPlateau, Δh, result_path::String; E_tol::Float64=1e-4, max_iter::Int=50, hcubature_opts::NamedTuple=NamedTuple(;), nlsolve_opts::NamedTuple=NamedTuple(;))
    (; J₁, J₂, nbands) = state

    isdir(result_path) || mkpath(result_path)

    iter = 1
    find_ub = false
    hc = calculate_hc(state)
    h_curr = hc + Δh
    h_lo = hc
    h_hi = h_curr

    swt, swt_ref = swts(state, hc)
    mean_field_values = calculate_mean_field_values_lswt(swt_ref; hcubature_opts...)
    println("Calculating reference mean field values of swt_ref")

    scnlswt = SelfConsistentNLSWT(swt)
    try
        solve_self_consistent_nlswt!(scnlswt; mean_field_values, hcubature_opts, nlsolve_opts)
    catch _
        @warn "Failed SCF for reference state, cannot proceed. The corresponding state cannot be stable"
        return NaN, NaN
    end

    mean_field_values0 = copy(scnlswt.mean_field_values)
    println("Reference mean field values calculated")
    E = renormalized_single_magnon_energies_scf(state, mean_field_values0, h_curr, result_path; hcubature_opts, nlsolve_opts)[nbands]
    @assert isnan(E) "Use more aggressive guess for Δh"
    h_curr = (h_hi + h_lo) / 2

    while !find_ub && iter ≤ max_iter
        println("--------------------------------")
        @show J₂, iter, h_curr
        E = renormalized_single_magnon_energies_scf(state, mean_field_values0, h_curr, result_path; hcubature_opts, nlsolve_opts)[nbands]
        println("Lowest magnon energy E=$E")
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

    return h_curr, E
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