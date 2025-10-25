function renormalized_single_magnon_energies_scf(state::AbstractTriangularPlateau, mean_field_values::Vector{ComplexF64}, h, result_path::String;hcubature_opts::NamedTuple=NamedTuple(;), nlsolve_opts::NamedTuple=NamedTuple(;))
    (; q_ordering, nbands) = state
    swt, _ = swts(state, h)
    scnlswt = SelfConsistentNLSWT(swt)
    try
       solve_self_consistent_nlswt!(scnlswt; mean_field_values, hcubature_opts, nlsolve_opts) 
       try
        E, _ = excitations_scnlswt(scnlswt, q_ordering)
        E_renormalized = E[1:nbands]
        jldsave(joinpath(result_path, "scf_$(typeof(state))_J1_$(state.J₁)_J2_$(state.J₂)_h_$(round(h, digits=4)).jld2"); E_renormalized=E_renormalized, mean_field_values=scnlswt.mean_field_values, h=h)
        return E_renormalized
       catch _
        @warn "Failed to compute renormalized energies after SCF, skipping renormalized energies calculation"
        E = zeros(nbands)
        fill!(E, NaN)
        return E
       end
    catch _
        @warn "Failed SCF, skipping renormalized energies calculation"
        E = zeros(nbands)
        fill!(E, NaN)
        return E
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
    @show "Calculating reference mean field values of swt_ref"
    scnlswt = SelfConsistentNLSWT(swt)
    solve_self_consistent_nlswt!(scnlswt; mean_field_values, hcubature_opts, nlsolve_opts)
    mean_field_values0 = copy(scnlswt.mean_field_values)
    @show "Reference mean field values calculated"

    E = renormalized_single_magnon_energies_scf(state, mean_field_values0, h_curr, result_path; hcubature_opts, nlsolve_opts)[nbands]
    @assert isnan(E) "Use more aggressive guess for Δh"
    h_curr = (h_hi + h_lo) / 2

    while !find_lb && iter ≤ max_iter
        @show "--------------------------------"
        @show "J₂=$J₂, Iteration $iter, h_curr=$h_curr"
        E = renormalized_single_magnon_energies_scf(state, mean_field_values0, h_curr, result_path; hcubature_opts, nlsolve_opts)[nbands]
        @show "Lowest magnon energy E=$E"
        @show "--------------------------------"
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

function find_ub_scf(state::AbstractTriangularPlateau, Δh; E_tol::Float64=1e-4, max_iter::Int=50, result_path::String=joinpath(@__DIR__, "results"), hcubature_opts::NamedTuple=NamedTuple(;), nlsolve_opts::NamedTuple=NamedTuple(;))
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
    @show "Calculating reference mean field values of swt_ref"
    scnlswt = SelfConsistentNLSWT(swt)
    solve_self_consistent_nlswt!(scnlswt; mean_field_values, hcubature_opts, nlsolve_opts)
    mean_field_values0 = copy(scnlswt.mean_field_values)
    @show "Reference mean field values calculated"

    E = renormalized_single_magnon_energies_scf(state, mean_field_values0, h_curr, result_path; hcubature_opts, nlsolve_opts)[nbands]
    @assert isnan(E) "Use more aggressive guess for Δh"
    h_curr = (h_hi + h_lo) / 2

    while !find_ub && iter ≤ max_iter
        @show "--------------------------------"
        @show "J₂=$J₂, Iteration $iter, h_curr=$h_curr"
        E = renormalized_single_magnon_energies_scf(state, mean_field_values0, h_curr, result_path; hcubature_opts, nlsolve_opts)[nbands]
        @show "Lowest magnon energy E=$E"
        @show "--------------------------------"
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