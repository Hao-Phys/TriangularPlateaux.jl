# The rigorous version of one-loop is not implmented in Sunny yet. Here is the function with some tweaks with tools avialable in Sunny
function renormalized_single_magnon_energies_one_loop(state::AbstractTriangularPlateau, h, q::Vec3; hcubature_opts::NamedTuple=NamedTuple(;))
    (; nbands) = state
    swt, swt_ref = swts(state, h)
    q_reshaped = Sunny.to_reshaped_rlu(swt.sys, q)

    real_space_cubic_vertices = calculate_real_space_cubic_vertices_dipole(swt.sys)
    real_space_quartic_vertices = calculate_real_space_quartic_vertices_dipole(swt.sys)

    ptt_ref = PerturbativeTheory(swt_ref; hcubature_opts...)

    ptt = PerturbativeTheory(swt, real_space_quartic_vertices, real_space_cubic_vertices, ptt_ref.mean_field_values)

    H_buf1 = zeros(ComplexF64, 2nbands, 2nbands)
    V_buf1 = zeros(ComplexF64, 2nbands, 2nbands)
    H_buf2 = zeros(ComplexF64, 2nbands, 2nbands)
    H_buf3 = zeros(ComplexF64, 2nbands, 2nbands)

    Sunny.dynamical_matrix!(H_buf1, swt_ref, q_reshaped)
    Sunny.dynamical_matrix!(H_buf3, swt, q_reshaped)
    @. H_buf3 -= H_buf1
    H_buf3 = Diagonal(H_buf3)

    # LSWT energies at the critical field hc
    E = Sunny.bogoliubov!(V_buf1, H_buf1)[1:nbands]

    # The additional Zeeman field contribution with respect to the critical field hc
    E_field = real.(diag(V_buf1' * H_buf3 * V_buf1))[1:nbands]
    @. E += E_field

    # The one-loop (1/S) corrections from the quartic vertices
    Sunny.swt_hamiltonian_dipole_nlsw!(H_buf2, ptt, q_reshaped)
    E_4 = real.(diag(V_buf1' * H_buf2 * V_buf1))[1:nbands]
    @. E += E_4

    sort!(E, rev=true)
    return E
end

function find_lb_ub_one_loop(state::AbstractTriangularPlateau, Δh, result_path::String; E_tol::Float64=1e-3, max_iter::Int=50, hcubature_opts::NamedTuple=NamedTuple(;))
    (; J₂, nbands) = state
    isdir(result_path) || mkpath(result_path)

    iter = 1
    find_lb = false
    hc = calculate_hc(state)
    h_curr = hc - Δh
    h_lo = h_curr
    h_hi = hc

    if typeof(state) == UUDPlateau
        @assert J₂ ≤ 1/8 * state.J₁ "UUD Plateau state unstable for J₂/J₁ > 1/8. This code cannot be used to find lb/ub in that regime."
        q_ordering = Vec3(1/3, 1/3, 0)
    elseif typeof(state) == UUUDPlateau
        @assert J₂ ≥ 1/8 * state.J₁ "UUUD Plateau state unstable for J₂/J₁ < 1/8. This code cannot be used to find lb/ub in that regime."
        q_ordering = Vec3(1/2, 1/2, 0)
    else
        error("Unsupported state type: $(typeof(state))")
    end

    E = renormalized_single_magnon_energies_one_loop(state, h_curr, q_ordering; hcubature_opts)[nbands]

    @warn !isnan(E) "Lowest magnon energy is $E at h=$h_curr, for J₂=$(round(J₂, digits=4)), consider using more aggressive guess for Δh"
    h_curr = (h_hi + h_lo) / 2

    println("Finding lower bound field h_lb")
    while !find_lb && iter ≤ max_iter
        println("--------------------------------")
        @show J₂, iter, h_curr
        E = renormalized_single_magnon_energies_one_loop(state, h_curr, q_ordering; hcubature_opts)[nbands]
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

    E = renormalized_single_magnon_energies_one_loop(state, h_curr, q_ordering; hcubature_opts)[nbands]
    @warn !isnan(E) "Lowest magnon energy is $E at h=$h_curr, for J₂=$(round(J₂, digits=4)), consider using more aggressive guess for Δh"
    h_curr = (h_hi + h_lo) / 2

    println("Finding upper bound field h_ub")
    while !find_ub && iter ≤ max_iter
        println("--------------------------------")
        @show J₂, iter, h_curr
        E = renormalized_single_magnon_energies_one_loop(state, h_curr, q_ordering; hcubature_opts)[nbands]
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