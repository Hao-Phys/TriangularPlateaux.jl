function swts(state::UUDPlateau)
    (; J₁, J₂, h) = state
    s = 1/2

    a = b = 1.0
    c = 10a
    lat_vecs = lattice_vectors(a, b, c, 90, 90, 120)
    basis_vecs = [[0, 0, 0]]

    L0 = 3
    cryst = Crystal(lat_vecs, basis_vecs)
    sys = System(cryst, [1 => Moment(; s, g = -1)], :dipole; dims=(L0, L0, 1))
    set_exchange!(sys, J₁, Bond(1,1,[1,0,0]))
    set_exchange!(sys, J₂, Bond(1,1,[1,2,0]))
    set_field!(sys, [0,0,h])

    set_dipole!(sys, ( 0,0, 1), (1,1,1,1))
    set_dipole!(sys, ( 0,0, 1), (2,1,1,1))
    set_dipole!(sys, ( 0,0,-1), (2,2,1,1))
    set_dipole!(sys, ( 0,0, 1), (3,2,1,1))
    set_dipole!(sys, ( 0,0,-1), (3,1,1,1))
    set_dipole!(sys, ( 0,0, 1), (2,3,1,1))
    set_dipole!(sys, ( 0,0,-1), (1,3,1,1))
    set_dipole!(sys, ( 0,0, 1), (1,2,1,1))
    set_dipole!(sys, ( 0,0, 1), (3,3,1,1))

    sys_min = reshape_supercell(sys, [1 0 0; -1 3 0; 0 0 1])
    swt = SpinWaveTheory(sys_min; measure=ssf_perp(sys_min))

    hc = 3J₁*s
    set_field!(sys, [0,0,hc])

    if J₂ > 1/8 * J₁
        set_exchange!(sys, 1/8, Bond(1,1,[1,2,0]))
        @warn "J₂/J₁ > 1/8. UUD States classically unstable. Setting J₂=1/8J₁ to continue the reference `swt_ref` object"
    end

    sys_min_ref = reshape_supercell(sys, [1 0 0; -1 3 0; 0 0 1])
    swt_ref = SpinWaveTheory(sys_min_ref; measure=ssf_perp(sys_min_ref))

    return swt, swt_ref
end

function swts(state::UUUDPlateau)
    (; J₁, J₂, h) = state
    s = 1/2

    a = b = 1.0
    c = 10a
    lat_vecs = lattice_vectors(a, b, c, 90, 90, 120)
    basis_vecs = [[0, 0, 0]]

    L0 = 2
    cryst = Crystal(lat_vecs, basis_vecs)
    sys = System(cryst, [1 => Moment(; s, g = -1)], :dipole; dims=(L0, L0, 1))
    set_exchange!(sys, J₁, Bond(1,1,[1,0,0]))
    set_exchange!(sys, J₂, Bond(1,1,[1,2,0]))
    set_field!(sys, [0,0,h])

    set_dipole!(sys, ( 0,0, 1), (1,1,1,1))
    set_dipole!(sys, ( 0,0, 1), (2,1,1,1))
    set_dipole!(sys, ( 0,0, 1), (1,2,1,1))
    set_dipole!(sys, ( 0,0,-1), (2,2,1,1))

    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))

    hc = 4(J₁+J₂)*s

    set_field!(sys, [0,0,hc])
    if J₂ < 1/8 * J₁
        set_exchange!(sys, 1/8, Bond(1,1,[1,2,0]))
        @warn "J₂/J₁ < 1/8. UUUD States classically unstable. Setting J₂=1/8J₁ to continue the reference `swt_ref` object"
    end

    swt_ref = SpinWaveTheory(sys; measure=ssf_perp(sys))

    return swt, swt_ref
end