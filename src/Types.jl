abstract type AbstractTriangularPlateau end

struct UUDPlateau <: AbstractTriangularPlateau
    J₁ :: Float64
    J₂ :: Float64
    mode :: Symbol
    nbands :: Int
    q_ordering :: Vec3
end

function UUDPlateau(J₁::Float64, J₂::Float64, mode::Symbol)
    @assert mode in (:scf, :one_loop, :one_loop_prime) "Mode must be one of :scf, :one_loop, or :one_loop_prime"
    return UUDPlateau(J₁, J₂, mode, 3, Vec3(1/3, 1/3, 0))
end

struct UUUDPlateau <: AbstractTriangularPlateau
    J₁ :: Float64
    J₂ :: Float64
    mode :: Symbol
    nbands :: Int
    q_ordering :: Vec3
end

function UUUDPlateau(J₁::Float64, J₂::Float64, mode::Symbol)
    @assert mode in (:scf, :one_loop, :one_loop_prime) "Mode must be one of :scf, :one_loop, or :one_loop_prime"
    return UUUDPlateau(J₁, J₂, mode, 4, Vec3(1/2, 1/2, 0))
end

function calculate_hc(state::UUDPlateau)
    (; J₁) = state
    s = 1/2
    hc = 3J₁*s
    return hc 
end

function calculate_hc(state::UUUDPlateau)
    (; J₁, J₂) = state
    s = 1/2
    hc = J₂ > J₁/8 ? 4(J₁+J₂)*s : 4(J₁+1/8)*s
    return hc 
end