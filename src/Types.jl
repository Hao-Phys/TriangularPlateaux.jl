abstract type AbstractTriangularPlateau end

mutable struct UUDPlateau <: AbstractTriangularPlateau
    J₁ :: Float64
    J₂ :: Float64
    h  :: Float64
    q_ordering :: Vec3
    mean_field_values :: Vector{ComplexF64}
end

UUDPlateau(J1::Float64, J2::Float64, h::Float64) = UUDPlateau(J1, J2,h, Vec3(1/3, 1/3, 0), ComplexF64[])

mutable struct UUUDPlateau <: AbstractTriangularPlateau
    J₁ :: Float64
    J₂ :: Float64
    h  :: Float64
    q_ordering :: Vec3
    mean_field_values :: Vector{ComplexF64}
end

UUUDPlateau(J1::Float64, J2::Float64, h::Float64) = UUUDPlateau(J1, J2, h, Vec3(1/2, 1/2, 0), ComplexF64[])