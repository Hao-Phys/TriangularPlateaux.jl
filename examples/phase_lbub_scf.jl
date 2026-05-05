using TriangularPlateaux
using JLD2

state = ARGS[1]
num_J2s = parse(Int, ARGS[2])

hcubature_opts = (; atol=1e-4)
nlsolve_opts = (; ftol=1e-4, iterations=100, show_trace=true)
E_tol = 1e-3

if state == "uud"
    J2s = collect(range(0.0, stop=1/8, length=num_J2s))
    Δh = 1.5
    result_path = joinpath(@__DIR__, "pub_results")
    isdir(result_path) || mkpath(result_path)
    lbub_res = zeros(4, length(J2s))

    Threads.@threads for i in eachindex(J2s)
        J2 = J2s[i]
        uud_state = UUDPlateau(1.0, J2, :scf)
        h_lb, h_ub, E_lb, E_ub = find_lb_ub_scf(uud_state, Δh, result_path; hcubature_opts=hcubature_opts, nlsolve_opts=nlsolve_opts, E_tol=E_tol, max_iter=50)
        lbub_res[:, i] = [h_lb; h_ub; E_lb; E_ub]
    end

    jldsave(joinpath(result_path, "uud_lbub_phase_data.jld2"); J2s=J2s, lbub_res=lbub_res)

elseif state == "uuud"
    J2s = collect(range(1/8, stop=0.5, length=num_J2s))
    Δh = 1.5
    result_path = joinpath(@__DIR__, "pub_results")
    isdir(result_path) || mkpath(result_path)
    lbub_res = zeros(4, length(J2s))
    
    Threads.@threads for i in eachindex(J2s)
        J2 = J2s[i]
        uuud_state = UUUDPlateau(1.0, J2, :scf)
        h_lb, h_ub, E_lb, E_ub = find_lb_ub_scf(uuud_state, Δh, result_path; hcubature_opts=hcubature_opts, nlsolve_opts=nlsolve_opts, E_tol=E_tol)
        lbub_res[:, i] = [h_lb; h_ub; E_lb; E_ub]
    end
    jldsave(joinpath(result_path, "uuud_lbub_phase_data.jld2"); J2s=J2s, lbub_res=lbub_res)
else
    error("Unknown state: $state. Use uud or uuud.")
end