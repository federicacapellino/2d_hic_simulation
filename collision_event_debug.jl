using Fluidum
using MonteCarloGlauber
using StaticArrays
using LinearAlgebra
using Integrals
using Cuba
using JLD2
using HDF5
using MuladdMacro
using OhMyThreads
using Base.Threads
using YAML
using Plots
@info "Julia threads" nthreads()

include("MCglauber.jl")
include("hdf5_io.jl")
include("observables.jl")
include("fastreso.jl")

# the convention here are T, ux, uy, piyy, pizz, pixy, piB this has to match with the matrix defined
twod_visc_hydro = Fields(
    NDField((:even, :ghost), (:even, :ghost), :temperature),
    NDField((:ghost, :ghost), (:ghost, :ghost), :ux),
    NDField((:ghost, :ghost), (:ghost, :ghost), :uy),
    NDField((:ghost, :ghost), (:even, :ghost), :piyy),
    NDField((:even, :ghost), (:even, :ghost), :pizz),
    NDField((:ghost, :ghost), (:ghost, :ghost), :pixy),
    NDField((:even, :ghost), (:even, :ghost), :piB),
    NDField((:even, :ghost), (:even, :ghost), :mu),
    NDField((:ghost, :ghost), (:ghost, :ghost), :nux),
    NDField((:ghost, :ghost), (:ghost, :ghost), :nuy)
)

# Load configuration from YAML
config = YAML.load_file("./examples/event-by-event/config.yaml")
physics_params = config["physics parameter"]
run_params = config["run parameter"]

# Run parameters
const Nev = run_params["Nev"]
const checkpoint_interval = run_params["checkpoint_interval"]
const checkpoint_file = run_params["checkpoint_file"]
const gridpoints = run_params["gridpoints"]
const xmax = run_params["xmax"]
const wavenum_m = run_params["wavenum"]

# Fluid evolution parameters
const tau0 = physics_params["fluid evolution"]["tau0"]
const tmax = physics_params["fluid evolution"]["tmax"]
const norm = physics_params["fluid evolution"]["norm"]
const eta_over_s = physics_params["fluid evolution"]["eta_over_s"]
const zeta_over_s = physics_params["fluid evolution"]["zeta_over_s"]
const DsT = physics_params["fluid evolution"]["DsT"]
const Tfo = physics_params["fluid evolution"]["Tfo"]

# Charm parameters
const dσ_QQdy = physics_params["charm parameter"]["dσ_QQdy"]
const σ_in = physics_params["charm parameter"]["σ_in"]

# Initial condition parameters
const s_NN = physics_params["initial conditions"]["s_NN"]
const w = physics_params["initial conditions"]["w"]
const k = physics_params["initial conditions"]["k"]
const p = physics_params["initial conditions"]["p"]
b = physics_params["initial conditions"]["b"]

# Additional parameters
const tau_eta_par = physics_params["additional parameters"]["tau_eta_par"]
const tau_zeta_par = physics_params["additional parameters"]["tau_zeta_par"]
const hq_mass = physics_params["additional parameters"]["hq_mass"]

# Set up nuclear parameters
spec = physics_params["initial conditions"]["n1"]  # e.g. "Lead()"
const n1 = eval(Meta.parse(spec))
spec = physics_params["initial conditions"]["n2"]  # e.g. "Lead()"
const n2 = eval(Meta.parse(spec))

# Create discretization
discretization = CartesianDiscretization(Fluidum.SymmetricInterval(gridpoints, xmax), Fluidum.SymmetricInterval(gridpoints, xmax))
# Prepare the field with the discretization
twod_visc_hydro_discrete = DiscreteFields(twod_visc_hydro, discretization, Float64)

if b == "minBias"
    participants = Participants(n1, n2, w, s_NN, k, p)
else
    b_tuple = eval(Meta.parse(b))
    participants = Participants(n1, n2, w, s_NN, k, p, b_tuple)
end

pion = particle_simple("pion", 0.13957, 1, 0)
D0 = particle_simple("D0", 1.86483, 1, 1)

Fj = fastreso_reader(pwd() * "/examples/event-by-event/PDGid_211_total_T0.1560_Fj.out")
const particle_full_π = particle_full("pion", 0.13957, 1, 0, Fj[1])
Fj = fastreso_reader(pwd() * "/examples/event-by-event/Dc1865zer_total_T0.1560_Fj.out")
const particle_full_D0 = particle_full("D0", 1.86483, 1, 1, Fj[1])

species_list = [particle_full_π]
event = rand(participants)
mult, x_com, y_com = center_of_mass(event, 100, 50)
xcm = x_com / mult
ycm = y_com / mult
profile = map(discretization.grid) do y
    y = y .+ (xcm, ycm)
    event(y...)
end
ncoll_event = event.n_coll
tspan = (tau0, tmax)
ccbar = ncoll_event * 2 * dσ_QQdy / σ_in
ccbar_norm = 2 * dσ_QQdy / σ_in / tau0
charm_number_hard(ncoll_event; xmax=xmax, ncoll_norm=ccbar_norm).u
eos = Heavy_Quark(readresonancelist(), ccbar)

viscosity = QGPViscosity(eta_over_s, tau_eta_par)
bulk = SimpleBulkViscosity(zeta_over_s, tau_zeta_par)
diffusion = ZeroDiffusion()
fluidproperty = FluidProperties(eos, viscosity, bulk, diffusion)
temperature_func = trento_event_eos(profile, norm=norm, exp_tail=false)


phi = set_array(temperature_func .+ 0.01, :temperature, twod_visc_hydro_discrete)

@warn "no diffusion, only pions"
fug_func = fug_(temperature_func, ncoll_event, eos, discretization; ncoll_norm=ccbar_norm)
#set_array!(phi, fug_func, :mu, twod_visc_hydro_discrete)

#heatmap(phi[8,:,:], title="phi[8,:,:] (mu field)", xlabel="x", ylabel="y", colorbar_title="μ", aspect_ratio=:equal)
#heatmap(phi[1,:,:], title="phi[1,:,:] (mu field)", xlabel="x", ylabel="y", colorbar_title="μ", aspect_ratio=:equal)
#plot(phi[8,50,:])
#plot!(phi[1,50,:])
@info "Max phi[8,:,:]" maximum(phi[8,:,:])
@info "Min phi[8,:,:]" minimum(phi[8,:,:])

result = Fluidum.isosurface(twod_visc_hydro_discrete, Fluidum.matrix2d_visc_HQ!, fluidproperty, phi, tspan, :temperature, Tfo)
cha = Fluidum.Chart(Fluidum.Surface(result[:surface]), (t, x, y) -> Fluidum.SVector{2}(atan(t, hypot(y, x)), atan(y, x)))
if length(cha.points) == 0
    @warn "No freezeout points found! Skipping observables..."
else
    fo_bg = Fluidum.freezeout_interpolation(cha, sort_index=2, ndim_tuple=50)
    vn = dvn_dp_list_delta(fo_bg, species_list, 0.0, wavenum_m; eta_min=-5.0, eta_max=5.0)
    observable_result = ObservableResult(mult, vn.u)
end

glauber_vec = extract_glauber_multiplicity([observable_result])
vn_vector = extract_vn([observable_result])
vn_vector[1,2,:,:,1]
vn_vector[1,1,:,:,1]

spectra_pion1 = vn_vector[1,3,:,1,1]

spectra_D01 = vn_vector[1,3,:,1,2]

multiplicity_event(observable_result,[pion,D0])
multiplicity_event(observable_result,[pion,D0])

sum(spectra_pion1)
sum(spectra_D01)
species_list = [particle_full_π, particle_full_D0]
g = g_species_event_pt_dependent(observable_result,species_list)
sum(g)
q_vector_event_integrated(observable_result,species_list,[2,3])
vns = [harmonic_coefficient([observable_result],species_list,[2,3]) for i in 1:10]
using LaTeXStrings
default(lw = 2, size=(800,600),xtickfontsize=16,ytickfontsize=16,xlabelfontsize=16,ylabelfontsize=16,legendfontsize=16,grid=false,framestyle=:box)

plot!(vns[1].vm_result[:,1,1], label = L"v_2\, \mathrm{\pi}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(vns[1].vm_result[:,1,2], label = L"v_3\, \mathrm{\pi}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(vns[1].vm_result[:,2,1], label = L"v_2\, \mathrm{D^0}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(vns[1].vm_result[:,2,2], label = L"v_3\, \mathrm{D^0}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(vns[1].vm_result_charged[:,1,1], label = L"v_2\, \mathrm{charged}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(vns[1].vm_result_charged[:,1,2], label = L"v_3\, \mathrm{charged}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n",legendtitle = L"\mathrm{0-10\% \,O-O}")


