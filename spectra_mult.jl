using Plots
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

include("MCglauber.jl")
include("hdf5_io.jl")
include("observables.jl")
include("fastreso.jl")


default(linewidth = 2,
    markersize = 8,
    grid = false,size = (700, 450),
    guidefontsize = 12,
    tickfontsize = 10,
    dpi = 150)


data = hdf5_to_ObservableResult(pwd()*"/event_by_event_results_debug.h5")
glauber_vec = extract_glauber_multiplicity(data)
Nev= length(glauber_vec)
Fj = fastreso_reader(pwd()*"/PDGid_211_total_T0.1560_Fj.out")
particle_full_π = particle_full("pion",0.13957,1,0,Fj[1])
Fj = fastreso_reader(pwd()*"/Dc1865zer_total_T0.1560_Fj.out")
particle_full_D0 = particle_full("D0",1.86483,1,1,Fj[1])

species_list = [particle_full_π]

centrality_bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
data_chunks = centralities_selection_events(data,centrality_bins)

data_chunks[1][1950]
data_chunks[11][2]
vni = extract_vn(data_chunks[11])
any(isless.(vni[1, 3, :, 1, 1],0))
pt_list_pi = particle_full_π.pt_list
# 1. Initialize the plot
p = plot(xlabel="pt", ylabel="spectra", legend=:topright)

# 2. Define a color palette (10 distinct colors)
colors = palette(:tab10); # or [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :black, :gray]

# 3. Loop through your chunks
for i in 1:10
    # Extract the specific slice
    vni = extract_vn(data_chunks[i])
    cc = i*10
    # Plot all lines for this chunk in the same color, no labels
    plot!(pt_list_pi, vni[1, 3, :, 1, 1], color=colors[i], label="$cc %", alpha=0.5)
    plot!(pt_list_pi, transpose(vni[2:end, 3, :, 1, 1]), color=colors[i], label="", alpha=0.5)
end
display(p)

p = plot(xlabel="pt", ylabel="spectra", legend=:topright)

# 2. Define a color palette (10 distinct colors)
colors = palette(:tab10); # or [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :black, :gray]

# 3. Loop through your chunks
for i in 1:10
    vni = spectra(data_chunks[i],species_list)
    cc = i*10
    plot!(pt_list_pi, vni[:, 1], color=colors[i], label="$cc %")
end
display(p)

p = plot(xlabel="cc", ylabel="multiplicity", legend=:topright)
colors = palette(:tab10); # or [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :black, :gray]
for i in 1:9
    # Extract the specific slice
    vni = extract_vn(data_chunks[i])
    mult = [sum(vni[i,3,:,1,1]) for i in axes(vni,1)]
    cc = i*10
    # Plot all lines for this chunk in the same color, no labels
    plot!(range(cc-10,cc,length(mult)),mult,color=colors[i],yscale=:log10)
end
display(p)

p = plot(xlabel="cc", ylabel="multiplicity", legend=:topright)

# 2. Define a color palette (10 distinct colors)
colors = palette(:tab10); # or [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :black, :gray]

# 3. Loop through your chunks
for i in 1:10
    # Extract the specific slice
    vni = extract_vn(data_chunks[i])
    mult = [sum(vni[i,3,:,1,1]) for i in axes(vni,1)]
    cc = i*10
    # Plot all lines for this chunk in the same color, no labels
    plot!(range(cc-10,cc,length(mult)),mult,color=colors[i],label="$cc %")
end
display(p)
