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
using LaTeXStrings
using Statistics

default(lw = 2, size=(800,600),xtickfontsize=16,ytickfontsize=16,xlabelfontsize=16,ylabelfontsize=16,legendfontsize=16,grid=false,framestyle=:box)

include("MCglauber.jl")
include("hdf5_io.jl")
include("observables.jl")
include("fastreso.jl")

data = hdf5_to_ObservableResult(pwd()*"/event_by_event_results_debug.h5")
glauber_vec = extract_glauber_multiplicity(data)
Nev= length(glauber_vec)
Fj = fastreso_reader(pwd()*"/PDGid_211_total_T0.1560_Fj.out")
particle_full_π = particle_full("pion",0.13957,1,0,Fj[1])
Fj = fastreso_reader(pwd()*"/Dc1865zer_total_T0.1560_Fj.out")
particle_full_D0 = particle_full("D0",1.86483,1,1,Fj[1])

species_list = [particle_full_π]


centrality_bins=[10,20,30,40,50,60]
data_chunks = centralities_selection_events(data,centrality_bins)

pt_list_pi = particle_full_π.pt_list

q_vector_event_pt_dependent(data_chunks[1][1],species_list,[2,3])
#q_vector_event_pt_dependent_im(data_chunks[1][1],species_list,[2,3])
multiplicity_event(data_chunks[1][1],species_list)

g = g_species_event_pt_dependent(data_chunks[2][190],species_list)
sum(g)
q_vector_event_integrated_complex(data_chunks[2][1],species_list,[2,3])


vns = [harmonic_coefficient_complex(data_chunks[i],species_list,[2,3]) for i in eachindex(centrality_bins)]
plot(pt_list_pi, (real.(vns[1].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 0-10", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
#plot!(pt_list_pi,(real.(vns[1].vm_result[:,1,2])), label = L"v_3\, \mathrm{\pi} 0-10", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(pt_list_pi,(real.(vns[2].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 10-20", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
#plot!(pt_list_pi,(real.(vns[2].vm_result[:,1,2])), label = L"v_3\, \mathrm{\pi} 10-20", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(pt_list_pi,(real.(vns[3].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 20-30", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(pt_list_pi,(real.(vns[4].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 40-50", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(pt_list_pi,(real.(vns[5].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 50-60", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")

#plot!(pt_list_pi,(real.(vns[4].vm_result[:,1,2])), label = L"v_3\, \mathrm{\pi} 20-30", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"Re v_n")

#plot!(vns[1].vm_result[:,2,1], label = L"v_2\, \mathrm{D^0}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
#plot!(vns[1].vm_result[:,2,2], label = L"v_3\, \mathrm{D^0}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(vns[1].vm_result_charged[:,1,1], label = L"v_2\, \mathrm{charged}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(vns[1].vm_result_charged[:,1], label = L"v_3\, \mathrm{charged}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n",legendtitle = L"\mathrm{0-10\% \,O-O}")


plot!(vns[2].vm_result[:,1,1], label = L"v_2\, \mathrm{\pi}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(vns[2].vm_result[:,1,2], label = L"v_3\, \mathrm{\pi}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
#plot!(vns[3].vm_result[:,2,1], label = L"v_2\, \mathrm{D^0}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
#plot!(vns[3].vm_result[:,2,2], label = L"v_3\, \mathrm{D^0}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(vns[3].vm_result_charged[:,1,1], label = L"v_2\, \mathrm{charged}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n")
plot!(vns[3].vm_result_charged[:,1,2], label = L"v_3\, \mathrm{charged}", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n",legendtitle = L"\mathrm{20-30\% \,O-O}")
vns[1].vm_result_charged_integrated

scatter(centrality_bins,[real(vns[i].vm_result_integrated[1,1]) for i in eachindex(centrality_bins)],xlabel = "Centrality class", label = L"v_2\, \mathrm{\pi}")
real(vns[4].vm_result_integrated[1,1])

vni = extract_vn(data_chunks[4])
mult = mean([sum(vni[i,3,:,1,1]) for i in axes(vni,1)])
mult_per_bin = mean(vni, dims = 1)[1,3,:,1,1]
vn_pt = real.(vns[4].vm_result[:,1,1])
sum(transpose(vn_pt)*mult_per_bin)/mult
#plot!([vns[i].vm_result_integrated[2,1] for i in 1:10],xlabel = "Centrality class", label = L"v_2\, \mathrm{D^0}")

plot!([real(vns[i].vm_result_integrated[1,2]) for i in 1:4])



#single events

    pTlists = pt_list.(species_list)        
    pt_length_max = maximum(length.(pt_list.(species_list)))
    
    vm_result = zeros(pt_length_max,length(species_list),length(wavenum_list))
    vm_result_integrated = zeros(length(species_list),length(wavenum_list))
    vm_result_charged = zeros(pt_length_max,length(wavenum_list))
    vm_result_charged_integrated = zeros(length(wavenum_list))
        
    for result in event_list        
        q_vector_pt = q_vector_event_pt_dependent(result,species_list,wavenum_list)
        q_vector_total = q_vector_event_integrated(result,species_list,wavenum_list)

        for k in eachindex(species_list)
            pTlist = pTlists[k]
            for i in eachindex(pTlist)
                for wavenum in eachindex(wavenum_list)
                    vm_result[i,k,wavenum] += q_vector_pt[i,k,wavenum]*q_vector_total[wavenum]/length(event_list)
                end
            end
        end

    end


