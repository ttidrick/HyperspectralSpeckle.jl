using GLMakie
using PerceptualColourMaps
GLMakie.activate!()

const PLOT_OPTIONS = (
    CMAP_OBJECT = cmap("L01"),
    CMAP_WAVEFRONT = cmap("L20")#:haline
)


mutable struct ReconstructionFigures
    object_fig::Figure
    opd_fig::Figure
    heights_fig::Figure
    object_ax::Axis
    opd_ax::Vector{Axis}
    heights_ax::Axis
    object_obs::Observable
    opd_obs::Vector{Observable}
    heights_obs::Vector{Observable}
    ϵ_obs::Vector{Observable}
    function ReconstructionFigures()
        return new()
    end
end

function plot_object(object; show=true, write=false, filename="", label="Flux [ph/nm]")
    fig = Figure(size=(600, 600))
    o_ax = Axis(fig[1, 1], titlesize=18)
    o_ax.title = "Object [mean]"
    hidedecorations!(o_ax)
    o_ax.aspect = DataAspect()
    o_obs = Observable(rotr90(dropdims(mean(object.object, dims=3), dims=3)))
    o_plt = plot!(o_ax, o_obs, colormap=PLOT_OPTIONS[:CMAP_OBJECT])
    c = Colorbar(fig[1, 2], o_plt, width=10, height=Relative(1), label=label, labelsize=16, ticklabelsize=16)
    trim!(fig.layout)

    if show == true
        display(GLMakie.Screen(), fig)
    end

    if write == true
        save(filename, fig, px_per_unit=2)
        println("Figure written to $(filename)")
    end
    return fig, o_ax, o_obs
end

function plot_layers(atmosphere; show=false, write=false, filename="", label="OPD [nm]")
    fig = Figure(size=(400*atmosphere.nlayers, 400))
    l_ax = [Axis(fig[1, 2*l], titlesize=18) for l=1:atmosphere.nlayers]
    [l_ax[l].title = "Layer $(l) - $(label)" for l=1:atmosphere.nlayers]
    [hidedecorations!(ax) for ax in l_ax]
    [ax.aspect = DataAspect() for ax in l_ax]
    l_obs = [Observable(rotr90(atmosphere.opd[:, :, l])) for l=1:atmosphere.nlayers]
    l_plt = [plot!(l_ax[l], l_obs[l], colormap=PLOT_OPTIONS[:CMAP_WAVEFRONT]) for l=1:atmosphere.nlayers]
    [Colorbar(fig[1, 2*l+1], l_plt[l], width=10, height=Relative(3/4), labelsize=16, ticklabelsize=16) for l=1:atmosphere.nlayers]
    trim!(fig.layout)
    
    if show == true
        display(GLMakie.Screen(), fig)
    end

    if write == true
        save(filename, fig, px_per_unit=2)
        println("Figure written to $(filename)")
    end
    return fig, l_ax, l_obs
end

function plot_heights(atmosphere; heights=[], ϵ=[], show=false)
    FTYPE = gettype(atmosphere)
    if (heights == []) || (ϵ == [])
        heights = [FTYPE[] for ~=1:atmosphere.nlayers-1]
        ϵ = [FTYPE[] for ~=1:atmosphere.nlayers-1]
    end
    ϵ_real = cat(ϵ..., dims=1)
    ϵ_real = ϵ_real[.!(isnan.(ϵ_real))]
    base10exp = (length(ϵ_real)==0) ? 0 : floor(Int64, log10(minimum(ϵ_real)))
    ylabel = rich("Criterion [×10", superscript(string(base10exp)) , "]")
    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1], xlabel="Height [km]", ylabel=ylabel, xlabelsize=18, xticklabelsize=16, ylabelsize=18, yticklabelsize=16)
    heights_obs = [Observable(heights[l]) for l=1:atmosphere.nlayers-1]
    ϵ_obs = [Observable(ϵ[l] ./ 10^base10exp) for l=1:atmosphere.nlayers-1]
    for l=1:atmosphere.nlayers-1
        scatterlines!(ax, heights_obs[l], ϵ_obs[l])
    end

    if show == true
        display(GLMakie.Screen(), fig)
    end
    return fig, ax, heights_obs, ϵ_obs
end

function update_object_figure(object, reconstruction)
    figs = reconstruction.figures
    figs.object_obs[] = rotr90(object)
end

function update_layer_figure(opd, atmosphere, reconstruction)
    figs = reconstruction.figures
    for l=1:atmosphere.nlayers
        figs.opd_obs[l][] = rotr90(opd[:, :, l])
        reset_limits!(figs.opd_ax[l])
    end
end

function update_heights_figure(heights, ϵ, atmosphere, reconstruction)
    ϵ_real = cat(ϵ..., dims=1)
    ϵ_real = ϵ_real[.!(isnan.(ϵ_real))]
    base10exp = (length(ϵ_real)==0) ? 0 : floor(Int64, log10(minimum(ϵ_real)))
    ylabel = rich("Criterion [×10", superscript(string(base10exp)) , "]")
    figs = reconstruction.figures
    figs.heights_ax.ylabel = ylabel
        for l=1:atmosphere.nlayers-1
            figs.heights_obs[l][] = heights[l]
            figs.ϵ_obs[l][] = ϵ[l] ./ 10^base10exp
            reset_limits!(figs.heights_ax)
        end
end

function savefig(filename, fig, px_per_unit)
    save(filename, fig, px_per_unit=px_per_unit)
end
