using GLMakie


function plot_object(object; write=false, filename="")
    fig, ax, plt = heatmap(rotr90(object), colormap=Reverse(:Greys), figure=(size=(800, 600),), axis=(title="Object Recovered (mean)", titlesize=23, aspect=DataAspect()))
    hidedecorations!(ax)
    Colorbar(fig[:, end+1], plt)
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    resize_to_layout!(fig)
    if write == true
        save(filename, fig, px_per_unit=16)
        println("Figure written to $(filename)")
    end
    return fig
end

function plot_recovered_layers(x, nlayers; write=false, filename="", label="Phase [rad]")
    fig = Figure(size=(800, 600))
    ax = Array{Any}(undef, nlayers)
    plt = Array{Any}(undef, nlayers)

    for l=1:nlayers
        ax[l] = Axis(fig[1, 2l-1], aspect=1)
        plt[l] = heatmap!(ax[l], rotr90(x[:, :, l]), colormap=:turbo)
        hidedecorations!(ax[l], label=false)
        ax[l].title = "Layer $(l)"
        ax[l].titlesize = 25
        colsize!(fig.layout, 2l-1, Aspect(1, 1.0))
        Colorbar(fig[1, 2l], plt[l], label=label, labelsize=23)
    end
    
    resize_to_layout!(fig)
    if write == true
        save(filename, fig, px_per_unit=2)
        println("Figure written to $(filename)")
    end
    return fig
end
