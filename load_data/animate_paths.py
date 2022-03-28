import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_paths(paths, scale = 2, clip_tail = float('inf'), linewidth = 0.5, clear_plot = True):
    """
    Make an animation showing a path through 3D space.
    
    Parameters
    ----------
    paths: DataFrame
        should have columns x, y, z and a multiindex of (incident_id, milliseconds)
        
    scale_length: float, optional
        the axes will extend from -scale to scale in each dimension.
        
    clip_tail: int or inf, optional
        many steps of  a tail will be shown for each point. If inf, the entire tail will be shown.
        
    linewidth: float, optional
        the width of the tails.
        
    clear_plot: bool, optional
        if True, clear the plot once done. This suppresses the automatic show() call at the end of a cell.
        This will ahve to be False if you want to show it in a module.
    """
    t = len(paths.groupby('milliseconds'))
    paths = paths.to_numpy().reshape([-1,t,3])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect([1,1,1])
    lines = [
        ax.plot(
            [],
            [],
            [],
            linewidth=linewidth,
            marker="o",
            markevery=[-1],
        )[0]
        for __ in paths
    ]

    ax.set(xlim3d=(-scale, scale), xlabel="x")
    ax.set(ylim3d=(-scale, scale), ylabel="y")
    ax.set(zlim3d=(-scale, scale), zlabel="z")

    def update_lines(num):
        p = paths[:, max(num-clip_tail, 0):num, :]
        for line, path in zip(lines, p):
            line.set_data(path[:, :2].T)
            line.set_3d_properties(path[:, 2])
        return lines

    ani = animation.FuncAnimation(
        fig, update_lines, frames=len(paths[0]), interval=40
    )

    if clear_plot:
        plt.close()
    
    return ani