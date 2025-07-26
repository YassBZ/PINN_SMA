from matplotlib import rcParams

def plotParams():
    """
    A function that sets matplotlib parameters for generating fancy plots.
    :param: None
    :return: None
    """
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Latin Modern Roman'

    # axes and tickmarks
    rcParams['axes.labelsize'] = 15
    # rcParams['axes.labelweight']=600
    rcParams['axes.linewidth'] = 1.5

    rcParams['xtick.labelsize'] = 14
    rcParams['xtick.top'] = True
    rcParams['xtick.direction'] = 'in'
    rcParams['xtick.major.size'] = 6
    rcParams['xtick.minor.size'] = 3
    rcParams['xtick.major.width'] = 1.2
    rcParams['xtick.minor.width'] = 1.2
    rcParams['xtick.minor.visible'] = True

    rcParams['ytick.labelsize'] = 14
    rcParams['ytick.right'] = True
    rcParams['ytick.direction'] = 'in'
    rcParams['ytick.major.size'] = 6
    rcParams['ytick.minor.size'] = 3
    rcParams['ytick.major.width'] = 1.2
    rcParams['ytick.minor.width'] = 1.2
    rcParams['ytick.minor.visible'] = True

    # points, errorbars, and lines
    rcParams['lines.linewidth'] = 2.0
    rcParams['lines.markeredgewidth'] = 0.5
    rcParams['lines.markersize'] = 6
    rcParams['errorbar.capsize'] = 2