import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASS_COLORS = ['#DDBB44','#00AA00','#DD5500','#33CCCC','#000000']
CLASS_LABELS = ['subterran', 'terran', 'superterran', 'giant','no class']


nameslist1 = ['TRAPPIST-1 d', 'TRAPPIST-1 e', 'TRAPPIST-1 f', 'TRAPPIST-1 g',
            'Ross 128 b', 'Kepler-186 f', 'Kepler-440 b', 'Wolf 1061 c',
            'Proxima Cen b', 'TOI-700 d', 'TOI-700 e', 'GJ 3323 c', 'GJ 514 b',
            'Kepler-436 b', 'Kepler-283 c', 'Kepler-1229 b', 'K2-3 d',
            'Kepler-1086 c', 'Kepler-1318 b', 'Kepler-1090 b', 'Kepler-1690 b']

nameslist2 = ['TRAPPIST-1 d', 'TRAPPIST-1 e', 'TRAPPIST-1 f', 'TRAPPIST-1 g',
            'Ross 128 b', 'Kepler-186 e', 'Kepler-186 f', 'Kepler-296 e',
            'Kepler-440 b', 'Teegarden\'s Star b', 'Teegarden\'s Star c',
            'Proxima Centauri b', 'TOI-700 d', 'TOI-700 e', 'GJ 3323 b',
            'GJ 3323 c']

#customize the name data used in plot
def custom_name_data1(name_data: pd.DataFrame):
    # print(name_data)
    name_data.set_index('pl_name', inplace=True)
    name_data.at['K2-3 d','yoff'] = -0.2
    name_data.at['Kepler-440 b','yoff'] = 0.1
    name_data.at['Kepler-1229 b','xoff'] = -0.075
    name_data.at['Kepler-1229 b','yoff'] = 0.25
    name_data.at['Kepler-1690 b','yoff'] = -0.5
    name_data.at['GJ 3293 d','yoff'] = -0.2
    name_data.at['Wolf 1061 c','xoff'] = 0.01
    name_data.at['Wolf 1061 c','yoff'] = -0.1
    name_data.at['GJ 273 b','yoff'] = -0.25
    name_data.at['Ross 128 b','xoff'] = 0.01
    name_data.at['Ross 128 b','yoff'] = -0.25
    name_data.at['TRAPPIST-1 g','yoff'] = 0.25
    name_data.at['TRAPPIST-1 f','yoff'] = 0.05
    name_data.at['TRAPPIST-1 e','yoff'] = -0.05
    name_data.at['TRAPPIST-1 d','yoff'] = -0.25
    name_data.at['TOI-700 d','yoff'] = 0.15
    name_data.at['TOI-700 e','yoff'] = -0.05
    name_data.reset_index(inplace=True)

    return name_data

#customize the name data used in plot
def custom_name_data(name_data: pd.DataFrame):
    # print(name_data)
    name_data.set_index('pl_name', inplace=True)
    name_data.at['Kepler-62 e','xoff'] = -0.18
    name_data.at['Kepler-62 f','xoff'] = -0.18
    name_data.at['Kepler-440 b','yoff'] = 0.1
    name_data.at['GJ 3293 d','yoff'] = -0.2
    name_data.at['Kepler-296 e','xoff'] = -0.2
    name_data.at['Kepler-296 e','yoff'] = -0.35
    name_data.at['Wolf 1061 c','xoff'] = -0.2
    name_data.at['Wolf 1061 c','yoff'] = 0.02
    name_data.at['LHS 1140 b','yoff'] = 0.15
    name_data.at['GJ 273 b','yoff'] = -0.25
    name_data.at['Ross 128 b','xoff'] = 0.01
    name_data.at['Ross 128 b','yoff'] = -0.25
    name_data.at['TRAPPIST-1 g','yoff'] = 0.15
    name_data.at['TRAPPIST-1 f','yoff'] = 0.05
    name_data.at['TRAPPIST-1 e','yoff'] = -0.05
    name_data.at['TRAPPIST-1 d','yoff'] = -0.15
    name_data.reset_index(inplace=True)

    return name_data

def plot(alfven_data, condition, color_conditions, title, imgname, errorbars=False, name_data=None):
    # should figure out way to make more interactive (choose class, subset, etc)
    # must speed up data input
    # maybe add data series on top of each other?
    subset = alfven_data[condition] #i think this and the next line is where the slice/index warning comes up
    color_subsets = [subset[color_cond] for color_cond in color_conditions]
    # xlim = np.ceil(2 * np.max(subset.Ro)) / 2.
    # ylim = np.ceil(2 * np.max(subset.orbit2alfven)) / 2.

    # xlim = 4.0
    # ylim = 30.
    xlim = 2.0
    ylim = 12.

    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)

    ax.set_title(title,fontsize=25)
    ax.set_xlabel("Ro",fontsize=20)
    ax.set_ylabel("MHC",fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.grid(visible=True)
    ax.set_xlim([0.0,xlim])
    ax.set_ylim([0.0,ylim])
    ax.axhline(y = 1.0, xmin = 0.00, xmax = xlim, linestyle='--', color='#666666')
    ax.text(xlim + .02, 0.98, 'AS', fontsize=18)
    fig.set_facecolor('white')
    fig.tight_layout()

    # ax.errorbar(habitable_alfven_all['Ro'], habitable_alfven_all['orbit2alfven'], xerr=habitable_alfven_all['dRo'], yerr=habitable_alfven_all['dorbit2alfven'], linestyle='None', ecolor='#333333')

    for sub in color_subsets:
        class_idx = sub.mass_class.iat[0]
        ax.scatter(sub.Ro, sub.orbit2alfven, color=CLASS_COLORS[class_idx], s=50, label=CLASS_LABELS[class_idx], alpha=0.6)
        if errorbars:
            ax.errorbar(sub.Ro, sub.orbit2alfven, xerr=sub.dRo, yerr=sub.dorbit2alfven, linestyle='None', ecolor='#333333')

    # habitable = alfven_data[alfven_data.habitable == 1]
    # for row in habitable.itertuples():
    #     ax.scatter(row.Ro, row.orbit2alfven, color='blue',marker='*', s=125)

    # ax.scatter(row.Ro, row.orbit2alfven, color='blue',marker='*', s=125, label='Goldilocks')

    alfven_data.set_index('pl_name', inplace=True)
    # colorlist = []
    # for row in name_data.itertuples():
    #     name_data.at[row.pl_name, 'color'] = CLASS_COLORS[alfven_data.at[row.pl_name, 'mass_class']]

    # print(name_data.color)

    ax.text(0.01, 1.9, 'TRAPPIST-1', fontsize=20)
    if name_data is not None:
        for row in name_data.itertuples():
            color = CLASS_COLORS[alfven_data.at[row.pl_name, 'mass_class']]
            ax.scatter(row.Ro, row.orbit2alfven, color=color, s=125, alpha=1.0, marker='*')

            if row.pl_name.find('TRAPPIST-1') != -1:
                pl_name = row.pl_name[len('TRAPPIST-1 '):]
                ax.text(row.Ro + row.xoff, row.orbit2alfven + row.yoff, pl_name, fontsize=15)
                continue
            
            ax.text(row.Ro + row.xoff, row.orbit2alfven + row.yoff, row.pl_name, fontsize=20)


    # for idx, row in subset_rows:
        # xpt = row.Ro
        # ypt = row.orbit2alfven
        # ax.scatter(xpt, ypt, color=CLASS_COLORS[row.mass_class], s=75, label=(CLASS_LABELS[row.mass_class] if row.pl_name in labelled else ''))
        # if errorbars:
        #     ax.errorbar(xpt, ypt, xerr=row.dRo, yerr=row.dorbit2alfven, linestyle='None', ecolor='#333333')
        
        # if names:
        #     # ADD CALLOUTS/LINES FROM POINT TO NAME
        #     lbl = row.pl_name
        #     ax.text(xpt + 0.02, ypt + 0.01, lbl, fontsize=20)

    ro_sol = 1.85
    dro_sol = 0.26
    ra_sol = 0.1
    dra_sol = 0.02
    r_sol = 1.

    # ax.scatter(ro_sol, 0.718 / ra_sol, color='#bbb52a', marker='$*$', s=200)
    # ax.scatter(ro_sol, 1. / ra_sol, color='#14a97f', marker='$*$', s=200)
    # ax.scatter(ro_sol, 1.381 / ra_sol, color='#be1a09', marker='$*$', s=200)
    ax.scatter(ro_sol, 0.718 / ra_sol, color='black', marker='$*$', s=200)
    ax.scatter(ro_sol, 1. / ra_sol, color='black', marker='$*$', s=200)
    # ax.scatter(ro_sol, 1.381 / ra_sol, color='black', marker='$*$', s=200)

    ax.text(ro_sol + 0.02, 0.718 / ra_sol + 0.01, 'Venus',fontsize=20)
    ax.text(ro_sol + 0.02, 1.000 / ra_sol + 0.01, 'Earth',fontsize=20)
    # ax.text(ro_sol+0.02, 1.381 / ra_sol + 0.01, 'Mars',fontsize=20)

    # print(subset.pl_name)
    ax.legend(loc=2,fontsize=18)
    plt.savefig(imgname, format='png')
    # plt.ion()
    plt.show()

def main():

    alfven_data  = pd.read_csv('current-exo-data/alfven_data.csv')

    condition = alfven_data.habitable == 1
    color_conditions = [alfven_data.mass_class == i for i in range(0,4)]
    # color_conditions = [alfven_data.rad_class == i for i in range(0,4)]
    # name_conditions = (alfven_data.habitable == 1)
    name_conditions = alfven_data.pl_name.isin(nameslist1)
    name_data = alfven_data[name_conditions][['pl_name','Ro','orbit2alfven']]
    name_data['xoff'] = 0.02
    name_data['yoff'] = 0.01
    name_data = custom_name_data1(name_data)
    title = "Magnetic Habitability Criterion of Goldilocks Exoplanets\nfor various Stellar Activity Levels"
    imgname = 'current-exo-data/plot_habitable.png'
    plot(alfven_data, condition, color_conditions, title, imgname, errorbars=False, name_data=name_data)
    
    # condition = alfven_data.mass_class >= 0
    # color_conditions = reversed([alfven_data.mass_class == i for i in range(0,4)])
    # title = "Magnetic Habitability Criterion of All Exoplanets\nfor various Stellar Activity Levels"
    # imgname = 'current-exo-data/plot_all.png'
    # plot(alfven_data, condition, color_conditions, title, imgname, errorbars=False)

    # start event loop
    # while(1):
        #add data series, vars, and condns like FRED database
        # pass

if __name__ == "__main__":
    main()