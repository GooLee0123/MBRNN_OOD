import logging
import os
import csv

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from matplotlib.ticker import MaxNLocator, NullFormatter
from scipy.stats import gaussian_kde
from sklearn import linear_model

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging_args = {'filemode': 'w',
                'format': LOG_FORMAT,
                'level': getattr(logging, LOG_LEVEL)}
logging.basicConfig(**logging_args)

gamma = str(0.0).replace('.', '_')
train_op = 'RA_combined_usample'
test_op = 'RA_combined_usample'
color_scheme = ['red', 'orange', 'green', 'blue', 'indigo', 'black']
outdn = 'ComphsvPlot'
if not os.path.exists(outdn):
    os.makedirs(outdn)


def tick_setting(ax,
                 xlabel=None, ylabel=None,
                 xlim=None, ylim=None,
                 xticks=None, yticks=None):
    ax.set(xlabel=xlabel)
    ax.set(ylabel=ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.tick_params(axis='y', which='both', direction='out')
    ax.tick_params(axis='x', which='both', direction='out')


def kde_scipy(x, x_grid=None, set_bandwidth=None):
    kde = gaussian_kde(x)
    if set_bandwidth is not None:
        kde.set_bandwidth(bw_method=kde.factor/set_bandwidth)
    if x_grid is not None:
        return kde.evaluate(x_grid)
    return kde.evaluate(x)


def get_cls_metrics_eval_global():
    adirs = sorted([d for d in os.listdir('./') if 'Analysis' in d])
    qdirs = []
    for ad in adirs:
        qdirs.append(
            [os.path.join(ad, 'Quantities', d, 'cls_metric.csv')
             for d in os.listdir(os.path.join(ad, 'Quantities'))
             if 'DCPW1_0_RA_combined_usample' in d][0])

    loc = qdirs[:-1]
    glob = qdirs[-1]

    loc_metrics = \
        [pd.read_csv(ld, header=0, delimiter=',').values.ravel() for ld in loc]
    glob_metrics = pd.read_csv(glob, header=0, delimiter=',').values.ravel()

    return np.array(loc_metrics).T, np.array(glob_metrics), loc


def get_outputs():
    dn1 = 'Analysis_ind_galaxy_unsup_%s' % train_op
    dn2 = 'Quantities'
    dn3 = 'quantity_NC64_TwoFoldTrain_BatchOFF_Gamma%s_DCPW1_0_%s_%s'
    ft_dn = dn3 % (gamma, test_op, 'fine_tuned')
    pt_dn = dn3 % (gamma, test_op, 'post_tuned')
    ft_fn = os.path.join(dn1, dn2, ft_dn, 'zphot.npy')
    pt_fn = os.path.join(dn1, dn2, pt_dn, 'zphot.npy')

    logging.info("Load outputs of phase-2 model from %s" % ft_fn)
    logging.info("Load outputs of phase-3 model from %s" % pt_fn)

    ft_out = np.load(ft_fn, allow_pickle=True)
    pt_out = np.load(pt_fn, allow_pickle=True)

    dcp_lab = [ft_out[3], ft_out[4], ft_out[5]]  # id, lo, ul
    zphots = [pt_out[0], pt_out[1], pt_out[2]]  # id, lo, ul  ## univ, spec

    return dcp_lab, zphots


def unnorm(normed):
    data = pd.read_csv('./data_original/PS-DR1_galaxy.txt',
                       header=None, delimiter=' ').values
    base_col = data[:, 4:21].astype(np.float32)
    base_col[:, -1] = np.log(base_col[:, -1])

    col_min = np.min(base_col, 0)
    col_max = np.max(base_col, 0)

    unnormed = normed*(col_max-col_min)+col_min
    unnormed[:, -1] = np.exp(unnormed[:, -1])

    return unnormed


def get_unnorm(option):
    dn = './data_processed_ind_galaxy'
    if option == 'LOOD':
        fn = 'PS-DR1_labeled_ood_eval.npy'
        start = 1
    elif option == 'UL':
        fn = 'PS-DR1_unlabeled_eval_RA_combined_usample.npy'
        start = 0
    else:
        fn = 'PS-DR1_galaxy_eval.npy'
        start = 1

    normed_fn = os.path.join(dn, fn)
    logging.info("Load normed from %s and unnorm" % normed_fn)
    normed = np.load(normed_fn, allow_pickle=True)

    if option == 'LOOD':
        ids = normed[:, -1]
    else:
        ids = None

    normed_col = normed[:, start:-1]
    unnormed_col = unnorm(normed_col)

    if option == 'UL':
        return unnormed_col
    else:
        zspec = normed[:, 0]
        return zspec, unnormed_col, ids


def get_ft_zphots():
    dn1 = 'Analysis_ind_galaxy_unsup_%s' % train_op
    dn2 = 'Quantities'
    dn3 = 'quantity_NC64_TwoFoldTrain_BatchOFF_Gamma%s_DCPW1_0_%s_%s'
    ft_dn = dn3 % (gamma, test_op, 'fine_tuned')
    ft_fn = os.path.join(dn1, dn2, ft_dn, 'zphot.npy')

    logging.info("Load outputs of phase-2 model from %s" % ft_fn)

    ft_out = np.load(ft_fn, allow_pickle=True)

    return ft_out[0][:, 0], ft_out[0][:, 1]


def PPplot(zspec, zphot, dcp_lab, criteria,
           option='LOOD', sample=False):
    fn = os.path.join(outdn, "PPplot_%s" % option)

    dcp = dcp_lab[:, 0]

    if sample:
        rseed = 163
        np.random.seed(rseed)

        nsample = 30000
        rand_idx = np.random.choice(np.arange(len(zspec)),
                                    nsample, replace=False)

        zspec = zspec[rand_idx]
        zphot = zphot[rand_idx]
        dcp = dcp[rand_idx]

    if 'LOOD' in option:
        lim = [0, 5]
        ticks = [0, 1, 2, 3, 4, 5]
    else:
        lim = [0, 1.5]
        ticks = [0, 0.5, 1.0, 1.5]
        # ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    bounds = [dcp.min(), criteria[0], criteria[1], dcp.max()]

    # mask = dcp <= criteria[0]
    # low_dcp = dcp[mask]
    # low_dcp_zs = zspec[mask]
    # low_dcp_zp = zphot[mask]

    # mask1 = dcp > criteria[0]
    # mask2 = dcp <= criteria[1]
    # mask = mask1*mask2
    # mid_dcp = dcp[mask]
    # mid_dcp_zs = zspec[mask]
    # mid_dcp_zp = zphot[mask]

    # mask = dcp > criteria[1]
    # high_dcp = dcp[mask]
    # high_dcp_zs = zspec[mask]
    # high_dcp_zp = zphot[mask]

    abs_diff = np.abs(zspec-zphot)/(zspec+1)
    cat = abs_diff > 0.15

    cmap = matplotlib.colors.ListedColormap(['blue', 'green', 'red'])
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    cat_tol = 0.15
    x = np.arange(0, 6)
    corr = np.corrcoef(zphot, zspec)[0, 1]
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()

        ax.plot(x, x, color='k', linestyle='dashed')
        sp = ax.scatter(zspec, zphot, c=dcp,
                        marker='.',
                        s=0.5, cmap=cmap,
                        edgecolor=None,
                        linewidth=0.3,
                        norm=norm)

        # ax.text(0.25, 0.9, "CorrCoef: %.3f" % corr,
        #         ha='center', va='center',
        #         fontsize=8,
        #         transform=ax.transAxes)

        xx = np.arange(0, 10)

        ax.plot(xx, -cat_tol*(xx+1)+xx,
                color='k', ls='dashed', lw=0.5)
        ax.plot(xx, cat_tol*(xx+1)+xx,
                color='k', ls='dashed', lw=0.5)

        tick_setting(ax, r"$z_{spec}$", r"$z_{phot}$",
                     lim, lim, ticks, ticks)

        rect = [0, 0, 1, 1]
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position = ax.transAxes.transform(rect[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]
        sub_ax = plt.axes([x, y, width, height])
        ip = InsetPosition(ax, [0.5, 0.5, 0.5, 0.5])
        sub_ax.set_axes_locator(ip)
        sub_ax.scatter(zspec[cat], zphot[cat],
                       marker='.', c=dcp[cat],
                       s=3, cmap=cmap,
                       linewidth=0.3,
                       norm=norm,
                       edgecolor=None)

        sub_ax.plot(xx, color='k',
                    ls='dashed', lw=0.6)
        sub_ax.plot(xx, -cat_tol*(xx+1)+xx,
                    color='k', ls='dashed', lw=0.5)
        sub_ax.plot(xx, cat_tol*(xx+1)+xx,
                    color='k', ls='dashed', lw=0.5)
        sub_ax.set_yticks([])
        sub_ax.set_xticks([])
        sub_ax.set_xlim(0, 1)
        sub_ax.set_ylim(0, 1)

        fig.subplots_adjust(left=0.2,
                            right=0.8,
                            bottom=0.1,
                            top=0.9)
        cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])

        cbar = fig.colorbar(sp, cax=cbar_ax, ticks=bounds)
        cbar.set_label(r"$L_{DCP}$")
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)

        logging.info("PP-plot is saved at %s" % fn)


def ColRedScattergram(colors, redshift, dcp_lab, criteria,
                      phot_red_id=None, color_id=None, option="LO",
                      ylabel=r"$z_{spec}$", coln=0, sample=False):
    fn = os.path.join(outdn, "ColRedScattergram%s" % str(coln))
    fn += "_"+option

    if option == 'ID':
        phot_red_id = redshift
        color_id = colors

    dcp = dcp_lab[:, 0]
    if sample:
        rseed = 192
        np.random.seed(rseed)

        nsample = 200000
        rand_idx = np.random.choice(np.arange(len(redshift)),
                                    nsample, replace=False)

        dcp = dcp[rand_idx]
        colors = [color[rand_idx] for color in colors]
        redshift = redshift[rand_idx]

    bounds = [dcp.min(), criteria[0], criteria[1], dcp.max()]

    mask = dcp <= criteria[0]
    low_dcp = dcp[mask]
    low_dcp_col = [color[mask] for color in colors]
    low_dcp_red = redshift[mask]

    mask = dcp > criteria[1]
    high_dcp = dcp[mask]
    high_dcp_col = [color[mask] for color in colors]
    high_dcp_red = redshift[mask]

    cmap = matplotlib.colors.ListedColormap(['blue', 'green', 'red'])
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    ylim = [0, 1]
    xlims = [[-1, 2.5], [-1, 1.5], [-1, 1]]
    xticks = [[-1, -0.5, 0, 0.5, 1, 1.5, 2],
              [-1, -0.5, 0, 0.5, 1],
              [-1, -0.6, -0.2, 0.2, 0.6, 1]]
    ytick = [0, 0.2, 0.4, 0.6, 0.8, 1]
    xlabels = [r"$(g-r)$", r"$(r-i)$", r"$(i-z)$"]
    ylabels = [ylabel, None, None]
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, axes = plt.subplots(1, len(colors), figsize=(9, 3))

        for i, color in enumerate(colors):
            logging.info("Compute density of col-spec/phot redshift")
            xx = np.linspace(color_id[i].min(), color_id[i].max(), 100)
            zz = np.linspace(0, 1, 100)
            XX, ZZ = np.meshgrid(xx, zz)
            positions = np.vstack([XX.ravel(), ZZ.ravel()])

            xz = np.vstack([color_id[i], phot_red_id])
            randidx = np.random.choice(np.arange(xz.shape[1]), 20000)
            xz = xz[:, randidx]

            density = np.reshape(kde_scipy(xz, positions).T, XX.shape)

            sp = axes[i].scatter(color, redshift, c=dcp,
                                 marker='.',
                                 s=0.5, cmap=cmap,
                                 linewidth=0.3,
                                 edgecolor=None,
                                 norm=norm)
            if option != 'ID':
                axes[i].scatter(low_dcp_col[i], low_dcp_red,
                                c=low_dcp, marker='.',
                                s=0.5, cmap=cmap,
                                linewidth=0.3,
                                edgecolor=None,
                                norm=norm)
            else:
                tot_nsamp = len(high_dcp_col[i])
                und_samp = int(tot_nsamp/1.)
                high_rand_idx = np.random.choice(np.arange(tot_nsamp),
                                                 und_samp)
                axes[i].scatter(high_dcp_col[i][high_rand_idx],
                                high_dcp_red[high_rand_idx],
                                c=high_dcp[high_rand_idx], marker='.',
                                s=1, cmap=cmap,
                                linewidth=0.3,
                                edgecolor=None,
                                norm=norm)

            axes[i].contour(XX, ZZ, density, np.linspace(density.min() +
                            (density.max()-density.min())/10., density.max(), 8),
                            colors='deepskyblue', linewidths=0.75)

            tick_setting(axes[i], xlabels[i], ylabels[i],
                         xlims[i], ylim, xticks[i], ytick)
            if i != 0:
                axes[i].set_yticks([])

        fig.subplots_adjust(left=0.2,
                            right=0.8,
                            bottom=0.1,
                            top=0.9,
                            wspace=0.05,
                            hspace=0.3)
        cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sp, cax=cbar_ax, ticks=bounds)
        cbar.set_label(r"$L_{DCP}$")
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("Col-Red Scattergram is saved at %s" % fn)


def ColColScattergram(cols, dcps, cols_id, idx1, idx2, criteria,
                      xlim=[None, None], ylim=[None, None],
                      xlabel=[None, None], ylabel=[None, None],
                      xticks=[None, None], yticks=[None, None],
                      sample=None, dtype="QSO"):
    fn = os.path.join(outdn, "ColColScattergram_%s" % dtype)

    # if dtype == 'QSO':
    xticks[0] = [-0.5, 0, 0.5, 1, 1.5]
    xticks[1] = [-0.5, 0, 0.5, 1, 1.5, 2]
    xids = [idx1[0], idx2[0]]
    yids = [idx1[1], idx2[1]]
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        for i in range(2):
            ax = axes[i]

            dcp = dcps

            xcol = cols[:, xids[i]]
            ycol = cols[:, yids[i]]

            xcol_id = cols_id[:, xids[i]]
            ycol_id = cols_id[:, yids[i]]

            logging.info("Compute density of col-col")

            xx = np.linspace(xcol_id.min(), xcol_id.max(), 100)
            yy = np.linspace(ycol_id.min(), ycol_id.max(), 100)
            XX, YY = np.meshgrid(xx, yy)
            positions = np.vstack([XX.ravel(), YY.ravel()])

            xy = np.vstack([xcol_id, ycol_id])
            np.random.seed(100)
            randidx = np.random.choice(np.arange(xy.shape[1]), 50000)
            xy = xy[:, randidx]

            density = np.reshape(kde_scipy(xy, positions).T, XX.shape)

            if sample is not None:
                logging.info("Random under sampling")
                rseed = 932
                np.random.seed(rseed)

                rand_idx = np.random.choice(np.arange(len(xcol)),
                                            sample, replace=False)

                dcp = dcp[rand_idx]
                xcol = xcol[rand_idx]
                ycol = ycol[rand_idx]

            bounds = [dcp.min(), criteria[0], criteria[1], dcp.max()]

            mask = dcp <= criteria[0]
            low_dcp = dcp[mask]
            low_dcp_xcol = xcol[mask]
            low_dcp_ycol = ycol[mask]

            mask1 = dcp > criteria[0]
            mask2 = dcp <= criteria[1]
            mask = mask1*mask2
            mid_dcp = dcp[mask]
            mid_dcp_xcol = xcol[mask]
            mid_dcp_ycol = ycol[mask]

            cmap = matplotlib.colors.ListedColormap(['blue', 'green', 'red'])
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

            sp = ax.scatter(xcol, ycol, c=dcp, marker='.',
                            s=0.5, linewidth=0.3,
                            cmap=cmap, norm=norm,
                            edgecolor=None)

            if dtype != 'Star':
                tot_nsamp = len(mid_dcp_xcol)
                und_samp = int(tot_nsamp)
                mid_rand_idx = np.random.choice(np.arange(tot_nsamp), und_samp)
                ax.scatter(mid_dcp_xcol[mid_rand_idx],
                           mid_dcp_ycol[mid_rand_idx],
                           c=mid_dcp[mid_rand_idx], marker='.',
                           s=0.5, cmap=cmap, linewidth=0.3,
                           edgecolor=None, norm=norm)

                tot_nsamp = len(low_dcp_xcol)
                und_samp = int(tot_nsamp)
                low_rand_idx = np.random.choice(np.arange(tot_nsamp), und_samp)
                ax.scatter(low_dcp_xcol[low_rand_idx],
                           low_dcp_ycol[low_rand_idx],
                           c=low_dcp[low_rand_idx], marker='.',
                           s=0.5, cmap=cmap, linewidth=0.3,
                           edgecolor=None, norm=norm)

            if xlim is None:
                xlim = [xcol.min(), xcol.max()]
            if ylim is None:
                ylim = [ycol.min(), ycol.max()]

            tick_setting(ax, xlabel[i], ylabel[i],
                         xlim[i], ylim[i], xticks[i], yticks[i])
            if i != 0:
                ax.set_yticks([])
            ax.contour(XX, YY, density, np.linspace(density.min() +
                       (density.max()-density.min())/4., density.max(), 8),
                       colors='k', linewidths=1)

        fig.subplots_adjust(left=0.2,
                            right=0.8,
                            bottom=0.1,
                            top=0.9,
                            wspace=0.05)
        cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])

        cbar = fig.colorbar(sp, cax=cbar_ax, ticks=bounds)
        cbar.set_label(r"$L_{DCP}$")
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("Col-Col Scattergram is saved at %s" % fn)


def PlotDistribution(zspec_id, zspec_lo):
    fn = os.path.join(outdn, "RedDistribution")

    star_mask = zspec_lo < 0.05
    qso_mask = np.logical_not(star_mask)
    zspec_star = zspec_lo[star_mask]
    zspec_qso = zspec_lo[qso_mask]
    zgrid = np.linspace(0, max(zspec_id.max(), zspec_lo.max()), 1000)
    zid_pdf = kde_scipy(zspec_id, zgrid)

    zstar_pdf = kde_scipy(zspec_star, zgrid)
    zqso_pdf = kde_scipy(zspec_qso, zgrid)

    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()

        ax.plot(zgrid, zid_pdf,
                color=color_scheme[0],
                linestyle='solid',
                label=r"$ID$")
        # ax.plot(zgrid, zstar_pdf,
        #         color=color_scheme[1],
        #         linestyle='solid',
        #         label=r"$LOOD_{star}$")
        ax.plot(zgrid, zqso_pdf,
                color=color_scheme[2],
                linestyle='solid',
                label=r"$LOOD_{qso}$")

        ax.legend()
        tick_setting(ax, "Redshift", "Number denisty",
                     None, None, None, None)

        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("Redshift distribution is saved at %s" % fn)


def OODScoreCatNoncat(zspec, zphot, dcp_lab, criterion,
                      option='LO', sample=False):
    fn = os.path.join(outdn, "OODscore_cat_noncat_%s" % option)

    ood_score = dcp_lab[:, 0]

    if sample:
        rseed = 163
        np.random.seed(rseed)

        nsample = 50000
        rand_idx = np.random.choice(np.arange(len(zspec)),
                                    nsample, replace=False)

        zspec = zspec[rand_idx]
        zphot = zphot[rand_idx]
        ood_score = ood_score[rand_idx]

    residual = np.abs(zspec - zphot)

    xbins = np.linspace(ood_score.min(), ood_score.max(), 51)

    rel_resid = residual/(zspec+1)
    cat = rel_resid > 0.15
    noncat = np.logical_not(cat)

    cat_resid_samples = ood_score[cat]
    noncat_resid_samples = ood_score[noncat]

    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()

        ax.hist(cat_resid_samples, xbins,
                alpha=0.5, density=True,
                color='orange', label="cat")
        ax.hist(noncat_resid_samples, xbins,
                alpha=0.5, density=True,
                color='green', label="non-cat")

        ax.vlines(criterion, 1e-2, 100,
                  color='k', linestyle='dashed')

        ax.set_yscale('log')
        tick_setting(ax, r"$L_{DCP}$", "Number density",
                     None, [1e-2, 10],
                     [0, 1, 2, 3, 4], [1e-2, 1e-1, 1, 10])
        ax.legend()

        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("OOD score and redshift related plot is saved at %s" % fn)


def ContourWithHist(zspecs, zphots, dcp_labs, lo_labels,
                    sample=False):
    fn = os.path.join(outdn, "ContourWithHist")

    qso_mask = lo_labels == 1
    star_mask = np.logical_not(qso_mask)

    zspecs_qso = zspecs[1][qso_mask]
    zspecs_star = zspecs[1][star_mask]

    zphots_qso = zphots[1][qso_mask]
    zphots_star = zphots[1][star_mask]

    zspecs = [zspecs[0], zspecs_qso, zspecs_star]
    zphots = [zphots[0], zphots_qso, zphots_star]

    dcp_id = dcp_labs[0][:, 0]
    dcp_lo = dcp_labs[1][:, 0]

    dcp_qso = dcp_lo[qso_mask]
    dcp_star = dcp_lo[star_mask]

    dcps = [dcp_id, dcp_qso, dcp_star]

    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.02

    # Set up the geometry of the three plots
    rect_contour = [left, bottom, width, height]  # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25]  # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height]  # dimensions of y-histogram

    ymin = 0
    ymax = max([t.max() for t in dcps])
    ybins = np.linspace(ymin, ymax, 51)
    xbins = np.linspace(0, 3, 51)
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure()

        axContour = plt.axes(rect_contour)  # contour plot
        axHistx = plt.axes(rect_histx)  # x histogram
        axHisty = plt.axes(rect_histy)  # y histogram

        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        labels = [r"$ID$", r"$LOOD_{qso}$", r"$LOOD_{star}$"]
        for i in range(3):
            temp_zspec = zspecs[i]
            temp_zphot = zphots[i]
            temp_dcp = dcps[i]

            if sample:
                nsample = 100000
                rand_idx = np.random.choice(np.arange(len(temp_zspec)), nsample)
                temp_zspec = temp_zspec[rand_idx]
                temp_zphot = temp_zphot[rand_idx]
                temp_dcp = temp_dcp[rand_idx]

            temp_abs_diff = np.abs(temp_zspec-temp_zphot)

            xx = np.linspace(temp_abs_diff.min(), temp_abs_diff.max(), 100)
            yy = np.linspace(temp_dcp.min(), temp_dcp.max(), 100)
            XX, YY = np.meshgrid(xx, yy)
            positions = np.vstack([XX.ravel(), YY.ravel()])

            xy = np.vstack([temp_abs_diff, temp_dcp])
            randidx = np.random.choice(np.arange(xy.shape[1]), 20000)
            xy = xy[:, randidx]

            density = np.reshape(kde_scipy(xy, positions).T, XX.shape)

            axContour.contour(XX, YY, density, np.linspace(density.min() +
                              (density.max()-density.min())/10., density.max(), 8),
                              colors=color_scheme[i], linestyles='solid',
                              linewidths=0.5)
            axContour.set_ylabel(r"$L_{dcp}$")
            axContour.set_xlabel(r"Redshift absolute difference")
            axContour.set_xlim(0, 3)
            axContour.set_ylim(0, ymax)

            axHistx.hist(temp_abs_diff, xbins,
                         color=color_scheme[i],
                         alpha=0.5, density=True)
            # axHistx.set_yscale('log')
            axHistx.set_xlim(0, 3)
            axHisty.hist(temp_dcp, ybins,
                         color=color_scheme[i],
                         orientation='horizontal',
                         alpha=0.5, density=True)
            # axHisty.set_xscale('log')
            axHisty.set_ylim(0, ymax)

        handle1 = matplotlib.lines.Line2D([], [], c=color_scheme[0])
        handle2 = matplotlib.lines.Line2D([], [], c=color_scheme[1])
        handle3 = matplotlib.lines.Line2D([], [], c=color_scheme[2])
        axContour.legend(handles=[handle1, handle2, handle3],
                         labels=labels, loc="lower right")
        fig.savefig(fn)
        logging.info("Joint plot is saved at %s" % fn)


def HeatmapWithHist(zspec, zphot, dcp_lab, sample=False):
    fn = os.path.join(outdn, "HeatmapWithHist")

    dcp = dcp_lab[:, 0]

    if sample:
        nsample = 100000
        rand_idx = np.random.choice(np.arange(len(zspec)), nsample)
        zspec = zspec[rand_idx]
        zphot = zphot[rand_idx]
        dcp = dcp[rand_idx]

    abs_diff = np.abs(zspec-zphot)

    xx = np.linspace(abs_diff.min(), abs_diff.max(), 100)
    yy = np.linspace(dcp.min(), dcp.max(), 100)
    XX, YY = np.meshgrid(xx, yy)
    positions = np.vstack([XX.ravel(), YY.ravel()])

    xy = np.vstack([abs_diff, dcp])
    randidx = np.random.choice(np.arange(xy.shape[1]), 20000)
    xy = xy[:, randidx]

    density = np.reshape(kde_scipy(xy, positions).T, XX.shape)

    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.02

    # Set up the geometry of the three plots
    rect_heatmap = [left, bottom, width, height]  # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25]  # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height]  # dimensions of y-histogram

    nxbins = 31
    nybins = 31

    xmin, xmax = 0, 0.2
    ymin, ymax = 0, 0.2

    xbins = np.linspace(xmin, xmax, nxbins)
    ybins = np.linspace(ymin, ymax, nybins)

    aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure()

        axHeatmap = plt.axes(rect_heatmap)  # heatmap
        axHistx = plt.axes(rect_histx)  # x histogram
        axHisty = plt.axes(rect_histy)  # y histogram

        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        H, xedges, yedges = np.histogram2d(dcp, abs_diff, bins=(ybins, xbins))

        axHeatmap.imshow(H, extent=[xmin, xmax, ymin, ymax],
                         interpolation='nearest', origin='lower',
                         aspect=aspectratio, norm=matplotlib.colors.LogNorm())

        # axHeatmap.contour(XX, YY, density, np.linspace(density.min() +
        #                       (density.max()-density.min())/10., density.max(), 8),
        #                       colors=color_scheme[0], linestyles='solid',
        #                       linewidths=0.5)

        axHeatmap.set_ylabel(r"$L_{dcp}$")
        axHeatmap.set_xlabel(r"Redshift absolute difference")
        axHeatmap.set_xlim(0, xmax)
        axHeatmap.set_ylim(0, ymax)

        axHistx.hist(abs_diff, xbins,
                     color=color_scheme[0],
                     alpha=0.5, density=True)
        axHistx.set_xlim(0, xmax)

        axHisty.hist(dcp, ybins,
                     color=color_scheme[0],
                     orientation='horizontal',
                     alpha=0.5, density=True)
        axHisty.set_ylim(0, ymax)

        fig.savefig(fn)
        logging.info("Heatmap with 1-d histograms is saved at %s" % fn)


def get_probs(option='ID', avg=True):
    dn = './Analysis_ind_galaxy_unsup_RA_combined_usample/Quantities/quantity_NC64_TwoFoldTrain_BatchOFF_Gamma0_0_DCPW1_0_RA_combined_usample_post_tuned/'
    fn = os.path.join(dn, 'probs_%s.npy' % option.lower())

    probs = np.load(fn)

    if avg:
        return (probs[0]+probs[1])/2.
    return probs


def plot_probs_wrt_cat(zspec, zphot, probs, option='ID'):
    fn = os.path.join(outdn, "Probs_wrt_cat_%s" % option)

    cat_tol = 0.15
    rel_residual = np.abs(zspec - zphot)/(zspec+1)
    cat = rel_residual > cat_tol
    noncat = np.logical_not(cat)

    cat_probs = [np.mean(probs[0][cat], 0), np.mean(probs[1][cat], 0)]
    noncat_probs = [np.mean(probs[0][noncat], 0), np.mean(probs[1][noncat], 0)]

    bin_edges = np.genfromtxt('./bin_edges/galaxy_redshifts_64-uniform.txt')
    bin_center = [(bin_edges[i]+bin_edges[i+1]) for i in range(len(bin_edges)-1)]
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()

        ax.plot(bin_center, cat_probs[0],
                color='r', label="cat_{HE}",
                linestyle='solid')
        ax.plot(bin_center, cat_probs[1],
                color='r', label="cat_{LE}",
                linestyle='dashed')

        ax.plot(bin_center, noncat_probs[0],
                color='b', label="non-cat_{HE}",
                linestyle='solid')
        ax.plot(bin_center, noncat_probs[1],
                color='b', label="non-cat_{LE}",
                linestyle='dashed')

        ax.legend()
        tick_setting(ax, "Redshift", "Probability",
                     [0, 2], [0, 1],
                     [0, 0.4, 0.8, 1.2, 1.6, 2],
                     [0, 0.2, 0.4, 0.6, 0.8, 1])
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("Probability w.r.t residual is saved at %s" % fn)


def plot_avg_mode_diff_wrt_dcp(dcp_lab, probs, zphot, option='ID'):
    fn = os.path.join(outdn, "Zdiff_wrt_dcp_%s" % option)

    dcp = dcp_lab[:, 0]

    dcp_int = dcp.max() - dcp.min()
    bounds = [dcp.min(),
              dcp.min() + dcp_int/3.,
              dcp.min() + dcp_int/3.*2.,
              dcp.max()]

    masks = []
    for i in range(len(bounds)-1):
        mask1 = dcp > bounds[i]
        mask2 = dcp <= bounds[i+1]
        mask = mask1*mask2
        masks.append(mask)

    bin_edges = np.genfromtxt('./bin_edges/galaxy_redshifts_64-uniform.txt')
    bin_center = np.array([(bin_edges[i]+bin_edges[i+1]) for i in range(len(bin_edges)-1)])

    arg_max = np.argmax(probs, 1)
    zmode = bin_center[arg_max]

    zdiff = np.abs(zphot - zmode)

    low_dcp_zdiff = zdiff[masks[0]]
    mid_dcp_zdiff = zdiff[masks[1]]
    hgh_dcp_zdiff = zdiff[masks[2]]

    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()

        ax.hist(hgh_dcp_zdiff, bin_edges,
                color='r', label=r"$High_{dcp}$",
                alpha=0.5, density=True)
        ax.hist(mid_dcp_zdiff, bin_edges,
                color='g', label=r"$Mid_{dcp}$",
                alpha=0.5, density=True)
        ax.hist(low_dcp_zdiff, bin_edges,
                color='b', label=r"$Low_{dcp}$",
                alpha=0.5, density=True)

        ax.legend()
        tick_setting(ax, r"$|z_{avg}-z_{mode}|$", "Number density",
                     [0, 2], None,
                     [0, 0.4, 0.8, 1.2, 1.6, 2],
                     None)
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("Zdiff w.r.t dcp is saved at %s" % fn)


def plot_avg_mode_diff_wrt_resid(probs, zspec, zphot, option='ID'):
    fn = os.path.join(outdn, "Zdiff_wrt_AD_%s" % option)

    residual = np.abs(zspec - zphot)/(zspec+1)
    sresid = np.sort(residual)
    bounds = [residual.min(),
              np.percentile(sresid, 33.333),
              np.percentile(sresid, 66.666),
              residual.max()]

    masks = []
    for i in range(len(bounds)-1):
        mask1 = residual > bounds[i]
        mask2 = residual <= bounds[i+1]
        mask = mask1*mask2
        masks.append(mask)

    bin_edges = np.genfromtxt('./bin_edges/galaxy_redshifts_64-uniform.txt')
    bin_center = np.array([(bin_edges[i]+bin_edges[i+1]) for i in range(len(bin_edges)-1)])

    arg_max = np.argmax(probs, 1)
    zmode = bin_center[arg_max]

    zdiff = np.abs(zphot - zmode)

    low_resid_zdiff = zdiff[masks[0]]
    mid_resid_zdiff = zdiff[masks[1]]
    hgh_resid_zdiff = zdiff[masks[2]]

    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()

        ax.hist(hgh_resid_zdiff, bin_edges,
                color='r', label=r"$High_{AD}$",
                alpha=0.5, density=True)
        ax.hist(mid_resid_zdiff, bin_edges,
                color='g', label=r"$Mid_{AD}$",
                alpha=0.5, density=True)
        ax.hist(low_resid_zdiff, bin_edges,
                color='b', label=r"$Low_{AD}$",
                alpha=0.5, density=True)

        ax.legend()
        tick_setting(ax, r"$|z_{avg}-z_{mode}|$", "Number density",
                     [0, 2], None,
                     [0, 0.4, 0.8, 1.2, 1.6, 2],
                     None)
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("Zdiff w.r.t resid is saved at %s" % fn)


def plot_dcp_confidence(probs, dcps, dcpmm, criteria, sample=True):
    fn = os.path.join(outdn, "dcp_confidence")

    if sample:
        nsample = 50000
        for i in range(3):
            rand_idx = np.random.choice(np.arange(len(dcps[i])), nsample)
            probs[i] = probs[i][rand_idx]
            dcps[i] = dcps[i][rand_idx]

    confs = [np.max(p, 1) for p in probs]

    xx = np.linspace(0, dcpmm[1], 100)
    zz = np.linspace(0, 1, 100)
    XX, ZZ = np.meshgrid(xx, zz)
    positions = np.vstack([XX.ravel(), ZZ.ravel()])

    xz = np.vstack([dcps[0], confs[0]])
    density = np.reshape(kde_scipy(xz, positions).T, XX.shape)

    # level = 20
    # level = np.linspace(density.min(), density.max(), 10)
    # level = np.hstack([np.array([-1000, -100, -30, -20, -10]), level])
    level = np.arange(1000000, 20000000, 5000000)*density.min()
    level = np.hstack([level, np.arange(density.min()*20000000, density.max(), density.min()*30000000)])

    dcp_conf_qso = np.vstack([dcps[1], confs[1]])
    dcp_conf_star = np.vstack([dcps[2], confs[2]])
    color_qso = kde_scipy(dcp_conf_qso)
    color_star = kde_scipy(dcp_conf_star)

    titles = [r"$ID_{galaxy}$", r"$OOD_{QSO}$", r"$OOD_{star}$"]
    colors = ['b', color_qso, color_star]
    # colors = ['b', 'r', 'r']
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        for i in range(3):
            if i == 0:
                axes[i].scatter(dcps[i], confs[i],
                                color=colors[i], s=0.5,
                                linewidth=0, edgecolor=None)
            else:
                axes[i].scatter(dcps[i], confs[i],
                                c=colors[i], s=0.5,
                                linewidth=0, edgecolor=None)
            axes[i].contour(XX, ZZ, density, level,
                            colors='deepskyblue', linewidths=1.5)
            if i == 0:
                tick_setting(axes[i], r"$L_{DCP}$", "Confidence",
                             [0, dcpmm[1]], [0, 1], None, None)
            else:
                axes[i].vlines(criteria, 0, 1,
                               color='k',
                               linestyle='dashed',)
                tick_setting(axes[i], r"$L_{DCP}$", None,
                             [0, dcpmm[1]], [0, 1], None, None)
                axes[i].set_yticks([])
            axes[i].set_title(titles[i])

        fig.subplots_adjust(wspace=0.05)
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("DCP and confidence scattergram is saved at %s" % fn)


def DiscriminatorNumberDensity(dcp_id, dcp_lo, dcp_ul, criteria):
    fn = os.path.join(outdn, 'DCPNumberDensity')

    dcps = [dcp_id[:, 0], dcp_lo[:, 0], dcp_ul[:, 0]]
    min_dscrm = min([dcp.min() for dcp in dcps])
    max_dscrm = max([dcp.max() for dcp in dcps])

    # This is an approximation, but sufficiently close.
    bins = np.linspace(min_dscrm, max_dscrm, 101)

    colors = ['b', 'r', 'g']
    labels = ["ID", "LOOD", "UL"]
    with plt.style.context(['science', 'ieee', 'high-vis', 'grid', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()
        for i, d in enumerate(dcps):
            ax.hist(d, bins,
                    histtype='step',
                    density=True,
                    color=colors[i],
                    label=labels[i])

        # ax.vlines(criteria, 0, 20,
        #           color='k', linestyle='dashed')
        ax.legend(loc='upper center')
        tick_setting(ax,
                     r"$L_{DCP}$",
                     "Number density",
                     None,
                     [0, 13],
                     [0, 1, 2, 3, 4],
                     [0, 2, 4, 6, 8, 10, 12])
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
    logging.info("DCP-loss number density distribution is saved at %s" % fn)


def PSCPDF():
    def _psc_filter(dcp, psc):
        psc_mask = np.where(~np.isnan(psc))[0]

        return dcp[psc_mask], psc[psc_mask]

    sn = os.path.join(outdn, 'PSCPDF')

    fn1 = 'Analysis_ind_galaxy_unsup_RA_combined_usample/Quantities/quantity_NC64_TwoFoldTrain_BatchOFF_Gamma0_0_DCPW1_0_RA_0_10_fine_tuned/zphot.npy'
    fn2 = fn1.replace('RA_0_10', 'RA_100_110')
    fn3 = fn1.replace('RA_0_10', 'RA_170_180')
    fns = [fn1, fn2, fn3]

    psc_fn1 = './data_processed_ind_galaxy/PSC_eval_RA_0_10.npy'
    psc_fn2 = psc_fn1.replace('RA_0_10', 'RA_100_110')
    psc_fn3 = psc_fn1.replace('RA_0_10', 'RA_170_180')
    pfns = [psc_fn1, psc_fn2, psc_fn3]

    dcp = np.hstack([np.load(fn, allow_pickle=True)[-1].T[0] for fn in fns])
    psc = np.hstack([np.load(pfn).ravel() for pfn in pfns])

    dcp_ul, psc = _psc_filter(dcp, psc)

    dcp_bins = np.linspace(dcp_ul.min(), dcp_ul.max(), 7)

    beps = 1e-5
    lws = [1, 1, 1]
    psc_bin = np.array([0.0, 0.1, 0.8, 1.0])
    psc_bin[-1] += beps
    bin_idx = np.digitize(psc, psc_bin)-1
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex', 'grid']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()
        dcp_grid = np.linspace(dcp_ul.min(), dcp_ul.max(), 1000)
        for i in range(len(psc_bin)-1):
            mask = i == bin_idx
            if i != len(psc_bin)-2:
                label = r"ps-score$ \, \in \, [%.1f, %.1f)$" % \
                        (psc_bin[i], psc_bin[i+1])
            else:
                label = r"ps-score$ \, \in \, [%.1f, %.1f]$" % \
                        (psc_bin[i], psc_bin[i+1])
            if np.sum(mask) > 0:
                # temp_pdf = kde_scipy(dcp_ul[mask], dcp_grid)
                # grid_width = dcp_grid[1]-dcp_grid[0]
                # temp_cdf = np.cumsum(temp_pdf)*grid_width
                # ax.plot(dcp_grid, temp_pdf,
                #         color=color_scheme[i],
                #         linestyle='solid',
                #         linewidth=lws[i],
                #         label=label)
                ax.hist(dcp_ul[mask], dcp_bins,
                        color=color_scheme[i],
                        histtype='step',
                        label=label, density=True)
        # plt.ylim(1e-3, 10)
        # plt.yscale('log')
        ax.legend(loc='upper left')
        # ax.set_xscale('log')
        tick_setting(ax, r"$L_{DCP}$", "Number density",
                     None, [0, 1.5], [0, 1, 2, 3, 4],
                     [0, 0.3, 0.6, 0.9, 1.2, 1.5])

        fig.savefig(sn)
        fig.savefig(sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("PSC-DCP pdf plot is saved at %s" % sn)


def ID_DCP_NumberDensity(zspec, zphot, dcp_id, dcpmm, criteria):
    fn = os.path.join(outdn, 'IDDCPNumberDensity')

    cat_tol = 0.15

    rel_resid = np.abs(zspec-zphot)/(zspec+1)
    cat = rel_resid > cat_tol
    noncat = np.logical_not(cat)

    cat_dcp = dcp_id[cat]
    noncat_dcp = dcp_id[noncat]

    # This is an approximation, but sufficiently close.
    bins = np.linspace(dcpmm[0], dcpmm[1], 51)

    dcps = [cat_dcp, noncat_dcp]
    colors = ['blue', 'darkturquoise']
    # colors = ['purple', 'blue']
    labels = ["non-cat", "cat"]
    with plt.style.context(['science', 'ieee', 'high-vis', 'grid', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()
        for i, d in enumerate(dcps[::-1]):
            ax.hist(d, bins,
                    histtype='step',
                    density=True,
                    color=colors[i],
                    label=labels[i],
                    linestyle='solid')

        ax.vlines(criteria, 0, 20,
                  color='k', linestyle='dashed')
        ax.legend(loc='upper right')
        ax.set_yscale('log')
        tick_setting(ax,
                     r"$L_{DCP}$",
                     "Number density",
                     None,
                     [None, 10],
                     [0, 1, 2, 3, 4],
                     None)
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
    logging.info("OOD-DCP number density distribution is saved at %s" % fn)


def OOD_DCP_NumberDensity(dcp_qso, dcp_star, dcpmm, criteria):
    fn = os.path.join(outdn, 'OODDCPNumberDensity')

    # This is an approximation, but sufficiently close.
    bins = np.linspace(dcpmm[0], dcpmm[1], 51)

    dcps = [dcp_qso, dcp_star]
    colors = ['coral', 'red']
    labels = ["QSO", "Star"]
    with plt.style.context(['science', 'ieee', 'high-vis', 'grid', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()
        for i, d in enumerate(dcps):
            ax.hist(d, bins,
                    histtype='step',
                    density=True,
                    color=colors[i],
                    label=labels[i],
                    linestyle='solid')

        ax.vlines(criteria, 0, 20,
                  color='k', linestyle='dashed')
        ax.legend(loc='upper left')
        ax.set_yscale('log')
        tick_setting(ax,
                     r"$L_{DCP}$",
                     "Number density",
                     None,
                     [None, 10],
                     [0, 1, 2, 3, 4],
                     None)
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
    logging.info("OOD-DCP number density distribution is saved at %s" % fn)


def plot_dcp_dispersion(zspec, zphot, dcpmm, probs, dcp_lab, option='ID'):
    fn = os.path.join(outdn, "dcp_variance%s" % option)

    dcp = dcp_lab[:, 0]

    cat_tol = 0.15

    rel_resid = np.abs(zspec-zphot)/(zspec+1)
    cat = rel_resid > cat_tol
    noncat = np.logical_not(cat)

    bin_edges = np.genfromtxt('./bin_edges/galaxy_redshifts_64-uniform.txt')
    bin_cents = np.array([(bin_edges[i]+bin_edges[2])/2. for i in range(len(bin_edges)-1)])

    variance = np.sum((bin_cents-zphot.reshape(-1, 1))**2. * probs, 1)
    cat_var = variance[cat]
    noncat_var = variance[noncat]

    cat_dcp = dcp[cat]
    noncat_dcp = dcp[noncat]

    dcps = [noncat_dcp, cat_dcp]
    variances = [noncat_var, cat_var]

    densities = []
    xx = np.linspace(dcpmm[0], dcpmm[1], 100)
    zz = np.linspace(0, 1, 100)
    XX, ZZ = np.meshgrid(xx, zz)
    positions = np.vstack([XX.ravel(), ZZ.ravel()])
    for i in range(2):
        xz = np.vstack([dcps[i], variances[i]])
        if len(xz) > 50000:
            randidx = np.random.choice(np.arange(xz.shape[1]), 50000)
            xz = xz[:, randidx]
        density = np.reshape(kde_scipy(xz, positions).T, XX.shape)
        densities.append(density)

    if option == 'ID':
        orders, colors, labels = [0, 1], ['b', 'r'], ['noncat', 'cat']

        level1 = np.linspace(densities[0].min(), densities[0].max(), 20)
        level1 = np.hstack([np.array([-100, -30, -20, -10]), level1])
        level2 = np.linspace(densities[1].min(), densities[1].max(), 8)

        levels = [level1, level2]
    else:
        orders, colors, labels = [1, 0], ['r', 'b'], ['cat', 'noncat']
        if option == 'QSO':
            eps = (densities[0].max()-densities[0].min())/5.

            level1 = np.linspace(densities[0].min(), densities[0].max(), 6)
            level2 = np.linspace(densities[1].min()+eps, densities[1].max(), 8)

            levels = [level1, level2]
        else:
            level1 = np.linspace(-2, densities[0].max(), 8)
            level2 = np.linspace(densities[1].min(), densities[1].max(), 20)
            level2 = np.hstack([np.array([-200, -100, -10]), level2])

            levels = [level1, level2]

    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width

    # Set up the geometry of the three plots
    rect_scatter = [left, bottom, width, height]  # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25]  # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height]  # dimensions of y-histogram

    if option == 'ID':
        orders, colors, labels = [0, 1], ['b', 'r'], ['noncat', 'cat']
    else:
        orders, colors, labels = [1, 0], ['r', 'b'], ['cat', 'noncat']

    nbin = 51
    xbins = np.linspace(dcpmm[0], dcpmm[1], nbin)
    ybins = np.linspace(0, 0.1, nbin)
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig = plt.figure()

        axScatter = plt.axes(rect_scatter)  # scatter plot
        axHistx = plt.axes(rect_histx)  # x histogram
        axHisty = plt.axes(rect_histy)  # y histogram

        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        axScatter.scatter(dcps[orders[0]], variances[orders[0]],
                          color=colors[0], label=labels[0],
                          s=0.5, alpha=0.5, linewidth=0, edgecolor=None)
        axScatter.scatter(dcps[orders[1]], variances[orders[1]],
                          color=colors[1], label=labels[1],
                          s=0.5, alpha=0.5, linewidth=0, edgecolor=None)
        axScatter.contour(XX, ZZ, densities[0], levels[0],
                          colors='cyan', linewidths=0.75)
        axScatter.contour(XX, ZZ, densities[1], levels[1],
                          colors='orange', linewidths=0.75)

        axHistx.hist(dcps[orders[0]], xbins,
                     linestyle='solid', color=colors[0],
                     histtype='step', density=False)
        axHistx.hist(dcps[orders[1]], xbins,
                     linestyle='solid', color=colors[1],
                     histtype='step', density=False)

        axHisty.hist(variances[orders[0]], ybins,
                     linestyle='solid', color=colors[0], histtype='step',
                     orientation='horizontal', density=False)
        axHisty.hist(variances[orders[1]], ybins,
                     linestyle='solid', color=colors[1], histtype='step',
                     orientation='horizontal', density=False)

        axHistx.set_yscale('log')
        axHisty.set_xscale('log')

        tick_setting(axScatter, r"$L_{DCP}$", "Variance",
                     [dcpmm[0], dcpmm[1]], [0, 0.1], None, None)
        tick_setting(axHistx, None, None,
                     [dcpmm[0], dcpmm[1]], None, None, None)
        tick_setting(axHisty, None, None, None, [0, 0.1], None, None)

        handle1 = matplotlib.lines.Line2D([], [], c='r')
        handle2 = matplotlib.lines.Line2D([], [], c='b')
        labels = ["cat", "non-cat"]
        axScatter.legend(handles=[handle1, handle2],
                         labels=labels, loc='upper right',
                         bbox_to_anchor=(1.5, 1.4))

        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("Variance and DCP scattergram is saved at %s" % fn)


def plot_dcp_mode_avg_dff(zspec, zphot, dcpmm, probs, dcp_lab, option='ID'):
    fn = os.path.join(outdn, "dcp_zdiff%s" % option)

    dcp = dcp_lab[:, 0]

    cat_tol = 0.15

    rel_resid = np.abs(zspec-zphot)/(zspec+1)
    cat = rel_resid > cat_tol
    noncat = np.logical_not(cat)

    bin_edges = np.genfromtxt('./bin_edges/galaxy_redshifts_64-uniform.txt')
    bin_cents = np.array([(bin_edges[i]+bin_edges[2])/2. for i in range(len(bin_edges)-1)])

    arg_max = np.argmax(probs, 1)
    zmode = bin_cents[arg_max]

    zdiff = np.abs(zphot - zmode)
    cat_zdiff = zdiff[cat]
    noncat_zdiff = zdiff[noncat]

    cat_dcp = dcp[cat]
    noncat_dcp = dcp[noncat]

    dcps = [noncat_dcp, cat_dcp]
    zdiffs = [noncat_zdiff, cat_zdiff]

    densities = []
    xx = np.linspace(dcpmm[0], dcpmm[1], 100)
    zz = np.linspace(0, 0.6, 100)
    XX, ZZ = np.meshgrid(xx, zz)
    positions = np.vstack([XX.ravel(), ZZ.ravel()])
    for i in range(2):
        xz = np.vstack([dcps[i], zdiffs[i]])
        if len(xz) > 50000:
            randidx = np.random.choice(np.arange(xz.shape[1]), 50000)
            xz = xz[:, randidx]
        density = np.reshape(kde_scipy(xz, positions).T, XX.shape)
        densities.append(density)

    if option == 'ID':
        orders, colors, labels = [0, 1], ['b', 'r'], ['noncat', 'cat']

        level1 = np.linspace(densities[0].min(), densities[0].max(), 20)
        level1 = np.hstack([np.array([-100, -30, -20, -10]), level1])
        level2 = np.linspace(densities[1].min(), densities[1].max(), 8)

        levels = [level1, level2]
    else:
        orders, colors, labels = [1, 0], ['r', 'b'], ['cat', 'noncat']
        if option == 'QSO':
            eps = (densities[0].max()-densities[0].min())/5.

            level1 = np.linspace(densities[0].min(), densities[0].max(), 6)
            level2 = np.linspace(densities[1].min()+eps, densities[1].max(), 8)

            levels = [level1, level2]
        else:
            level1 = np.linspace(-2, densities[0].max(), 8)
            level2 = np.linspace(densities[1].min(), densities[1].max(), 20)
            level2 = np.hstack([np.array([-200, -100, -10]), level2])

            levels = [level1, level2]

    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width

    # Set up the geometry of the three plots
    rect_scatter = [left, bottom, width, height]  # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25]  # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height]  # dimensions of y-histogram

    if option == 'ID':
        orders, colors, labels = [0, 1], ['b', 'r'], ['noncat', 'cat']
    else:
        orders, colors, labels = [1, 0], ['r', 'b'], ['cat', 'noncat']

    nbin = 51
    xbins = np.linspace(dcpmm[0], dcpmm[1], nbin)
    ybins = np.linspace(0, 0.6, nbin)
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig = plt.figure()

        axScatter = plt.axes(rect_scatter)  # scatter plot
        axHistx = plt.axes(rect_histx)  # x histogram
        axHisty = plt.axes(rect_histy)  # y histogram

        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        axScatter.scatter(dcps[orders[0]], zdiffs[orders[0]],
                          color=colors[0], label=labels[0],
                          s=0.5, alpha=0.5, linewidth=0, edgecolor=None)
        axScatter.scatter(dcps[orders[1]], zdiffs[orders[1]],
                          color=colors[1], label=labels[1],
                          s=0.5, alpha=0.5, linewidth=0, edgecolor=None)
        axScatter.contour(XX, ZZ, densities[0], levels[0],
                          colors='cyan', linewidths=0.75)
        axScatter.contour(XX, ZZ, densities[1], levels[1],
                          colors='orange', linewidths=0.75)

        axHistx.hist(dcps[orders[0]], xbins,
                     linestyle='solid', color=colors[0],
                     histtype='step', density=False)
        axHistx.hist(dcps[orders[1]], xbins,
                     linestyle='solid', color=colors[1],
                     histtype='step', density=False)

        axHisty.hist(zdiffs[orders[0]], ybins,
                     linestyle='solid', color=colors[0], histtype='step',
                     orientation='horizontal', density=False)
        axHisty.hist(zdiffs[orders[1]], ybins,
                     linestyle='solid', color=colors[1], histtype='step',
                     orientation='horizontal', density=False)

        axHistx.set_yscale('log')
        axHisty.set_xscale('log')

        tick_setting(axScatter, r"$L_{DCP}$", "Variance",
                     [dcpmm[0], dcpmm[1]], [0, 0.6], None, None)
        tick_setting(axHistx, None, None,
                     [dcpmm[0], dcpmm[1]], None, None, None)
        tick_setting(axHisty, None, None, None, [0, 0.6], None, None)

        handle1 = matplotlib.lines.Line2D([], [], c='r')
        handle2 = matplotlib.lines.Line2D([], [], c='b')
        labels = ["cat", "non-cat"]
        axScatter.legend(handles=[handle1, handle2],
                         labels=labels, loc='upper right',
                         bbox_to_anchor=(1.5, 1.4))

        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("zdiff and DCP scattergram is saved at %s" % fn)


def quantity_scattergram(zphot, dcpmm, probs, dcp_lab, criteria,
                         option='ID', xy=['dcp', 'conf']):
    fn = os.path.join(outdn, "%s_%s%s" % (xy[0], xy[1], option))

    dcp = dcp_lab[:, 0]

    lims = []
    quants = []
    alabels = []
    for q in xy:
        if q == 'dcp':
            quant = dcp
            lims.append([dcpmm[0], dcpmm[1]])
            alabels.append(r"$L_{DCP}$")
        elif q == 'conf':
            confidence = np.max(probs, 1)
            quant = confidence

            lims.append([0, 1])
            alabels.append("Confidence")
        elif q == 'var':
            bin_edges = np.genfromtxt('./bin_edges/galaxy_redshifts_64-uniform.txt')
            bin_cents = np.array([(bin_edges[i]+bin_edges[2])/2. for i in range(len(bin_edges)-1)])

            variance = np.sum((bin_cents-zphot.reshape(-1, 1))**2. * probs, 1)
            quant = variance

            lims.append([0, 0.1])
            alabels.append("Variance")
        else:
            bin_edges = np.genfromtxt('./bin_edges/galaxy_redshifts_64-uniform.txt')
            bin_cents = np.array([(bin_edges[i]+bin_edges[2])/2. for i in range(len(bin_edges)-1)])

            arg_max = np.argmax(probs, 1)
            zmode = bin_cents[arg_max]
            zdiff = np.abs(zphot - zmode)
            quant = zdiff

            lims.append([0, 0.6])
            alabels.append(r"$|z_{avg} - z_{mode}|$")

        quants.append(quant)

    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width

    # Set up the geometry of the three plots
    rect_scatter = [left, bottom, width, height]  # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25]  # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height]  # dimensions of y-histogram

    nbin = 51
    xbins = np.linspace(lims[0][0], lims[0][1], nbin)
    ybins = np.linspace(lims[1][0], lims[1][1], nbin)

    bounds = [dcp.min(), criteria[0], criteria[1], dcp.max()]
    cmap = matplotlib.colors.ListedColormap(['blue', 'green', 'red'])
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig = plt.figure()

        axScatter = plt.axes(rect_scatter)  # scatter plot
        axHistx = plt.axes(rect_histx)  # x histogram
        axHisty = plt.axes(rect_histy)  # y histogram

        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        sp = axScatter.scatter(quants[0], quants[1],
                               c=dcp, cmap=cmap, norm=norm,
                               s=0.5, alpha=0.5, linewidth=0, edgecolor=None)

        axHistx.hist(quants[0], xbins,
                     color='blueviolet', density=False)

        axHisty.hist(quants[1], ybins, color='blueviolet',
                     orientation='horizontal', density=False)

        axHistx.set_yscale('log')
        axHisty.set_xscale('log')

        tick_setting(axScatter, alabels[0], alabels[1],
                     lims[0], lims[1], None, None)
        tick_setting(axHistx, None, None,
                     lims[0], None, None, None)
        tick_setting(axHisty, None, None, None, lims[1], None, None)

        fig.subplots_adjust(left=0.2,
                            right=0.6,
                            bottom=0.1,
                            top=0.9)
        cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])

        cbar = fig.colorbar(sp, cax=cbar_ax, ticks=bounds)
        cbar.set_label(r"$L_{DCP}$")

        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("%s and %s scattergram is saved at %s" % (xy[0], xy[1], fn))


def PPplot_density(ozspec, zphots, sample=False):
    fn = os.path.join(outdn, "PPplot_density_phase%s")
    phase = '2' if len(zphots) == 2 else '3'
    fn = fn % phase

    cat_tol = 0.15
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        if phase == '2':
            fig, axes = plt.subplots(1, len(zphots), figsize=(9, 3))
        else:
            fig, ax = plt.subplots()

        if len(zphots) == 2:
            titles = [r"$\bf{(a) \, Phase2 - HE}$", r"$\bf{(b) \, Phase2 - LE}$"]
        else:
            titles = [r"$\bf{Phase3 - avg}$"]

        for i, zphot in enumerate(zphots):
            if phase == '2':
                ax = axes[i]
            if sample:
                nsample = 20000
                rand_idx = np.random.choice(np.arange(len(ozspec)), nsample)
                temp_zspec = ozspec[rand_idx]
                zphot = zphot[rand_idx]
            else:
                temp_zspec = ozspec

            corr = np.corrcoef(temp_zspec, zphot)[0, 1]

            zsp = np.vstack([temp_zspec, zphot])
            zc = kde_scipy(zsp, zsp)
            cmap = plt.cm.get_cmap('jet')
            norm = matplotlib.colors.LogNorm()

            sc = ax.scatter(temp_zspec, zphot,
                            marker='.', c=zc,
                            s=0.5, cmap=cmap,
                            linewidth=0.3,
                            vmin=1, vmax=100,
                            norm=norm,
                            edgecolor=None)

            xx = np.arange(0, 10)
            ax.plot(xx, color='k',
                    ls='dashed', lw=0.6)
            ax.plot(xx, -cat_tol*(xx+1)+xx,
                    color='k', ls='dashed', lw=0.5)
            ax.plot(xx, cat_tol*(xx+1)+xx,
                    color='k', ls='dashed', lw=0.5)
            ax.title.set_text(titles[i])
            if phase == '3':
                ax.text(0.25, 0.9, r"$\rho = %.3f$" % corr,
                        ha='center', va='center',
                        transform=ax.transAxes)

            tick_setting(ax,
                         r"$z_{spec}$",
                         r"$z_{phot}$",
                         [0, 1], [0, 1],
                         [0, 0.2, 0.4, 0.6, 0.8, 1],
                         [0, 0.2, 0.4, 0.6, 0.8, 1])

        if len(zphots) == 2:
            fig.subplots_adjust(left=0.2,
                                right=0.8,
                                bottom=0.1,
                                top=0.9,
                                wspace=0.3)
            cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
        else:
            fig.subplots_adjust(left=0.2,
                                right=0.8,
                                bottom=0.1,
                                top=0.9)
            cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])

        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label("Density")
        cbar.ax.tick_params(axis='y', which='both', direction='out')

        fig.savefig(fn+'.png', format='png', dpi=200)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        plt.close()

        logging.info("Scattergram is saved at %s" % fn)


def plot_residual_pdf(zspec, zphots):
    fn = os.path.join(outdn, "redshift_difference_distribution")

    bin_edges = np.linspace(0, 1, 501)
    bin_center = [(bin_edges[i]+bin_edges[i+1])/2.
                  for i in range(len(bin_edges)-1)]
    pdf_zspec = kde_scipy(zspec, bin_center)

    linestyles = ['-', '--', '-.'][::-1]

    labels = ["Phase2 - HE", "Phase2 - LE", "Phase3 - avg"]

    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()

        ax.plot(bin_center, pdf_zspec/2.,
                color='k', linestyle='-',
                label=r"$z_{spec}$")
        for i, zphot in enumerate(zphots):
            pdf_zphot = kde_scipy(zphot, bin_center)
            pdf_residual = pdf_zspec-pdf_zphot

            ax.plot(bin_center, pdf_residual,
                    label=labels[i], linestyle=linestyles[i],
                    color=color_scheme[2-i])
            ax.hlines(0, -0.1, 1, linestyle='--', linewidth=0.5, color='k')
        ax.legend()
        tick_setting(ax,
                     "Redshift",
                     "Number density \n (or redshift difference)",
                     [0, 1], None,
                     [0, 0.2, 0.4, 0.6, 0.8, 1],
                     [-2, -1, 0, 1])
        fig.savefig(fn+'.png', format='png')
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        plt.close()

        logging.info("Number density distribution is saved at %s" % fn)


def metric_analysis(zspec, zphots):
    zspec = zspec.ravel()

    outs = []
    for zphot in zphots:
        zphot = zphot.ravel()

        resid = (zspec-zphot)/(1+zspec)
        resid_abs = np.abs(resid)
        bias = np.abs(np.mean(resid))
        mar = np.mean(resid_abs)
        sig = np.std(resid)
        sig68 = np.percentile(resid_abs, 68)
        nmad = np.median(resid_abs)*1.4826
        rcat = np.sum(resid_abs > 0.15)/len(resid_abs)

        header = ['bias', 'mar', 'sig', 'sig68', 'nmad', 'rcat']
        outs.append([bias, mar, sig, sig68, nmad, rcat])

    sfn = os.path.join(outdn, 'metrics.csv')

    with open(sfn, 'w') as outf:
        writer = csv.writer(outf)
        writer.writerow(header)
        for x in outs:
            out = [x[i] for i in range(len(x))]
            writer.writerow(out)
    logging.info("metric is saved at %s" % sfn)


def plot_dcp_confidence_single(prob, dcp, dcpmm, criteria,
                               zphot, zspec, key='ID', sample=False):
    fn = os.path.join(outdn, "dcp_confidence_cat_noncat_%s" % key)

    if sample:
        nsample = 20000
        rand_idx = np.random.choice(np.arange(len(dcp)), nsample)
        prob = prob[rand_idx]
        dcp = dcp[rand_idx]

    conf = np.max(prob, 1)

    abs_diff = np.abs(zspec-zphot)/(zspec+1)
    cat = abs_diff > 0.15
    ncat = abs_diff <= 0.15

    cat_dcp = dcp[cat]
    cat_conf = conf[cat]

    ncat_dcp = dcp[ncat]
    ncat_conf = conf[ncat]

    colors = ['darkturquoise', 'blue']
    labels = ["non-cat", "cat"]
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()

        # ax.scatter(cat_dcp, cat_conf,
        #            color=colors[0], s=0.5,
        #            linewidth=0, edgecolor=None,
        #            label=labels[0])
        ax.scatter(ncat_dcp, ncat_conf,
                   color=colors[1], s=0.5,
                   linewidth=0, edgecolor=None)
        ax.scatter(cat_dcp, cat_conf,
                   color=colors[0], marker='o',
                   s=3, linewidth=0, edgecolor=None)
        # ax.contour(XX, ZZ, density, level,
        #            colors='deepskyblue', linewidths=1.5)
        tick_setting(ax, r"$L_{DCP}$", "Confidence",
                     [0, dcpmm[1]], [0, 1], None, None)
        # ax.legend(loc='upper right', markerscale=4)
        fig.savefig(fn)
        fig.savefig(fn+'.pdf', format='pdf', dpi=300)
        logging.info("DCP and confidence scattergram is saved at %s" % fn)


def main():
    zphot_id_univ, zphot_id_spec = get_ft_zphots()

    dcp_lab, zphots = get_outputs()
    zspec_id, colors_id, _ = get_unnorm('ID')
    zspec_lo, colors_lo, lo_labels = get_unnorm('LOOD')
    colors_ul = get_unnorm('UL')

    probs_id_avg = get_probs('id')
    # probs_lo_avg = get_probs('lo')
    # probs_ul_avg = get_probs('ul')

    # probs_id = get_probs('id', avg=False)
    probs_lo = get_probs('lo', avg=False)
    # probs_ul = get_probs('ul', avg=False)

    zphot_id_avg = (zphots[0][:, 0] + zphots[0][:, 1])/2.
    zphot_lood_avg = (zphots[1][:, 0] + zphots[1][:, 1])/2.
    zphot_ul_avg = (zphots[2][:, 0] + zphots[2][:, 1])/2.

    zspecs = [zspec_id, zspec_lo]
    zphots_avg = [zphot_id_avg, zphot_lood_avg]

    qso_mask = lo_labels == 1
    star_mask = lo_labels == 3

    zspec_qso = zspec_lo[qso_mask]
    zspec_star = zspec_lo[star_mask]

    zphot_qso = zphot_lood_avg[qso_mask]
    zphot_star = zphot_lood_avg[star_mask]

    dcp_lab_qso = dcp_lab[1][qso_mask]
    dcp_lab_star = dcp_lab[1][star_mask]

    probs_qso = [probs_lo[0][qso_mask], probs_lo[1][qso_mask]]
    probs_star = [probs_lo[0][star_mask], probs_lo[1][star_mask]]

    probs_qso_avg = (probs_qso[0]+probs_qso[1])/2.
    probs_star_avg = (probs_star[0]+probs_star[1])/2.

    probs_avg = [probs_id_avg, probs_qso_avg, probs_star_avg]

    # criterion1 = np.percentile(dcp_lab[0][:, 0], 95)
    # criterion2 = np.percentile(dcp_lab[0][:, 0], 99)
    dcp_id = dcp_lab[0][:, 0]
    dcp_int = dcp_id.max() - dcp_id.min()
    criterion1 = dcp_id.min() + dcp_int/3.
    criterion2 = dcp_id.min() + dcp_int/3.*2
    criteria = [criterion1, criterion2]

    colors_qso = colors_lo[qso_mask]
    colors_star = colors_lo[star_mask]

    dcp_id = dcp_lab[0][:, 0]
    dcp_qso = dcp_lab[1][:, 0][qso_mask]
    dcp_star = dcp_lab[1][:, 0][star_mask]

    colors_cprhsv = [colors_id, colors_qso, colors_star]
    dcp_cprhsv = [dcp_id, dcp_qso, dcp_star]

    dcp_min = min([d.min() for d in dcp_cprhsv])
    dcp_max = max([d.max() for d in dcp_cprhsv])

    dcp_min_max = [dcp_min, dcp_max]

    zphot_ppden = [zphot_id_univ, zphot_id_spec, zphot_id_avg]

    # ID_DCP_NumberDensity(zspec_id, zphot_id_avg, dcp_id, dcp_min_max, criteria)
    # plot_dcp_confidence_single(probs_id_avg, dcp_id, dcp_min_max, criteria,
    #                            zphot_id_avg, zspec_id, key='ID', sample=False)
    plot_dcp_confidence(probs_avg, dcp_cprhsv, dcp_min_max, criteria)
    quit()
    # metric_analysis(zspec_id, zphot_ppden)

    # plot_residual_pdf(zspec_id, zphot_ppden)

    # PPplot_density(zspec_id, zphot_ppden[:2], sample=False)
    # PSCPDF()
    # quit()
    # PPplot_density(zspec_id, [zphot_ppden[-1]], sample=False)


    # DiscriminatorNumberDensity(dcp_lab[0], dcp_lab[1], dcp_lab[2], criteria)

    # OOD_DCP_NumberDensity(dcp_qso, dcp_star, dcp_min_max, criteria)


    # OODScoreCatNoncat(zspec_id, zphot_id_avg, dcp_lab[0], criteria,
    #                   option='ID', sample=False)
    # OODScoreCatNoncat(zspec_qso, zphot_qso, dcp_lab_qso, criteria,
    #                   option='qso', sample=False)
    # OODScoreCatNoncat(zspec_star, zphot_star, dcp_lab_star, criteria,
    #                   option='star', sample=True)
    # PPplot(zspecs[0], zphots_avg[0], dcp_lab[0], criteria,
    #        option='ID_avg', sample=False)
    # quit()

    id_col = [colors_id[:, 0], colors_id[:, 4], colors_id[:, 8]]
    ul_col = [colors_ul[:, 0], colors_ul[:, 4], colors_ul[:, 8]]

    # ColRedScattergram(id_col, zspec_id, dcp_lab[0], criteria,
    #                   option="ID", sample=False)
    # ColRedScattergram(ul_col, zphot_ul_avg, dcp_lab[2], criteria,
    #                   color_id=id_col, option="UL",
    #                   phot_red_id=zspec_id,
    #                   ylabel=r"$z_{phot}$", sample=True)
    # quit()

    dtypes = ["ID", "QSO", "Star"]
    samples = [None, None, 500000]
    xlims1 = [None, [-0.5, 2], [-0.5, 2.0]]
    ylims1 = [None, [-0.5, 1.0], [-0.2, 0.6]]

    xlims2 = [None, [-0.5, 1], [-0.5, 2]]
    ylims2 = [None, [-0.5, 1.0], [-0.2, 0.6]]

    # [0, 4], [4,8], [8,12], [0, 8], [0, 12], [4, 12]
    idx1 = [[2, 6], [0, 12], [0, 12]]
    idx2 = [[3, 7], [4, 12], [4, 12]]

    xlabels1 = [None, r"$(g-r)$", r"$(g-r)$"]
    xlabels2 = [None, r"$(r-i)$", r"$(r-i)$"]

    ylabels1 = [None, r"$(z-y)$", r"$(z-y)$"]
    ylabels2 = [None, None, None]
    for i in range(2, 3):
        ColColScattergram(colors_cprhsv[i], dcp_cprhsv[i],
                          colors_id, idx1[i], idx2[i], criteria,
                          xlim=[xlims1[i], xlims2[i]],
                          ylim=[ylims1[i], ylims2[i]],
                          xlabel=[xlabels1[i], xlabels2[i]],
                          ylabel=[ylabels1[i], ylabels2[i]],
                          dtype=dtypes[i], sample=samples[i])

        # ColColScattergram(colors_cprhsv[i][:, idx1[i][0]],
        #                   colors_cprhsv[i][:, idx1[i][1]], dcp_cprhsv[i],
        #                   colors_id[:, idx1[i][0]], colors_id[:, idx1[i][1]],
        #                   colors_ul[:, idx1[i][0]], colors_ul[:, idx1[i][1]],
        #                   criteria, xlim=xlims1[i], ylim=ylims1[i],
        #                   xlabel=xlabels1[i], ylabel=ylabels1[i],
        #                   dtype=dtypes[i], sample=samples[i])

        # ColColScattergram(colors_cprhsv[i][:, idx2[i][0]],
        #                   colors_cprhsv[i][:, idx2[i][1]], dcp_cprhsv[i],
        #                   colors_id[:, idx2[i][0]], colors_id[:, idx2[i][1]],
        #                   colors_ul[:, idx2[i][0]], colors_ul[:, idx2[i][1]],
        #                   criteria, xlim=xlims2[i], ylim=ylims2[i],
        #                   xlabel=xlabels2[i], ylabel=ylabels2[i],
        #                   coln=1, dtype=dtypes[i], sample=samples[i])

if __name__ == '__main__':
    main()
