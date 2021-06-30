import os
import logging
import itertools

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
import pandas as pd
import scipy.ndimage as ndimage

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde  # accurate but slow
from kde_methods import kde_histogram  # not accurate but fast


# print(matplotlib.matplotlib_fname())
color_scheme = ['red', 'orange', 'green', 'blue', 'indigo', 'black']


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def random_under_sampling(data, n=20000, ridx=None, axis=0):
    ndata = data.shape[axis]
    assert ndata > n
    if ridx is None:
        ridx = np.random.choice(np.arange(ndata), n, replace=False)
    if axis:
        return data[:, ridx], ridx
    return data[ridx], ridx


def tick_setting(ax, xlabel, ylabel, xlim, ylim, xticks, yticks):
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


def DatasetColorPairPlot(Xnormed, dtype='ID',
                         ID='galaxy', err=False, kind='scatter'):
    paper_plot_dn = 'DataPlots_ind_'+ID
    dn = os.path.join(paper_plot_dn, dtype)
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = 'PairPlot'
    fn += '_err' if err else ''
    sn = os.path.join(dn, fn+'_'+kind)

    column_header = [r"$(g-r)_", r"$(r-i)_",
                     r"$(i-z)_", r"$(z-y)_", r"$E(B-V)$"]
    column_tail = [r"{ce}$", r"{de}$"] if err else [r"{c}$", r"{d}$"]

    columns = []
    for i in range(Xnormed.shape[1]):
        if not err and i == Xnormed.shape[1]-1:
            columns.append(column_header[-1])
        else:
            columns.append(column_header[i//2]+column_tail[i % 2])

    Xnormed_DF = pd.DataFrame(Xnormed, columns=columns)
    if kind == 'scatter':
        g = sns.pairplot(Xnormed_DF,
                         markers='.',
                         plot_kws={'s': 10},
                         diag_kind='hist',
                         diag_kws={'bins': 50})
    else:
        g = sns.pairplot(Xnormed_DF, kind='kde')
        g = g.map_offdiag(sns.kdeplot, lw=1)

    g.fig.set_size_inches(20, 20)
    plt.savefig(sn)
    plt.savefig(sn+'.pdf', format='pdf', dpi=300)
    plt.close('all')
    logging.info("Pair plot is saved at %s" % sn)


def DatasetXCScattergram(Xnormed, zspec, dtype='ID', ID='galaxy'):
    paper_plot_dn = 'DataPlots_ind_'+ID
    if not os.path.exists(paper_plot_dn):
        os.makedirs(paper_plot_dn)

    zspec = zspec.ravel()
    dn = dtype
    dn = os.path.join(paper_plot_dn, dn)
    if not os.path.exists(dn):
        os.makedirs(dn)

    xlabel_header = [r"$(g-r)_", r"$(r-i)_",
                     r"$(i-z)_", r"$(z-y)_", r"$E(B-V)$"]
    xlabel_tail = [r"{c}$", r"{ce}$", r"{d}$", r"{de}$"]
    for i, X in enumerate(Xnormed):
        zspec = zspec
        sn = os.path.join(dn, 'XCScattergram'+str(i+1).zfill(2))
        if i != 16:
            xlabel = xlabel_header[i//4]+xlabel_tail[i % 4]
        else:
            xlabel = xlabel_header[4]

        with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
            plt.rcParams["font.family"] = "Times New Roman"

            cm = plt.cm.get_cmap('jet')
            fig, ax = plt.subplots()
            density = kde_histogram(X, zspec)
            pc = ax.scatter(X, zspec, c=density,
                            marker='.', alpha=0.5,
                            s=0.5, cmap=cm,
                            linewidth=0.3,
                            norm=matplotlib.colors.LogNorm(),
                            edgecolor=None,
                            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$z_{spec}$")

            fig.colorbar(pc)
            # fig.tight_layout()
            fig.savefig(sn)
            fig.savefig(sn+'.pdf', format='pdf', dpi=300)
            plt.close('all')
            logging.info("Color-redshift scattergram is saved at %s" % sn)


def DatasetColorNumberDensity(Xtrain, Xval, Xeval, dtype='ID', ID='galaxy'):
    paper_plot_dn = 'DataPlots_ind_'+ID
    if not os.path.exists(paper_plot_dn):
        os.makedirs(paper_plot_dn)
    dn = dtype
    dn = os.path.join(paper_plot_dn, dn)
    if not os.path.exists(dn):
        os.makedirs(dn)

    xlabel_header = [r"$(g-r)_", r"$(r-i)_",
                     r"$(i-z)_", r"$(z-y)_", r"$E(B-V)$"]
    xlabel_tail = [r"{c}$", r"{ce}$", r"{d}$", r"{de}$"]
    for i, (Xt, Xv, Xe) in enumerate(zip(Xtrain, Xval, Xeval)):
        if dtype == 'ID':
            bin_min = min(Xt.min(), Xv.min(), Xe.min())
            bin_max = max(Xt.max(), Xv.max(), Xe.max())
            bins = np.linspace(bin_min, bin_max, 101)
        else:
            if i == 0 and 'UL' not in dtype:
                bin_max = max(Xt.max(), Xv.max(), Xe.max())
                bins = np.linspace(0, bin_max, 101)
            else:
                bins = np.linspace(0, 1, 101)
        sn = os.path.join(dn, 'CHistogram'+str(i+1).zfill(2))

        if 'UL' not in dtype:
            if i == 0:
                xlabel = r"$z_{spec}$"
            elif i != 17:
                xlabel = xlabel_header[(i-1)//4]+xlabel_tail[(i-1) % 4]
            else:
                xlabel = xlabel_header[4]
        else:
            if i != 16:
                xlabel = xlabel_header[(i)//4]+xlabel_tail[(i) % 4]
            else:
                xlabel = xlabel_header[4]

        with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
            plt.rcParams["font.family"] = "Times New Roman"

            fig, ax = plt.subplots()
            ax.hist(Xt, bins,
                    histtype='step', color=color_scheme[0],
                    density=True, label='train')
            ax.hist(Xv, bins,
                    histtype='step', color=color_scheme[1],
                    density=True, label='validation')
            ax.hist(Xe, bins,
                    histtype='step', color=color_scheme[2],
                    density=True, label='evaluation')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Number density')
            ax.legend()

            fig.savefig(sn)
            fig.savefig(sn+'.pdf', format='pdf', dpi=300)
            plt.close('all')
            logging.info("Color histogram is saved at %s" % sn)


def ColorColorScattergram(db, dcp_label, dcp_ul, opt, color=True):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = 'ColorColorScattergram'
    fn += '' if color else '_err'
    sn = os.path.join(dn, fn)
    density_dn = './density'
    density_dn += '' if color else '_err'
    if not os.path.exists(density_dn):
        os.makedirs(density_dn)

    color_id = db['eval_in'].dataset.X.numpy().T
    color_lo = db['eval_lo'].dataset.X.numpy().T
    color_ul = db['eval_ul'].dataset.X.numpy().T

    dcp_id = dcp_label[0][:, 0]
    dcp_lo = dcp_label[1][:, 0]

    dcp = np.hstack((dcp_id, dcp_lo, dcp_ul))
    bound_int = dcp.min()+dcp.max()
    bounds = [dcp.min(), bound_int/3., bound_int*2/3., dcp.max()]

    # color_id, ridx_id = random_under_sampling(color_id, axis=1)
    # color_lo, ridx_lo = random_under_sampling(color_lo, axis=1)
    # color_ul, ridx_ul = random_under_sampling(color_ul, axis=1)

    # dcp_id = dcp_id[ridx_id]
    # dcp_lo = dcp_lo[ridx_lo]
    # dcp_ul = dcp_ul[ridx_ul]
    dcps = [dcp_id, dcp_lo, dcp_ul]

    column_header = [r"$(g-r)_", r"$(r-i)_",
                     r"$(i-z)_", r"$(z-y)_", r"$E(B-V)$"]
    column_tail = [r"{c}$", r"{d}$"] if color else [r"{ce}$", r"{de}$"]

    start_i = 0 if color else 1
    iterators = np.arange(start_i, len(color_id)-2, 4)
    titles = ["In-Distribution", "Labeled Out-of-Distribution", "Unlabeled"]
    if opt.psc:
        titles[-1] += " (LPSC)" if opt.psc_reg == 'low' else " (HPSC)"
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        cmap = matplotlib.colors.ListedColormap(['blue', 'green', 'red'])
        for l, i in enumerate(iterators):
            if color:
                if i == iterators[-1]:
                    si = -1
                    i = 0
                else:
                    si = 4
            else:
                si = 2
            fig, axes = plt.subplots(1, 3, figsize=(9, 4))
            ci = [color_id[i], color_id[i+si]]
            cl = [color_lo[i], color_lo[i+si]]
            cu = [color_ul[i], color_ul[i+si]]
            colors = [ci, cl, cu]

            den_id_fn = 'density_id_%s.npy' % str(l).zfill(2)
            den_lo_fn = 'density_lo_%s.npy' % str(l).zfill(2)
            den_id_fn = os.path.join(density_dn, den_id_fn)
            den_lo_fn = os.path.join(density_dn, den_lo_fn)
            if os.path.exists(den_id_fn) and os.path.exists(den_lo_fn):
                logging.info("Load ID density from %s" % den_id_fn)
                logging.info("Load LO density from %s" % den_lo_fn)
                XX_id, YY_id, density_id = np.load(den_id_fn)
                XX_lo, YY_lo, density_lo = np.load(den_lo_fn)
                XXs = [XX_id, XX_lo]
                YYs = [YY_id, YY_lo]
                densities = [density_id, density_lo]
            else:
                XXs, YYs = [], []
                densities = []
                den_fn = [den_id_fn, den_lo_fn]
                for k, col in enumerate(colors[:-1]):
                    xx = np.linspace(col[0].min(), col[0].max(), 100)
                    yy = np.linspace(col[1].min(), col[1].max(), 100)
                    XX, YY = np.meshgrid(xx, yy)
                    XXs.append(XX)
                    YYs.append(YY)
                    positions = np.vstack([XX.ravel(), YY.ravel()])

                    cc = np.vstack([col[0], col[1]])
                    randidx = np.random.choice(np.arange(cc.shape[1]), 20000)
                    cc = cc[:, randidx]

                    density = np.reshape(kde_scipy(cc, positions).T, XX.shape)
                    densities.append(density)

                    np.save(den_fn[k], [XX, YY, density])

                    logging.info("Density is saved at %s" % den_fn[k])

            for j, (c1, c2) in enumerate(colors):
                dcp = dcps[j]

                nsamples = [np.sum((bounds[k] <= dcp) &
                            (bounds[k+1] > dcp))
                            for k in range(cmap.N)]
                nsamples = np.array(nsamples[::-1])/np.sum(nsamples)
                # scolors = ['r', 'g', 'b']
                # legend_handles = []
                # for k in range(cmap.N):
                #     handle = mlines.Line2D([], [], color=scolors[k],
                #                            marker='.', linestyle='None',
                #                            markersize=5)
                #     legend_handles.append(handle)
                # axes[j].legend(loc='best', handles=legend_handles)

                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                sf = axes[j].scatter(c1, c2, c=dcp,
                                     marker='.', alpha=0.5,
                                     s=0.5, cmap=cmap,
                                     linewidth=0.3,
                                     norm=norm,
                                     edgecolor=None)

                den_int = (densities[0].max()-densities[0].min())/10.
                axes[j].contour(XXs[0], YYs[0], densities[0],
                                np.linspace(densities[0].min()+den_int,
                                            densities[0].max(), 8),
                                colors='k', linewidths=0.5)
                if si == -1:
                    axes[j].set_xlabel(column_header[i]+column_tail[0])
                else:
                    axes[j].set_xlabel(column_header[i//4]+column_tail[0])
                    if j == 2:
                        axes[j].set_xlim(axes[0].get_xlim())
                        axes[j].set_ylim(axes[0].get_ylim())

                axes[j].set_title(titles[j])

                if j == 0:
                    if si == -1:
                        axes[j].set_ylabel(column_header[i+si])
                    else:
                        axes[j].set_ylabel(column_header[i//4+1]+column_tail[0])

            fig.subplots_adjust(left=0.2,
                                right=0.8,
                                bottom=0.1,
                                top=0.9,
                                wspace=0.25,
                                hspace=0.3)
            cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(sf, cax=cbar_ax, ticks=bounds)
            cbar.set_label("Discrepancy loss")
            cbar.ax.tick_params(axis='y', which='both', direction='out')

            # fig.tight_layout()
            temp_sn = sn+str(l).zfill(2)
            plt.savefig(temp_sn)
            plt.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
            plt.close('all')
            logging.info("Color-Color diagram is saved at %s" % temp_sn)


def DiscriminatorNumberDensity(dcp_label, dcp_ul, opt, dscrm):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    sn = os.path.join(dn, dscrm.replace(' ', '')+'NumberDensity')

    # uniform_p = np.zeros(opt.ncls)+1./opt.ncls
    # max_entropy = np.sum(-uniform_p * np.log(uniform_p))

    min_dscrm = min([d_l.T[0].min() for d_l in dcp_label])
    max_dscrm = max([d_l.T[0].max() for d_l in dcp_label])
    xlim = [1, max_dscrm+0.2]

    # This is an approximation, but sufficiently close.
    bins = np.linspace(min_dscrm, max_dscrm, 101)

    colors = ['b', 'g']
    labels = ["ID", "LO"]
    with plt.style.context(['science', 'ieee', 'high-vis', 'grid', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()
        for i, d_l in enumerate(dcp_label):
            d = d_l.T[0]

            # dpdf = kde_scipy(d, bins, set_bandwidth=0.5)
            ax.hist(d, bins,
                    histtype='step',
                    density=True,
                    color=colors[i],
                    label=labels[i])
            # ax.plot(bins, dpdf,
            #         color=color_scheme[i],
            #         label=labels[i])
        ax.hist(dcp_ul, bins,
                histtype='step',
                density=True,
                color='r',
                label="UL")
        ax.legend(loc='upper center')
        ax.set_ylabel("Number density")
        ax.set_xlabel("Discrepancy loss")
        tick_setting(ax,
                     "Discrepancy loss",
                     "Number density",
                     None,
                     [0, 13],
                     [0, 1, 2, 3, 4],
                     [0, 2, 4, 6, 8, 10, 12])
        fig.savefig(sn)
        fig.savefig(sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
    logging.info("%s number density distribution is saved at %s" %
                 (dscrm, sn))


def ROC_PR_Curves(dcp_label, opt):
    sn = os.path.join(opt.plot_fd, 'ROCPRCurves')

    dcp_label = np.vstack(dcp_label)
    dcp = dcp_label.T[0]
    label = dcp_label.T[1]
    # label = np.where(label == 0, 1, 0)
    with plt.style.context(['science', 'ieee', 'high-vis', 'grid', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots(1, 2, figsize=(7, 4))
        ax[0].set_title("ROC Curve")

        fpr, tpr, _ = metrics.roc_curve(label, dcp)
        prc, rec, _ = metrics.precision_recall_curve(label, dcp)
        roc_auc = metrics.auc(fpr, tpr)
        prc_auc = metrics.auc(rec, prc)

        no_skill = np.linspace(0, 1, 100)

        ax[0].plot(no_skill, no_skill,
                   color='k', linestyle='dashed')
        ax[0].plot(fpr, tpr,
                   color=color_scheme[0],
                   label="AUC = %0.3f" % roc_auc)

        # ax[0].set_xscale('log')
        ax[0].legend(loc='lower right')
        tick_setting(ax[0],
                     "False positive rate",
                     "True positive rate",
                     [-0.05, 1.05],
                     [-0.05, 1.05],
                     #  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                     [0, 0.2, 0.4, 0.6, 0.8, 1],
                     [0, 0.2, 0.4, 0.6, 0.8, 1])

        ax[1].set_title("PR Curve")

        no_skill = len(label[label == 1])/len(label)
        ax[1].hlines(no_skill, 0, 1, color='k', linestyle='dashed')

        ax[1].plot(rec, prc,
                   color=color_scheme[0],
                   label='AUC = %0.3f' % prc_auc)

        # ax[1].set_xscale('log')
        ylim_eps = 0.05*(1-no_skill)
        yticks = np.arange(1, no_skill-0.05, -0.05)[::-1]
        ax[1].legend(loc='lower right')
        tick_setting(ax[1],
                     "Recall",
                     "Precision",
                     [-0.05, 1.05],
                     [no_skill-ylim_eps, 1.0+ylim_eps],
                     #  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                     [0, 0.2, 0.4, 0.6, 0.8, 1],
                     yticks)

        fig.subplots_adjust(wspace=0.2)
        # fig.tight_layout()
        fig.savefig(sn+'.png', format='png')
        fig.savefig(sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("ROC and PR curves are saved at %s" % sn)

    return roc_auc, prc_auc


def lkhd_ROCCurves(rll_label, ll_label, opt):
    sn = os.path.join(opt.plot_fd, 'ROCCurves')

    rll_label = np.vstack(rll_label)
    ll_label = np.vstack(ll_label)
    rll = rll_label.T[0]
    ll = ll_label.T[0]
    label = ll_label.T[-1]
    with plt.style.context(['science', 'ieee', 'high-vis', 'grid', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()
        ax.set_title("Receiver Operating Characteristic")

        rfpr, rtpr, _ = metrics.roc_curve(label, rll)
        fpr, tpr, _ = metrics.roc_curve(label, ll)

        rroc_auc = metrics.auc(rfpr, rtpr)
        roc_auc = metrics.auc(fpr, tpr)

        rfpr_nonzero_args = rfpr.nonzero()
        rfpr = rfpr[rfpr_nonzero_args]
        rtpr = rtpr[rfpr_nonzero_args]

        fpr_nonzero_args = fpr.nonzero()
        fpr = fpr[fpr_nonzero_args]
        tpr = tpr[fpr_nonzero_args]

        ax.plot(rfpr, rtpr,
                color=color_scheme[0],
                label='Log-lkhd-ratio\nAUC = %0.3f' % rroc_auc)
        ax.plot(fpr, tpr,
                color=color_scheme[1],
                label='Log-lkhd\nAUC = %0.3f' % roc_auc)

        ax.set_xscale('log')
        ax.legend(loc='upper left')
        tick_setting(ax,
                     "False positive rate",
                     "True positive rate",
                     [1e-5, 2],
                     [0, 1.05],
                     [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                     [0, 0.2, 0.4, 0.6, 0.8, 1]
                     )

        # fig.tight_layout()
        fig.savefig(sn+'.png', format='png')
        fig.savefig(sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("ROC curves are saved at %s" % sn)


def Scattergram(zspec, zpred, dcp_l, dcp_ul, 
                prefixs, opt, dtype='ID', col='kde'):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    sn = os.path.join(dn, 'redshift_scattergram')
    if col == 'dcp':
        didx = 0 if dtype == 'ID' else 1
        dcp = dcp_l[didx][:, 0]
        sn += '_color_dcp'

        dcp_id = dcp_l[0][:, 0]
        dcp_lo = dcp_l[1][:, 0]
        dcp_bound = np.hstack((dcp_id, dcp_lo, dcp_ul))
        bound_int = dcp_bound.min()+dcp_bound.max()
        bounds = [dcp_bound.min(), bound_int/3.,
                  bound_int*2/3., dcp_bound.max()]

    # zspec, ridx = random_under_sampling(zspec)

    lims = [0, 1] if dtype == 'ID' else [0, 8]
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1] if dtype == 'ID' else [0, 2, 4, 6, 8]

    zspec = zspec.ravel()
    cat_tol = 0.15
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        ztypes = ['average', 'mode']
        for i, zphots in enumerate(zpred[:1]):
            for j, zphot in enumerate(zphots):
                # zphot = zphot[ridx]
                corr = np.corrcoef(zphot, zspec)[0, 1]
                if col == 'kde':
                    zsp = np.vstack([zphot, zspec])
                    zc = kde_scipy(zsp, zsp)
                    cmap = plt.cm.get_cmap('jet')
                    norm = matplotlib.colors.LogNorm()
                else:
                    cmap = matplotlib.colors.ListedColormap(['blue', 'green', 'red'])
                    assert cmap.N == len(bounds)-1
                    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                    zc = dcp  # [ridx]

                fig, ax = plt.subplots()
                sc = ax.scatter(zphot, zspec,
                                marker='.', c=zc,
                                s=0.5, cmap=cmap,
                                linewidth=0.3,
                                vmin=1, vmax=100,
                                norm=norm,
                                edgecolor=None)

                ax.plot(np.arange(0, 10), color='k',
                        ls='dashed', lw=0.6)
                ax.plot(np.arange(0, 10),
                        np.arange(0, 10)/(1+cat_tol)-cat_tol/(1+cat_tol),
                        color='k', ls='dashed', lw=0.5)
                ax.plot(np.arange(0, 10),
                        np.arange(0, 10)/(1-cat_tol)+cat_tol/(1-cat_tol),
                        color='k', ls='dashed', lw=0.5)
                ax.text(0.25, 0.9, r"$\rho = %.3f$" % corr,
                        ha='center', va='center',
                        transform=ax.transAxes)

                tick_setting(ax,
                             r"$z_{phot}$",
                             r"$z_{spec}$",
                             lims, lims,
                             ticks, ticks)

                cbar = fig.colorbar(sc)
                if col == 'kde':
                    cbar.set_label("Density")
                else:
                    cbar.set_label("Discrepancy loss")

                temp_sn = sn+'_'+prefixs[j]+'_'+dtype+'_'+ztypes[i]
                fig.savefig(temp_sn+'.png', format='png')
                fig.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
                plt.close('all')

                logging.info("Scattergram is saved at %s" % temp_sn)


def AE_Scattergram(zspec, ezpred, dcp_l, dcp_ul, opt,
                   dtype='ID', col='kde', key='ensemble'):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    sn = os.path.join(dn, 'redshift_scattergram')
    if col == 'dcp':
        didx = 0 if dtype == 'ID' else 1
        dcp = dcp_l[didx][:, 0]
        sn += '_color_dcp'

        dcp_id = dcp_l[0][:, 0]
        dcp_lo = dcp_l[1][:, 0]
        dcp_bound = np.hstack((dcp_id, dcp_lo, dcp_ul))
        bound_int = dcp_bound.min()+dcp_bound.max()
        bounds = [dcp_bound.min(), bound_int/3., 
                  bound_int*2/3., dcp_bound.max()]

    # zspec, ridx = random_under_sampling(zspec)

    lims = [0, 1] if dtype == 'ID' else [0, 8]
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1] if dtype == 'ID' else [0, 2, 4, 6, 8]

    zspec = zspec.ravel()
    cat_tol = 0.15
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        ztypes = ['average', 'mode']
        for i, zphot in enumerate(ezpred[:1]):
            # zphot = zphot[ridx]
            corr = np.corrcoef(zphot, zspec)[0, 1]
            if col == 'kde':
                zsp = np.vstack([zphot, zspec])
                zc = kde_scipy(zsp, zsp)
                cmap = plt.cm.get_cmap('jet')
                norm = matplotlib.colors.LogNorm()
            else:
                cmap = matplotlib.colors.ListedColormap(
                    ['blue', 'green', 'red'])
                assert cmap.N == len(bounds)-1
                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                zc = dcp  # [ridx]

            fig, ax = plt.subplots()
            sc = ax.scatter(zphot, zspec,
                            marker='.', c=zc,
                            s=0.5, cmap=cmap,
                            linewidth=0.3,
                            vmin=1, vmax=100,
                            norm=norm,
                            edgecolor=None)

            ax.plot(np.arange(0, 10), color='k',
                    ls='dashed', lw=0.6)
            ax.plot(np.arange(0, 10),
                    np.arange(0, 10)/(1+cat_tol)-cat_tol/(1+cat_tol),
                    color='k', ls='dashed', lw=0.5)
            ax.plot(np.arange(0, 10),
                    np.arange(0, 10)/(1-cat_tol)+cat_tol/(1-cat_tol),
                    color='k', ls='dashed', lw=0.5)
            ax.text(0.25, 0.9, r"$\rho = %.3f$" % corr,
                    ha='center', va='center',
                    transform=ax.transAxes)

            tick_setting(ax,
                         r"$z_{phot}$",
                         r"$z_{spec}$",
                         lims, lims,
                         ticks, ticks)

            cbar = fig.colorbar(sc)
            if col == 'kde':
                cbar.set_label("Density")
            else:
                cbar.set_label("Discrepancy loss")

            temp_sn = sn+'_'+key+'_'+opt.ecase+'_'+dtype+'_'+ztypes[i]
            fig.savefig(temp_sn+'.png', format='png')
            fig.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
            plt.close('all')

            logging.info("Ensemble scattergram is saved at %s" % temp_sn)


def PCAScattergram(db, dcp_label, dcp_ul, opt, scolor='per'):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = 'PCAScattergram'
    if scolor == 'dcp':
        fn += '_ColDCP'
    sn = os.path.join(dn, fn)

    density_dn = './density'
    if not os.path.exists(density_dn):
        os.makedirs(density_dn)
    pca_in_den_fn = os.path.join(density_dn, 'pca_density_in.npy')
    pca_ul_den_fn = os.path.join(density_dn, 'pca_density_ul.npy')

    data_in = db['eval_in'].dataset.X.numpy()  # [:, ::2]
    data_lo = db['eval_lo'].dataset.X.numpy()  # [:, ::2]
    data_ul = db['eval_ul'].dataset.X.numpy()  # [:, ::2]

    # PCA fitting
    std_pca = make_pipeline(StandardScaler(), PCA(n_components=2))
    std_pca.fit(data_in)
    pca = std_pca.named_steps['pca']
    scaler = std_pca.named_steps['standardscaler']

    coeff = np.transpose(pca.components_)

    pca_exp_var = pca.explained_variance_ratio_
    compre_importance = np.sqrt(np.sum(coeff**2, 1))
    pca_importance = np.hstack((np.arange(len(coeff)).reshape(-1, 1),
                                coeff, compre_importance.reshape(-1, 1)))
    pca_exp_var_fn = os.path.join(opt.quant_fd, 'pca_exp_var.txt')
    pca_feat_imp_fn = os.path.join(opt.quant_fd, 'pca_features_importance.txt')
    np.savetxt(pca_exp_var_fn, pca_exp_var, fmt='%.3f')
    np.savetxt(pca_feat_imp_fn, pca_importance, fmt='%.3f')
    logging.info("PCA explained variance is saved at %s" % pca_exp_var_fn)
    logging.info("PCA feature importance is saved at %s" % pca_feat_imp_fn)

    dcp_in = dcp_label[0][:, 0]
    dcp_lo = dcp_label[1][:, 0]

    dcp_bound = np.hstack((dcp_in, dcp_lo, dcp_ul))
    bound_int = dcp_bound.min()+dcp_bound.max()
    bounds = [dcp_bound.min(), bound_int/3., bound_int*2/3., dcp_bound.max()]

    # data_in, ridx_in = random_under_sampling(data_in)
    # data_lo, ridx_lo = random_under_sampling(data_lo)
    # data_ul, ridx_ul = random_under_sampling(data_ul)

    # dcp_in = dcp_in[ridx_in]
    # dcp_lo = dcp_lo[ridx_lo]
    # dcp_ul = dcp_ul[ridx_ul]

    dcps = [dcp_in, dcp_lo, dcp_ul]
    data = [data_in, data_lo, data_ul]

    pcs = []
    for i, x in enumerate(data):
        pc = pca.transform(scaler.transform(x)).T
        pcs.append(pc)

    XXs, YYs, densities = [], [], []
    if os.path.exists(pca_in_den_fn) and os.path.exists(pca_ul_den_fn):
        logging.info("Load PCA density from %s" % pca_in_den_fn)
        XXin, YYin, den_in = np.load(pca_in_den_fn)
        XXs.append(XXin)
        YYs.append(YYin)
        densities.append(den_in)
        logging.info("Load PCA density from %s" % pca_ul_den_fn)
        XXul, YYul, den_ul = np.load(pca_ul_den_fn)
        XXs.append(XXul)
        YYs.append(YYul)
        densities.append(den_ul)
    else:
        den_fns = [pca_in_den_fn, pca_ul_den_fn]
        indices = [0, 2]
        for i, idx in enumerate(indices):
            xx = np.linspace(pcs[idx][0].min(), pcs[idx][0].max(), 100)
            yy = np.linspace(pcs[idx][1].min(), pcs[idx][1].max(), 100)
            XX, YY = np.meshgrid(xx, yy)

            positions = np.vstack([XX.ravel(), YY.ravel()])

            randidx = np.random.choice(np.arange(pcs[idx].shape[1]), 20000)
            temp_pc = pcs[idx][:, randidx]

            density = np.reshape(kde_scipy(temp_pc, positions).T, XX.shape)

            np.save(den_fns[i], [XX, YY, density])
            logging.info("PCA density is saved at %s" % den_fns[i])

            XXs.append(XX)
            YYs.append(YY)
            densities.append(density)

    titles = ["In-Distribution", "Labeled Out-of-Distribution", "Unlabeled"]
    if opt.psc:
        titles[-1] += " (LPSC)" if opt.psc_reg == 'low' else " (HPSC)"
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, axes = plt.subplots(1, 3, figsize=(9, 4))
        for i, pc in enumerate(pcs):
            if scolor == 'per':
                cmap = matplotlib.colors.ListedColormap(
                    ['blue', 'green', 'red'])
                assert cmap.N == len(bounds)-1
                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                #  count_samples
                nsamples = [np.sum((bounds[j] <= dcps[i]) &
                            (bounds[j+1] > dcps[i]))
                            for j in range(cmap.N)]
                nsamples = np.array(nsamples[::-1])/np.sum(nsamples)
                # colors = ['r', 'g', 'b']
                # legend_handles = []
                # for k in range(cmap.N):
                #     handle = mlines.Line2D([], [], color=colors[k],
                #                            marker='.', linestyle='None',
                #                            markersize=5,
                #                            label="R = %.2f" % nsamples[k])
                #     legend_handles.append(handle)
            else:
                cmap = plt.cm.get_cmap('jet')
                norm = None

            scalex = 1.0/(pc[0].max()-pc[0].min())
            scaley = 1.0/(pc[1].max()-pc[1].min())

            sf = axes[i].scatter(pc[0]*scalex, pc[1]*scaley,
                                 c=dcps[i], marker='.',
                                 alpha=0.5, s=0.5, cmap=cmap,
                                 linewidth=0.3, norm=norm,
                                 edgecolor=None,
                                 vmin=dcps[i].min(),
                                 vmax=dcps[i].max())
            den_int = (densities[0].max()-densities[0].min())/10.
            axes[i].contour(XXs[0]*scalex, YYs[0]*scaley, densities[0],
                            np.linspace(densities[0].min()+den_int,
                                        densities[0].max(), 8),
                            colors='k', linewidths=1)
            # if i == 0:
            #     headers = ['c', 'ce', 'd', 'de']
            #     numb = ['1', '2', '3', '4']
            #     for ei in range(coeff.shape[0]):
            #         if ei % 4 == 0:
            #             axes[i].arrow(0, 0, coeff[ei, 0], coeff[ei, 1],
            #                           color='magenta', alpha=0.5, overhang=0.2)
            #             if ei == coeff.shape[0]-1:
            #                 lab = 'ebv'
            #             else:
            #                 lab = headers[ei % 4]+numb[ei // 4]
            #             axes[i].text(coeff[ei, 0]*1.15, coeff[ei, 1]*1.15,
            #                          lab, color='magenta',
            #                          ha='center', va='center')
            axes[i].set_title(titles[i])
            axes[i].set_ylim(-0.4, 0.4)
            axes[i].set_xlim(-0.5, 0.5)
            # if scolor == 'per':
            #     axes[i].legend(loc='upper right', handles=legend_handles)
            if i == 0:
                axes[i].set_xlabel('PC1 (%.2f)' % pca_exp_var[0])
                axes[i].set_ylabel('PC2 (%.2f)' % pca_exp_var[1])
            else:
                axes[i].set_xlabel('PC1' % pca_exp_var[0])

            if i == 2:
                den_int = (densities[1].max()-densities[1].min())/10.
                axes[i].contour(XXs[1]*scalex, YYs[1]*scaley, densities[1],
                                np.linspace(densities[1].min()+den_int,
                                            densities[1].max(), 8),
                                colors='w', linewidths=0.5)

        fig.subplots_adjust(left=0.2,
                            right=0.8,
                            bottom=0.1,
                            top=0.9,
                            wspace=0.2,
                            hspace=0.3)
        cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
        if scolor == 'per':
            cbar = fig.colorbar(sf, cax=cbar_ax, ticks=bounds)
        else:
            cbar = fig.colorbar(sf, cax=cbar_ax)
        cbar.set_label("Discrepancy loss")
        fig.savefig(sn)
        fig.savefig(sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("PCA scattergram is saved at %s" % sn)


def SpecRedshiftColorScattergram(db, dcp_label, dcp_ul, opt):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = 'SpecRedshiftColorScattergram'
    sn = os.path.join(dn, fn)

    key = 'spec'
    density_dn = './density'
    if not os.path.exists(density_dn):
        os.makedirs(density_dn)
    zc_in_den_fn = os.path.join(density_dn, 'ZspecC_density_in%s.npy')
    zc_lo_den_fn = os.path.join(density_dn, 'ZspecC_density_lo%s.npy')

    data_in = db['eval_in'].dataset.X.numpy()  # [:, ::2]
    data_lo = db['eval_lo'].dataset.X.numpy()  # [:, ::2]

    data_in = db['eval_in'].dataset.unnorm(data_in)
    data_lo = db['eval_lo'].dataset.unnorm(data_lo)

    dcp_in = dcp_label[0][:, 0]
    dcp_lo = dcp_label[1][:, 0]

    dcp_bound = np.hstack((dcp_in, dcp_lo, dcp_ul))
    bound_int = dcp_bound.min()+dcp_bound.max()
    bounds = [dcp_bound.min(), bound_int/3., bound_int*2/3., dcp_bound.max()]

    # data_in, ridx_in = random_under_sampling(data_in)
    # data_lo, ridx_lo = random_under_sampling(data_lo)

    # dcp_in = dcp_in[ridx_in]
    # dcp_lo = dcp_lo[ridx_lo]

    specz_in = db['eval_in'].dataset.z.numpy().ravel()  # [ridx_in]
    specz_lo = db['eval_lo'].dataset.z.numpy().ravel()  # [ridx_lo]

    dcps = [dcp_in, dcp_lo]
    dset = [data_in, data_lo]
    zset = [specz_in, specz_lo]

    indices = [0]
    color_idx = [0]  #, 4, 8, 12, 16]
    XXs, YYs, densities = [], [], []
    den_fns = [zc_in_den_fn, zc_lo_den_fn]
    for ci, cidx in enumerate(color_idx):
        for i, idx in enumerate(indices):
            temp_den_fn = den_fns[i] % str(ci)

            if os.path.exists(temp_den_fn):
                logging.info("Load ZspecC density from %s" % temp_den_fn)
                XX, YY, density = np.load(temp_den_fn)
                XXs.append(XX)
                YYs.append(YY)
                densities.append(density)
            else:
                xx = np.linspace(dset[idx][:, cidx].min(),
                                 dset[idx][:, cidx].max(), 100)
                yy = np.linspace(zset[idx].min(),
                                 zset[idx].max(), 100)
                XX, YY = np.meshgrid(xx, yy)

                positions = np.vstack([XX.ravel(), YY.ravel()])

                cz = np.vstack((dset[idx][:, cidx], zset[idx]))
                randidx = np.random.choice(np.arange(cz.shape[1]), 20000)
                temp_cz = cz[:, randidx]

                density = np.reshape(kde_scipy(temp_cz, positions).T, XX.shape)

                np.save(temp_den_fn, [XX, YY, density])
                logging.info("Redshift color density is saved at %s" %
                             temp_den_fn)

                XXs.append(XX)
                YYs.append(YY)
                densities.append(density)

    ylabels = [r"$z_{spec}$", r"$z_{spec}$"]
    xlabels = [r"$(g-r)_{c}$", r"$(r-i)_{c}$",
               r"$(i-z)_{c}$", r"$(z-y)_{c}$", r"$E(B-V)$"]

    titles = ["In-Distribution", "Labeled Out-of-Distribution"]
    prefix = ["_ID", "_LOOD"]
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        cmap = matplotlib.colors.ListedColormap(['blue', 'green', 'red'])
        assert cmap.N == len(bounds)-1
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        for ci, cidx in enumerate(color_idx):
            for i in range(len(dset)):
                fig, ax = plt.subplots()
                #  count_samples
                nsamples = [np.sum((bounds[j] <= dcps[i]) &
                            (bounds[j+1] > dcps[i]))
                            for j in range(cmap.N)]
                nsamples = np.array(nsamples[::-1])/np.sum(nsamples)
                colors = ['r', 'g', 'b']
                # legend_handles = []
                # for k in range(cmap.N):
                #     handle = mlines.Line2D([], [], color=colors[k],
                #                            marker='.', linestyle='None',
                #                            markersize=5,
                #                            label="R = %.2f" % nsamples[k])
                #     legend_handles.append(handle)

                sf = ax.scatter(dset[i][:, cidx],
                                zset[i],
                                c=dcps[i], marker='.', alpha=0.5,
                                s=0.5, cmap=cmap, linewidth=0.3,
                                norm=norm, edgecolor=None)
                den_int = (densities[ci].max()-densities[ci].min())/10.
                ax.contour(XXs[ci], YYs[ci], densities[ci],
                           np.linspace(densities[ci].min()+den_int,
                                       densities[ci].max(), 8),
                           colors='k', linewidths=0.5)
                # ax.legend(loc='upper right', handles=legend_handles)
                ax.set_xlabel(xlabels[ci])
                # if i == 0:
                ax.set_ylabel(ylabels[i])
                ax.set_title(titles[i])

                ax.set_ylim(0, 1)
                ax.set_xlim(-1, 2.5)

            # fig.subplots_adjust(left=0.2,
            #                     right=0.8,
            #                     bottom=0.1,
            #                     top=0.9,
            #                     wspace=0.1,
            #                     hspace=0.3)
            # cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
                cbar = fig.colorbar(sf, ticks=bounds)
                cbar.set_label("Discrepancy loss")
                temp_sn = sn+prefix[i]+str(ci)
                fig.savefig(temp_sn)
                fig.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
                plt.close('all')
                logging.info("Z%sC scattergram is saved at %s" % (key, temp_sn))


def PhotRedshiftColorScattergram(db, dcp_label, dcp_ul,
                                 photz_in, photz_lo, photz_ul,
                                 opt, model='universal'):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = '%sRedshiftColorScattergram_%s%s%s'
    sn = os.path.join(dn, fn)

    density_dn = './density'
    if not os.path.exists(density_dn):
        os.makedirs(density_dn)
    zc_den_fn = os.path.join(density_dn, 'Z%sC_density_%s_in%s.npy')

    data_in = db['eval_in'].dataset.X.numpy()  # [:, ::2]
    data_lo = db['eval_lo'].dataset.X.numpy()  # [:, ::2]
    data_ul = db['eval_ul'].dataset.X.numpy()  # [:, ::2]

    data_in = db['eval_in'].dataset.unnorm(data_in)
    data_lo = db['eval_lo'].dataset.unnorm(data_lo)
    data_ul = db['eval_ul'].dataset.unnorm(data_ul)

    dcp_in = dcp_label[0][:, 0]
    dcp_lo = dcp_label[1][:, 0]

    dcp_bound = np.hstack((dcp_in, dcp_lo, dcp_ul))
    bound_int = dcp_bound.min()+dcp_bound.max()
    bounds = [dcp_bound.min(), bound_int/3., bound_int*2/3., dcp_bound.max()]

    # data_in, ridx_in = random_under_sampling(data_in)
    # data_lo, ridx_lo = random_under_sampling(data_lo)
    # data_ul, ridx_ul = random_under_sampling(data_ul)

    # dcp_in = dcp_in[ridx_in]
    # dcp_lo = dcp_lo[ridx_lo]
    # dcp_ul = dcp_ul[ridx_ul]

    dcps = [dcp_in, dcp_lo, dcp_ul]
    dset = [data_in, data_lo, data_ul]
    midx = 0 if model == 'universal' else 1

    # zavg = [photz_in[0][midx][ridx_in],
    #         photz_lo[0][midx][ridx_lo],
    #         photz_ul[0][midx][ridx_ul]]
    # zmode = [photz_in[1][midx][ridx_in],
    #          photz_lo[1][midx][ridx_lo],
    #          photz_ul[1][midx][ridx_ul]]
    zavg = [photz_in[0][midx],
            photz_lo[0][midx],
            photz_ul[0][midx]]
    # zmode = [photz_in[1][midx],
    #          photz_lo[1][midx],
    #          photz_ul[1][midx]]
    zset = [zavg] # , zmode]
    estim = ['avg'] #, 'mode']

    color_idx = [0] #, 4, 8, 12, 16]
    XXs, YYs, densities = [[], []], [[], []], [[], []]
    for i, z in enumerate(zset):
        for ci, cidx in enumerate(color_idx):
            temp_zc_den_fn = zc_den_fn % (estim[i], model, str(ci))

            if os.path.exists(temp_zc_den_fn):
                logging.info("Load ZphotC density of the %s model from %s" %
                             (model, temp_zc_den_fn))
                XX, YY, density = np.load(temp_zc_den_fn)
            else:
                xx = np.linspace(dset[0][:, cidx].min(),
                                 dset[0][:, cidx].max(), 100)
                yy = np.linspace(z[0].min(),
                                 z[0].max(), 100)
                XX, YY = np.meshgrid(xx, yy)

                positions = np.vstack([XX.ravel(), YY.ravel()])

                cz = np.vstack((dset[0][:, cidx], z[0]))
                randidx = np.random.choice(np.arange(cz.shape[1]), 20000)
                temp_cz = cz[:, randidx]

                density = np.reshape(kde_scipy(temp_cz, positions).T, XX.shape)

                np.save(temp_zc_den_fn, [XX, YY, density])
                logging.info("ZphotC density is saved at %s" %
                             temp_zc_den_fn)

            XXs[i].append(XX)
            YYs[i].append(YY)
            densities[i].append(density)

    ylabels = [r"$z_{phot}$", r"$z_{mode}$"]
    xlabels = [r"$(g-r)_{c}$", r"$(r-i)_{c}$",
               r"$(i-z)_{c}$", r"$(z-y)_{c}$", r"$E(B-V)$"]
    titles = ["In-Distribution", "Labeled Out-of-Distribution", "Unlabeled"]
    prefix = ["_ID", "_LOOD", "_UL"]
    if opt.psc:
        titles[-1] += " (LPSC)" if opt.psc_reg == 'low' else " (HPSC)"
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        cmap = matplotlib.colors.ListedColormap(['blue', 'green', 'red'])
        assert cmap.N == len(bounds)-1
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        for i, z in enumerate(zset):
            for ci, cidx in enumerate(color_idx):
                for j in range(len(dset)):
                    fig, ax = plt.subplots()
                    #  count_samples
                    nsamples = [np.sum((bounds[k] <= dcps[j]) &
                                (bounds[k+1] > dcps[j]))
                                for k in range(cmap.N)]
                    nsamples = np.array(nsamples[::-1])/np.sum(nsamples)
                    colors = ['r', 'g', 'b']
                    # legend_handles = []
                    # for k in range(cmap.N):
                    #     handle = mlines.Line2D([], [], color=colors[k],
                    #                            marker='.', linestyle='None',
                    #                            markersize=5,
                    #                            label="R = %.2f" % nsamples[k])
                    #     legend_handles.append(handle)

                    sf = ax.scatter(dset[j][:, cidx], z[j],
                                    c=dcps[j], marker='.',
                                    alpha=0.5, s=0.5, cmap=cmap,
                                    linewidth=0.3, norm=norm,
                                    edgecolor=None)
                    den_int = \
                        (densities[i][ci].max()-densities[i][ci].min())/10.
                    ax.contour(XXs[i][ci], YYs[i][ci], densities[i][ci],
                               np.linspace(densities[i][ci].min()+den_int,
                                           densities[i][ci].max(), 8),
                               colors='k', linewidths=0.5)
                    # ax.legend(loc='best', handles=legend_handles)
                    ax.set_xlabel(xlabels[ci])
                    # if j == 0:
                    ax.set_ylabel(ylabels[i])
                    ax.set_title(titles[j])
                    ax.set_ylim(0, 1.1)
                    ax.set_xlim(-3, 3)

                    # fig.subplots_adjust(left=0.2,
                    #                     right=0.8,
                    #                     bottom=0.1,
                    #                     top=0.9,
                    #                     wspace=0.2,
                    #                     hspace=0.3)
                    # cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
                    cbar = fig.colorbar(sf, ticks=bounds)
                    cbar.set_label("Discrepancy loss")
                    temp_sn = sn % (estim[i], model, prefix[j], str(ci))
                    fig.savefig(temp_sn)
                    fig.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
                    plt.close('all')
                    logging.info("Z%sC scattergram of %s is saved at %s" %
                                 (estim[i], model, temp_sn))


def ProbDensityDistribution(probs, binc, dcp_label,
                            dcp_ul, opt, model='universal'):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = 'ProbabilityDistribution_%s' % model
    sn = os.path.join(dn, fn)

    dcp_in = dcp_label[0][:, 0]
    dcp_lo = dcp_label[1][:, 0]

    dcp_bound = np.hstack((dcp_in, dcp_lo, dcp_ul))
    bound_int = dcp_bound.min()+dcp_bound.max()
    bounds = [dcp_bound.min(), bound_int/3., bound_int*2/3., dcp_bound.max()]

    dx = binc[1]-binc[0]
    titles = ["In-Distribution", "Labeled Out-of-Distribution", "Unlabeled"]
    if opt.psc:
        titles[-1] += " (LPSC)" if opt.psc_reg == 'low' else " (HPSC)"
    dcps = [dcp_in, dcp_lo, dcp_ul]
    prefix = ['_ID', '_LOOD', '_UL']
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        for i in range(len(dcps)):
            fig, ax = plt.subplots(1)
            masks = [(bounds[j] <= dcps[i]) & (bounds[j+1] > dcps[i]) for j in range(len(bounds)-1)]

            probs_pred_id = np.mean(probs[i][masks[0]], 0)
            probs_pred_aside = np.mean(probs[i][masks[1]], 0)
            probs_pred_ood = np.mean(probs[i][masks[2]], 0)

            area_ood = np.trapz(probs_pred_ood, dx=dx)
            area_aside = np.trapz(probs_pred_aside, dx=dx)
            area_id = np.trapz(probs_pred_id, dx=dx)

            ax.plot(binc, probs_pred_id/area_id,
                    color='b', label=r"$ID_{pred}$")
            ax.plot(binc, probs_pred_aside/area_aside,
                    color='g', label=r"$PEND_{pred}$")
            ax.plot(binc, probs_pred_ood/area_ood,
                    color='r', label=r"$OOD_{pred}$")

            # ax.set_title(titles[i])
            ax.set_xlabel("Redshift")
            ax.legend(loc='upper right')
            ax.set_xlim(0, 1)
            # if i == 0:
            ax.set_ylabel("Sample mean probability density")

            fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

            temp_sn = sn+prefix[i]
            fig.savefig(temp_sn)
            fig.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
            plt.close('all')
            logging.info("Probability distribution of %s model is saved at %s" %
                         (model, temp_sn))


def AE_ProbDensityDistribution(probs, binc, dcp_label,
                               dcp_ul, opt, key='ensemble'):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = 'ProbabilityDistribution_'+key
    sn = os.path.join(dn, fn)

    dcp_in = dcp_label[0][:, 0]
    dcp_lo = dcp_label[1][:, 0]

    dcp_bound = np.hstack((dcp_in, dcp_lo, dcp_ul))
    bound_int = dcp_bound.min()+dcp_bound.max()
    bounds = [dcp_bound.min(), bound_int/3., bound_int*2/3., dcp_bound.max()]

    dx = binc[1]-binc[0]
    dcps = [dcp_in, dcp_lo, dcp_ul]
    prefix = ['_ID', '_LOOD', '_UL']
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        for i in range(len(dcps)):
            fig, ax = plt.subplots(1)
            masks = [(bounds[j] <= dcps[i]) & (bounds[j+1] > dcps[i])
                     for j in range(len(bounds)-1)]

            probs_pred_id = np.mean(probs[i][masks[0]], 0)
            probs_pred_aside = np.mean(probs[i][masks[1]], 0)
            probs_pred_ood = np.mean(probs[i][masks[2]], 0)

            area_ood = np.trapz(probs_pred_ood, dx=dx)
            area_aside = np.trapz(probs_pred_aside, dx=dx)
            area_id = np.trapz(probs_pred_id, dx=dx)

            ax.plot(binc, probs_pred_id/area_id,
                    color='b', label=r"$ID_{pred}$")
            ax.plot(binc, probs_pred_aside/area_aside,
                    color='g', label=r"$PEND_{pred}$")
            ax.plot(binc, probs_pred_ood/area_ood,
                    color='r', label=r"$OOD_{pred}$")

            ax.set_xlabel("Redshift")
            ax.legend(loc='upper right')
            ax.set_xlim(0, 1)
            ax.set_ylabel("Sample mean probability density")

            fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

            temp_sn = sn+prefix[i]
            fig.savefig(temp_sn)
            fig.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
            plt.close('all')
            logging.info(
                "Probability distribution of ensemble/" +
                "average model is saved at %s" %
                temp_sn)


def ConfusionMatrix(y, prd_y, opt):
    fn = 'ConfusionMatrix'
    if not os.path.exists(opt.plot_fd):
        os.makedirs(opt.plot_fd)
    sn = os.path.join(opt.plot_fd, fn)

    title = 'Confusion Matrix'
    cmap = plt.cm.Blues

    y = np.array(y, dtype=np.long)
    prd_y = np.array(prd_y, dtype=np.long)

    xtick_marks = ["ID", "OOD"]
    ytick_marks = ["OOD", "ID"]

    nclss = 2
    cm = np.zeros((nclss, nclss), dtype=int)
    for p, l in zip(y, prd_y):
        cm[p, l] += 1

    cmr = np.zeros(np.shape(cm))
    for i, row in enumerate(cm):
        row_sum = float(np.sum(row))
        cmr[i] = row/row_sum*100

    permutation = [1, 0]
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))

    cm = cm[idx, :]
    cmr = cmr[idx, :]

    tick_numbs = np.arange(cm.shape[1])
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        plt.figure()
        plt.cla()
        plt.imshow(cmr, interpolation='nearest', cmap=cmap)
        cbar = plt.colorbar()
        cbar.set_label("Accuracy")
        ax = plt.gca()
        ax.set_xticklabels((ax.get_xticks()+1).astype(str))
        plt.xticks(tick_numbs, xtick_marks)
        plt.yticks(tick_numbs, ytick_marks,
                   rotation=90, va='center')

        thresh = cmr.max() / 2.

        for i, j in itertools.product(range(cmr.shape[0]), range(cmr.shape[1])[::-1]):
            plt.text(j, i, "%s\n(%.2f%%)" % (cm[i, j], cmr[i, j]),
                     horizontalalignment='center', verticalalignment='center',
                     color='white' if cmr[i, j] > thresh else 'black')
        plt.ylim(-.5, 1.5)
        plt.clim(0, 100)
        plt.title(title)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.xticks([])
        plt.yticks([])
        plt.savefig(sn+'.png', bbox_inches='tight', format='png')
        plt.savefig(sn+'.pdf', bbox_inches='tight', format='pdf')
        plt.close('all')
        logging.info("Confusion matrix is saved at %s" % sn)


def LossVariation(tcls, tdcp, vdcp, opt):
    fn = 'LossVariation'
    if not os.path.exists(opt.plot_fd):
        os.makedirs(opt.plot_fd)
    sn = os.path.join(opt.plot_fd, fn)

    tepoch_intv = opt.pevery
    vepoch_intv = opt.vevery

    spec_cls = tcls[1]
    tdcp_in, tdcp_ul = tdcp[0], tdcp[1]
    vdcp_in, vdcp_ul = vdcp[0], vdcp[1]

    tepoch = np.arange(1, len(spec_cls)+1)*tepoch_intv
    vepoch = np.arange(1, len(vdcp_in)+1)*vepoch_intv

    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()
        lp1, = ax.plot(tepoch, spec_cls,
                       color=color_scheme[0],
                       label=r"$train_{cls}^{ID}$",
                       linestyle='solid')
        lp2, = ax.plot(tepoch[-len(tdcp_in):], tdcp_in,
                       color=color_scheme[1],
                       label=r"$train_{dcp}^{ID}$",
                       linestyle='solid')
        lp3, = ax.plot(tepoch[-len(tdcp_ul):], tdcp_ul,
                       color=color_scheme[2],
                       label=r"$train_{dcp}^{UL}$",
                       linestyle='solid')
        lp4, = ax.plot(vepoch, vdcp_in,
                       color=color_scheme[3],
                       label=r"$val_{dcp}^{ID}$",
                       linestyle='solid')
        lp5, = ax.plot(vepoch, vdcp_ul,
                       color=color_scheme[4],
                       label=r"$val_{dcp}^{UL}$",
                       linestyle='solid')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        handles = [lp1, lp2, lp3, lp4, lp5]
        ax.legend(loc='best')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        # fig.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.35)
        plt.legend(handles=handles,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left')

        fig.savefig(sn)
        fig.savefig(sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')


def ColorColorContour(db, dcp_label, dcp_ul, opt, color=True):
    lim_eps = 0.05

    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = 'ColorColorContour'
    fn += '' if color else '_err'
    sn = os.path.join(dn, fn)
    density_dn = './density'
    density_dn += '' if color else '_err'
    if not os.path.exists(density_dn):
        os.makedirs(density_dn)

    color_id = db['eval_in'].dataset.X.numpy().T
    color_lo = db['eval_lo'].dataset.X.numpy().T
    color_ul = db['eval_ul'].dataset.X.numpy().T

    dcp_id = dcp_label[0][:, 0]
    dcp_lo = dcp_label[1][:, 0]

    dcp = np.hstack((dcp_id, dcp_lo, dcp_ul))
    bound_int = dcp.min()+dcp.max()
    bounds = [dcp.min(), bound_int/3., bound_int*2/3., dcp.max()][::-1]

    dcps = [dcp_id, dcp_lo, dcp_ul]

    column_header = [r"$(g-r)_", r"$(r-i)_",
                     r"$(i-z)_", r"$(z-y)_", r"$E(B-V)$"]
    column_tail = [r"{c}$", r"{d}$"] if color else [r"{ce}$", r"{de}$"]

    start_i = 0 if color else 1
    iterators = np.arange(start_i, len(color_id)-2, 4)
    titles = ["In-Distribution", "Labeled Out-of-Distribution", "Unlabeled"]
    if opt.psc:
        titles[-1] += " (LPSC)" if opt.psc_reg == 'low' else " (HPSC)"
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        cmap = matplotlib.colors.ListedColormap(['blue', 'green', 'red'])
        for l, i in enumerate(iterators):
            if color:
                if i == iterators[-1]:
                    si = -1
                    i = 0
                else:
                    si = 4
            else:
                si = 2
            fig, axes = plt.subplots(1, 3, figsize=(9, 4))
            ci = [color_id[i], color_id[i+si]]
            cl = [color_lo[i], color_lo[i+si]]
            cu = [color_ul[i], color_ul[i+si]]
            colors = [ci, cl, cu]

            den_id_fn = 'density_id_%s.npy' % str(l).zfill(2)
            den_lo_fn = 'density_lo_%s.npy' % str(l).zfill(2)
            den_id_fn = os.path.join(density_dn, den_id_fn)
            den_lo_fn = os.path.join(density_dn, den_lo_fn)
            if os.path.exists(den_id_fn) and os.path.exists(den_lo_fn):
                logging.info("Load ID density from %s" % den_id_fn)
                logging.info("Load LO density from %s" % den_lo_fn)
                XX_id, YY_id, density_id = np.load(den_id_fn)
                XX_lo, YY_lo, density_lo = np.load(den_lo_fn)
                XXs = [XX_id, XX_lo]
                YYs = [YY_id, YY_lo]
                densities = [density_id, density_lo]
            else:
                XXs, YYs = [], []
                densities = []
                den_fn = [den_id_fn, den_lo_fn]
                for k, col in enumerate(colors[:-1]):
                    xx = np.linspace(col[0].min(), col[0].max(), 100)
                    yy = np.linspace(col[1].min(), col[1].max(), 100)
                    XX, YY = np.meshgrid(xx, yy)
                    XXs.append(XX)
                    YYs.append(YY)
                    positions = np.vstack([XX.ravel(), YY.ravel()])

                    cc = np.vstack([col[0], col[1]])
                    randidx = np.random.choice(np.arange(cc.shape[1]), 20000)
                    cc = cc[:, randidx]

                    density = np.reshape(kde_scipy(cc, positions).T, XX.shape)
                    densities.append(density)

                    np.save(den_fn[k], [XX, YY, density])

                    logging.info("Density is saved at %s" % den_fn[k])

            for j, (c1, c2) in enumerate(colors):
                dcp = dcps[j]

                # legend_handles = []
                scolors = ['r', 'g', 'b']
                xllim, xulim = [], []
                yllim, yulim = [], []
                for k in range(cmap.N):
                    bounded_idx = (bounds[k] > dcp) & (bounds[k+1] <= dcp)
                    nsample = np.sum(bounded_idx)
                    # rsample = nsample/len(dcp)

                    # handle = mlines.Line2D([], [], color=scolors[k],
                    #                        marker='.', linestyle='None',
                    #                        markersize=5)
                    # legend_handles.append(handle)

                    if j != 0 and nsample >= 10000:
                        temp_c1 = c1[bounded_idx]
                        temp_c2 = c2[bounded_idx]

                        cxx = np.linspace(temp_c1.min(), temp_c1.max(), 100)
                        cyy = np.linspace(temp_c2.min(), temp_c2.max(), 100)
                        cXX, cYY = np.meshgrid(cxx, cyy)

                        cpositions = np.vstack([cXX.ravel(), cYY.ravel()])

                        ccc = np.vstack([temp_c1, temp_c2])
                        randidx = np.random.choice(
                            np.arange(ccc.shape[1]), 10000)
                        ccc = ccc[:, randidx]

                        cdensity = np.reshape(
                            kde_scipy(ccc, cpositions).T, cXX.shape)

                        cden_int = (cdensity.max()-cdensity.min())/10.
                        axes[j].contour(cXX, cYY,
                                        ndimage.gaussian_filter(
                                            cdensity,
                                            sigma=1.0,
                                            order=0),
                                        np.linspace(cdensity.min()+cden_int,
                                                    cdensity.max(), 8),
                                        colors=scolors[k], linewidths=0.5)

                        xllim.append(cXX.min())
                        xulim.append(cXX.max())
                        yllim.append(cYY.min())
                        yulim.append(cYY.max())

                    den_int = (densities[0].max()-densities[0].min())/10.
                    axes[j].contour(XXs[0], YYs[0],
                                    ndimage.gaussian_filter(
                                        densities[0],
                                        sigma=1.0,
                                        order=0),
                                    np.linspace(densities[0].min()+den_int,
                                                densities[0].max(), 8),
                                    colors='k', linewidths=0.5)
                    xllim.append(XXs[0].min())
                    xulim.append(XXs[0].max())
                    yllim.append(YYs[0].min())
                    yulim.append(YYs[0].max())

                if si == -1:
                    axes[j].set_xlabel(column_header[i]+column_tail[0])
                    axes[j].set_xlim(min(xllim)-lim_eps, max(xulim)+lim_eps)
                    axes[j].set_ylim(min(yllim)-lim_eps, max(yulim)+lim_eps)
                else:
                    axes[j].set_xlabel(column_header[i//4]+column_tail[0])
                    if j == 2:
                        axes[j].set_xlim(axes[0].get_xlim())
                        axes[j].set_ylim(axes[0].get_ylim())
                    else:
                        axes[j].set_xlim(min(xllim)-lim_eps, max(xulim)+lim_eps)
                        axes[j].set_ylim(min(yllim)-lim_eps, max(yulim)+lim_eps)

                # axes[j].legend(loc='best', handles=legend_handles)
                axes[j].set_title(titles[j])

                if j == 0:
                    if si == -1:
                        axes[j].set_ylabel(column_header[i+si])
                    else:
                        axes[j].set_ylabel(column_header[i//4+1]+column_tail[0])

            fig.subplots_adjust(left=0.2,
                                right=0.8,
                                bottom=0.1,
                                top=0.9,
                                wspace=0.25,
                                hspace=0.3)
            # cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
            # cbar = fig.colorbar(sf, cax=cbar_ax, ticks=bounds)
            # cbar.set_label("Discrepancy loss")
            # cbar.ax.tick_params(axis='y', which='both', direction='out')

            # fig.tight_layout()
            temp_sn = sn+str(l).zfill(2)
            plt.savefig(temp_sn)
            plt.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
            plt.close('all')
            logging.info("Color-Color diagram is saved at %s" % temp_sn)


def ColorColorDensity(db, dcp_label, dcp_ul, opt, color=True):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = 'ColorColorDensity'
    fn += '' if color else '_err'
    sn = os.path.join(dn, fn)
    density_dn = './density'
    density_dn += '' if color else '_err'
    if not os.path.exists(density_dn):
        os.makedirs(density_dn)

    color_id = db['eval_in'].dataset.X.numpy()
    color_lo = db['eval_lo'].dataset.X.numpy()
    color_ul = db['eval_ul'].dataset.X.numpy()

    color_id = db['eval_in'].dataset.unnorm(color_id).T
    color_lo = db['eval_lo'].dataset.unnorm(color_lo).T
    color_ul = db['eval_ul'].dataset.unnorm(color_ul).T

    dcp_id = dcp_label[0][:, 0]
    dcp_lo = dcp_label[1][:, 0]

    dcp = np.hstack((dcp_id, dcp_lo, dcp_ul))
    bound_int = dcp.min()+dcp.max()
    bounds = [dcp.min(), bound_int/3., bound_int*2/3., dcp.max()]

    dcps = [dcp_id, dcp_lo, dcp_ul]

    column_header = [r"$(g-r)$", r"$(r-i)$",
                     r"$(i-z)_", r"$(z-y)_", r"$E(B-V)$"]
    column_tail = ['', r"{d}$"] if color else [r"{ce}$", r"{de}$"]

    start_i = 0 if color else 1
    iterators = np.arange(start_i, len(color_id)-2, 4)
    fprefix = ["LOOD", "UL"]
    titles = [r"Low $L_{DCP}$", r"Middle $L_{DCP}$", r"High $L_{DCP}$"]
    if opt.psc:
        titles[-1] += " (LPSC)" if opt.psc_reg == 'low' else " (HPSC)"
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        cmap = truncate_colormap(plt.cm.get_cmap('Spectral_r'), 0.5, 1.0)
        norm = matplotlib.colors.BoundaryNorm(
            np.linspace(0, 100, 256), 255)
        for l, i in enumerate(iterators):
            if color:
                if i == iterators[-1]:
                    si = -1
                    i = 0
                else:
                    si = 4
            else:
                si = 2
            ci = [color_id[i], color_id[i+si]]
            cl = [color_lo[i], color_lo[i+si]]
            cu = [color_ul[i], color_ul[i+si]]
            colors = [ci, cl, cu]

            den_id_fn = 'density_id_%s.npy' % str(l).zfill(2)
            den_lo_fn = 'density_lo_%s.npy' % str(l).zfill(2)
            den_id_fn = os.path.join(density_dn, den_id_fn)
            den_lo_fn = os.path.join(density_dn, den_lo_fn)
            if os.path.exists(den_id_fn) and os.path.exists(den_lo_fn):
                logging.info("Load ID density from %s" % den_id_fn)
                logging.info("Load LO density from %s" % den_lo_fn)
                XX_id, YY_id, density_id = np.load(den_id_fn)
                XX_lo, YY_lo, density_lo = np.load(den_lo_fn)
                XXs = [XX_id, XX_lo]
                YYs = [YY_id, YY_lo]
                densities = [density_id, density_lo]
            else:
                XXs, YYs = [], []
                densities = []
                den_fn = [den_id_fn, den_lo_fn]
                for k, col in enumerate(colors[:-1]):
                    xx = np.linspace(col[0].min(), col[0].max(), 100)
                    yy = np.linspace(col[1].min(), col[1].max(), 100)
                    XX, YY = np.meshgrid(xx, yy)
                    XXs.append(XX)
                    YYs.append(YY)
                    positions = np.vstack([XX.ravel(), YY.ravel()])

                    cc = np.vstack([col[0], col[1]])
                    randidx = np.random.choice(np.arange(cc.shape[1]), 40000)
                    cc = cc[:, randidx]

                    density = np.reshape(kde_scipy(cc, positions).T, XX.shape)
                    densities.append(density)

                    np.save(den_fn[k], [XX, YY, density])

                    logging.info("Density is saved at %s" % den_fn[k])

            lim_tol = densities[0].max()/10
            coor_x = XXs[0].ravel()
            coor_y = YYs[0].ravel()
            coor_mask = densities[0].ravel() > lim_tol
            show_x = coor_x[coor_mask]
            show_y = coor_y[coor_mask]

            for j, (c1, c2) in enumerate(colors[1:]):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                dcp = dcps[j+1]

                cpositions = np.vstack([XXs[0].ravel(), YYs[0].ravel()])
                comb_ccc = np.vstack([c1, c2])
                comb_ridx = np.random.choice(
                    np.arange(comb_ccc.shape[1]), 20000)
                comb_ccc = comb_ccc[:, comb_ridx]

                comb_cden = np.reshape(
                    kde_scipy(comb_ccc, cpositions).T, XXs[0].shape)

                for k in range(3):
                    bounded_idx = (bounds[k] <= dcp) & (bounds[k+1] > dcp)

                    temp_c1 = c1[bounded_idx]
                    temp_c2 = c2[bounded_idx]

                    ccc = np.vstack([temp_c1, temp_c2])
                    randidx = np.random.choice(
                        np.arange(ccc.shape[1]), 20000)
                    ccc = ccc[:, randidx]

                    cdensity = np.reshape(
                        kde_scipy(ccc, cpositions).T, XXs[0].shape)

                    # sf = axes[k].imshow(norm_cdensity,
                    #                     extent=(np.amin(XXs[0]),
                    #                             np.amax(XXs[0]),
                    #                             np.amin(YYs[0]),
                    #                             np.amax(YYs[0])),
                    #                     interpolation='spline36',
                    #                     origin='lower',
                    #                     aspect='auto',
                    #                     cmap=cmap)

                    temp_density = densities[0]/densities[0].max() * 100
                    sf = axes[k].tricontourf(XXs[0].ravel(), YYs[0].ravel(),
                                             temp_density.ravel(),
                                             np.linspace(0, 100, 17),
                                             cmap=cmap, norm=norm)

                    # den_int = (cdensity.max()-cdensity.min())/6.
                    axes[k].contour(XXs[0],
                                    YYs[0],
                                    cdensity,
                                    colors='k', linewidths=1,
                                    levels=np.linspace(cdensity.min(),
                                                       cdensity.max(), 10))
                                    # ndimage.gaussian_filter(
                                    #               cdensity,
                                    #               sigma=1.0,
                                    #               order=0),
                    # if k == 2:
                    #     cden_int = (comb_cden.max()-comb_cden.min())/6.
                    #     axes[k].contour(XXs[0], YYs[0],
                    #                     ndimage.gaussian_filter(
                    #                         comb_cden,
                    #                         sigma=1.0,
                    #                         order=0),
                    #                     np.linspace(comb_cden.min()+cden_int,
                    #                                 comb_cden.max(), 8),
                    #                     colors='b', linewidths=1,
                    #                     linestyles='dashed')

                    if si == -1:
                        axes[k].set_xlabel(column_header[i]+column_tail[0])
                    else:
                        axes[k].set_xlabel(column_header[i//4]+column_tail[0])

                    axes[k].set_xlim(show_x.min(), show_x.max())
                    axes[k].set_ylim(show_y.min(), show_y.max())
                    axes[k].set_title(titles[k])

                    if k == 0:
                        axes[k].yaxis.tick_left()
                        axes[k].xaxis.tick_bottom()
                        axes[k].set_xticks([], minor=True)
                        axes[k].set_yticks([], minor=True)
                        axes[k].tick_params(axis='y', which='both', direction='out')
                        axes[k].tick_params(axis='x', which='both', direction='out')
                        if si == -1:
                            axes[k].set_ylabel(column_header[i+si])
                        else:
                            axes[k].set_ylabel(column_header[i//4+1]+column_tail[0])
                    else:
                        axes[k].yaxis.tick_left()
                        axes[k].xaxis.tick_bottom()
                        axes[k].set_xticks([], minor=True)
                        axes[k].set_yticks([], minor=True)
                        axes[k].tick_params(axis='y', which='both', direction='out')
                        axes[k].tick_params(axis='x', which='both', direction='out')
                        axes[k].set_yticks([])

                fig.subplots_adjust(left=0.2,
                                    right=0.8,
                                    bottom=0.1,
                                    top=0.9,
                                    wspace=0.1)

                cbar_ax = fig.add_axes([0.815, 0.18, 0.015, 0.64])
                cbar = fig.colorbar(sf, cax=cbar_ax)
                cbar.set_ticks([0, 25, 50, 75, 100])
                cbar.set_ticklabels([0, 25, 50, 75, 100])
                cbar.set_label("Density")
                cbar.ax.tick_params(axis='y', which='both', direction='out')

                temp_sn = sn+'_'+fprefix[j]+str(l).zfill(2)
                plt.savefig(temp_sn)
                plt.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
                plt.close('all')
                logging.info("Color-Color Density is saved at %s" % temp_sn)


def ColorColorDensityDiff(db, dcp_label, dcp_ul, opt, color=True):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = 'ColorColorDensityDiff'
    fn += '' if color else '_err'
    sn = os.path.join(dn, fn)
    density_dn = './density'
    density_dn += '' if color else '_err'
    if not os.path.exists(density_dn):
        os.makedirs(density_dn)

    color_id = db['eval_in'].dataset.X.numpy().T
    color_lo = db['eval_lo'].dataset.X.numpy().T
    color_ul = db['eval_ul'].dataset.X.numpy().T

    dcp_id = dcp_label[0][:, 0]
    dcp_lo = dcp_label[1][:, 0]

    dcp = np.hstack((dcp_id, dcp_lo, dcp_ul))
    bound_int = dcp.min()+dcp.max()
    bounds = [dcp.min(), bound_int/3., bound_int*2/3., dcp.max()]

    dcps = [dcp_id, dcp_lo, dcp_ul]

    column_header = [r"$(g-r)_", r"$(r-i)_",
                     r"$(i-z)_", r"$(z-y)_", r"$E(B-V)$"]
    column_tail = [r"{c}$", r"{d}$"] if color else [r"{ce}$", r"{de}$"]

    start_i = 0 if color else 1
    iterators = np.arange(start_i, len(color_id)-2, 4)
    fprefix = ["LOOD", "UL"]
    titles = [r"$ID_{pred}$", r"$PEND_{pred}$", r"$OOD_{pred}$"]
    if opt.psc:
        titles[-1] += " (LPSC)" if opt.psc_reg == 'low' else " (HPSC)"
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        cmap = plt.cm.get_cmap('Spectral_r')
        norm = matplotlib.colors.BoundaryNorm(
            np.linspace(-100, 100, 256), 255)
        for l, i in enumerate(iterators):
            if color:
                if i == iterators[-1]:
                    si = -1
                    i = 0
                else:
                    si = 4
            else:
                si = 2
            ci = [color_id[i], color_id[i+si]]
            cl = [color_lo[i], color_lo[i+si]]
            cu = [color_ul[i], color_ul[i+si]]
            colors = [ci, cl, cu]

            den_id_fn = 'density_id_%s.npy' % str(l).zfill(2)
            den_lo_fn = 'density_lo_%s.npy' % str(l).zfill(2)
            den_id_fn = os.path.join(density_dn, den_id_fn)
            den_lo_fn = os.path.join(density_dn, den_lo_fn)
            if os.path.exists(den_id_fn) and os.path.exists(den_lo_fn):
                logging.info("Load ID density from %s" % den_id_fn)
                logging.info("Load LO density from %s" % den_lo_fn)
                XX_id, YY_id, density_id = np.load(den_id_fn)
                XX_lo, YY_lo, density_lo = np.load(den_lo_fn)
                XXs = [XX_id, XX_lo]
                YYs = [YY_id, YY_lo]
                densities = [density_id, density_lo]
            else:
                XXs, YYs = [], []
                densities = []
                den_fn = [den_id_fn, den_lo_fn]
                for k, col in enumerate(colors[:-1]):
                    xx = np.linspace(col[0].min(), col[0].max(), 100)
                    yy = np.linspace(col[1].min(), col[1].max(), 100)
                    XX, YY = np.meshgrid(xx, yy)
                    XXs.append(XX)
                    YYs.append(YY)
                    positions = np.vstack([XX.ravel(), YY.ravel()])

                    cc = np.vstack([col[0], col[1]])
                    randidx = np.random.choice(np.arange(cc.shape[1]), 20000)
                    cc = cc[:, randidx]

                    density = np.reshape(kde_scipy(cc, positions).T, XX.shape)
                    densities.append(density)

                    np.save(den_fn[k], [XX, YY, density])

                    logging.info("Density is saved at %s" % den_fn[k])

            lim_tol = densities[0].max()/10
            coor_x = XXs[0].ravel()
            coor_y = YYs[0].ravel()
            coor_mask = densities[0].ravel() > lim_tol
            show_x = coor_x[coor_mask]
            show_y = coor_y[coor_mask]

            for j, (c1, c2) in enumerate(colors[1:]):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                dcp = dcps[j+1]

                for k in range(3):
                    bounded_idx = (bounds[k] <= dcp) & (bounds[k+1] > dcp)

                    temp_c1 = c1[bounded_idx]
                    temp_c2 = c2[bounded_idx]

                    cpositions = np.vstack([XXs[0].ravel(), YYs[0].ravel()])

                    ccc = np.vstack([temp_c1, temp_c2])
                    randidx = np.random.choice(
                        np.arange(ccc.shape[1]), 20000)
                    ccc = ccc[:, randidx]

                    cdensity = np.reshape(
                        kde_scipy(ccc, cpositions).T, XXs[0].shape)

                    normed_ref_density = densities[0]/densities[0].max()*100
                    normed_comp_density = cdensity/cdensity.max()*100

                    density_diff = normed_ref_density - normed_comp_density

                    # sf = axes[k].imshow(density_diff,
                    #                     extent=(np.amin(XXs[0]),
                    #                             np.amax(XXs[0]),
                    #                             np.amin(YYs[0]),
                    #                             np.amax(YYs[0])),
                    #                     interpolation='spline36',
                    #                     origin='lower',
                    #                     aspect='auto',
                    #                     cmap=cmap,
                    #                     norm=matplotlib.colors.BoundaryNorm(
                    #                         np.linspace(-100, 100, 256), 255))

                    sf = axes[k].tricontourf(XXs[0].ravel(),
                                             YYs[0].ravel(),
                                             ndimage.gaussian_filter(
                                                density_diff.ravel(),
                                                sigma=1.0,
                                                order=0),
                                             cmap=cmap,
                                             norm=norm,
                                             levels=np.linspace(-100, 100, 17))

                    den_int = (densities[0].max()-densities[0].min())/6.
                    axes[k].contour(XXs[0], YYs[0],
                                    ndimage.gaussian_filter(
                                        densities[0],
                                        sigma=1.0,
                                        order=0),
                                    np.linspace(densities[0].min()+den_int,
                                                densities[0].max(), 8),
                                    colors='k', linewidths=1)

                    if si == -1:
                        axes[k].set_xlabel(column_header[i]+column_tail[0])
                    else:
                        axes[k].set_xlabel(column_header[i//4]+column_tail[0])

                    axes[k].set_xlim(show_x.min(), show_x.max())
                    axes[k].set_ylim(show_y.min(), show_y.max())
                    axes[k].set_title(titles[k])

                    if k == 0:
                        if si == -1:
                            axes[k].set_ylabel(column_header[i+si])
                        else:
                            axes[k].set_ylabel(column_header[i//4+1]+column_tail[0])
                    else:
                        axes[k].set_yticks([])

                fig.subplots_adjust(left=0.2,
                                    right=0.8,
                                    bottom=0.1,
                                    top=0.9,
                                    wspace=0.1)

                cbar_ax = fig.add_axes([0.815, 0.18, 0.015, 0.64])
                cbar = fig.colorbar(sf, cax=cbar_ax)
                cbar.set_label("Density difference")
                cbar.set_ticks([-100, -50, 0, 50, 100])
                cbar.set_ticklabels([-100, -50, 0, 50, 100])

                temp_sn = sn+'_'+fprefix[j]+str(l).zfill(2)
                plt.savefig(temp_sn)
                plt.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
                plt.close('all')
                logging.info("Color-Color Density difference is saved at %s" % temp_sn)


def _load_gll(opt, rad_conv=False):
    _fn = 'GLL_eval_%s.npy' % opt.ul_prefix
    _fn = os.path.join(opt.data_dn, _fn)

    p = Path(_fn)
    with p.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        gll = np.load(f)
        while f.tell() < fsz:
            gll = np.vstack((gll, np.load(f)))
    if rad_conv:
        return gll*np.pi/180

    return gll.T


def Mollweide_GLL(dcp_ul, opt):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = "Mollweide_GLL"
    sn = os.path.join(dn, fn)

    cmap = plt.cm.get_cmap('jet')

    gll = _load_gll(opt, True)
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        plt.subplot(111, projection='mollweide')
        sf = plt.scatter(gll[0], gll[1],
                         c=dcp_ul, cmap=cmap,
                         marker='.', alpha=0.5,
                         s=0.5, linewidth=0.3,
                         edgecolor=None)
        cbar = plt.colorbar(sf)
        cbar.set_label("Discrepancy loss")
        plt.savefig(sn)
        plt.savefig(sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("Mollweide plot is saved at %s" % sn)


def GLatPDF(dcp_ul, opt):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = "GLatPDF"
    sn = os.path.join(dn, fn)

    beps = 1e-5
    nbin = 6
    gl = _load_gll(opt)[1]
    gl_bin = np.linspace(gl.min()-beps, gl.max()+beps, nbin+1)
    bin_idx = np.digitize(gl, gl_bin)-1
    lws = [1, 0.4, 0.4, 0.4, 0.4, 1]
    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        plt.figure()
        dcp_grid = np.linspace(dcp_ul.min(), dcp_ul.max(), 1000)
        for i in range(nbin):
            if i != nbin-1:
                label = r"$b \, \in \, [%.2f, %.2f)$" % \
                        (gl_bin[i], gl_bin[i+1])
            else:
                label = r"$b \, \in \, [%.2f, %.2f]$" % \
                        (gl_bin[i], abs(gl_bin[i+1]))
            mask = i == bin_idx
            temp_pdf = kde_scipy(dcp_ul[mask], dcp_grid)
            plt.plot(dcp_grid, temp_pdf,
                     color=color_scheme[i],
                     linestyle='solid',
                     linewidth=lws[i],
                     label=label)
        plt.xlabel("Discrepancy loss")
        plt.ylabel("Number density")
        plt.legend()
        plt.savefig(sn)
        plt.savefig(sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("Longitude-DCP pdf plot is saved at %s" % sn)


def _load_psc(opt):
    _fn = 'PSC_eval_%s.npy' % opt.ul_prefix
    _fn = os.path.join(opt.data_dn, _fn)

    p = Path(_fn)
    with p.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        psc = np.load(f)
        while f.tell() < fsz:
            psc = np.vstack((psc, np.load(f)))

    return psc.ravel()


def _psc_filter(dcp_ul, psc):
    psc_mask = np.where(~np.isnan(psc))[0]

    return dcp_ul[psc_mask], psc[psc_mask]


def PSCPDF(dcp_ul, opt):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = "PSCPDF_linear"
    sn = os.path.join(dn, fn)

    psc = _load_psc(opt)
    dcp_ul, psc = _psc_filter(dcp_ul, psc)

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


def PSC_DCP_scatter(dcp_ul, opt):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = "PSC_DCP_scattergram"
    sn = os.path.join(dn, fn)

    psc = _load_psc(opt)
    dcp_ul, psc = _psc_filter(dcp_ul, psc)

    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()
        ax.scatter(dcp_ul, psc,
                   color='b', s=0.5,
                   linewidth=0.3,
                   edgecolor=None)

        tick_setting(ax, r"$L_{DCP}$", "PSC",
                     None, None, None, None)
        fig.savefig(sn)
        fig.savefig(sn+'.pdf', format='pdf', dpi=300)
        plt.close('all')
        logging.info("PSC DCP scattergram is saved at %s" % sn)


def LPSC_scattergram(db, dcp_ul, opt):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = "LPSC_scattergram"
    sn = os.path.join(dn, fn)

    deval = db['eval_ul']
    evalset = deval.dataset.X.numpy()

    psc = _load_psc(opt)
    # dcp_ul, psc = _psc_filter(dcp_ul, psc)

    lpsc_mask = np.where(~np.isnan(psc))[0]
    evalset = evalset[lpsc_mask]
    dcp_ul = dcp_ul[lpsc_mask]
    psc = psc[lpsc_mask]

    psc_mask = psc <= 0.1
    ldcp_mask = np.where((dcp_ul <= 2) & psc_mask)[0]
    hdcp_mask = np.where((dcp_ul > 2) & psc_mask)[0]

    ldeval = evalset[ldcp_mask]
    hdeval = evalset[hdcp_mask]

    xidx = [0, 4, 8, 0]
    yidx = [4, 8, 12, -1]
    xlabel = [r"$(g-r)_{c}$", r"$(r-i)_{c}$", r"$(i-z)_{c}$", r"$(g-r)_{c}$"]
    ylabel = [r"$(r-i)_{c}$", r"$(i-z)_{c}$", r"$(z-y)_{c}$", r"E(B-V)"]
    for i in range(4):
        density_dn = './density'
        den_id_fn = 'density_id_%s.npy' % str(i).zfill(2)
        den_id_fn = os.path.join(density_dn, den_id_fn)
        if os.path.exists(den_id_fn):
            logging.info("Load ID density from %s" % den_id_fn)
            XX, YY, density = np.load(den_id_fn)
        else:
            raise NameError("Can't find %s" % den_id_fn)

        with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
            plt.rcParams["font.family"] = "Times New Roman"

            fig, ax = plt.subplots()
            ax.scatter(ldeval[:, xidx[i]],
                       ldeval[:, yidx[i]],
                       color='b', marker='+',
                       alpha=0.5, s=0.5,
                       linewidth=0.3,
                       edgecolor=None,
                       label="Low DCP-loss")

            ax.scatter(hdeval[:, xidx[i]],
                       hdeval[:, yidx[i]],
                       color='r', marker='x',
                       alpha=0.5, s=0.5,
                       linewidth=0.3,
                       edgecolor=None,
                       label="High DCP-loss")

            den_int = (density.max()-density.min())/10.
            ax.contour(XX, YY, density,
                       np.linspace(density.min()+den_int,
                                   density.max(), 8),
                       colors='k', linewidths=0.5)

            plt.xlabel(xlabel[i])
            plt.ylabel(ylabel[i])

            scolors = ['b', 'r']
            label_handles = ["Low DCP-Loss", "High DCP-Loss"]
            legend_handles = []
            for k in range(2):
                handle = mlines.Line2D([], [], color=scolors[k],
                                       marker='.', linestyle='None',
                                       markersize=5,
                                       label=label_handles[k])
                legend_handles.append(handle)
            plt.legend(loc='best', handles=legend_handles)

            temp_sn = sn + str(i)
            plt.savefig(temp_sn)
            plt.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
            plt.close('all')
            logging.info("LPSC scattergram is saved at %s" % temp_sn)


def LPSC_histogram(db, dcp_ul, opt):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = "LPSC_histogram"
    sn = os.path.join(dn, fn)

    deval_in = db['eval_in']
    evalset_in = deval_in.dataset.X.numpy()

    deval_ul = db['eval_ul']
    evalset_ul = deval_ul.dataset.X.numpy()

    psc = _load_psc(opt)
    # dcp_ul, psc = _psc_filter(dcp_ul, psc)

    lpsc_mask = np.where(~np.isnan(psc))[0]
    levalset = evalset_ul[lpsc_mask]
    dcp_ul = dcp_ul[lpsc_mask]
    psc = psc[lpsc_mask]

    psc_mask = psc <= 0.1
    ldcp_mask = np.where((dcp_ul <= 2) & psc_mask)[0]
    hdcp_mask = np.where((dcp_ul > 2) & psc_mask)[0]

    ldeval = levalset[ldcp_mask]
    hdeval = levalset[hdcp_mask]

    matplotlib.mathtext.SHRINK_FACTOR = 0.5
    matplotlib.mathtext.GROW_FACTOR = 1 / 0.5

    xidx = [0, 4, 8, 12, -1]
    xlabel = [r"$(g-r)_{c}$", r"$(r-i)_{c}$",
              r"$(i-z)_{c}$", r"$(z-y)_{c}$", r"$E(B-V)$"]
    for i in range(5):
        with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
            plt.rcParams["font.family"] = "Times New Roman"

            fig, ax = plt.subplots()
            temp_inx = evalset_in[:, xidx[i]]
            # temp_ulx = evalset_ul[:, xidx[i]]
            temp_lx = ldeval[:, xidx[i]]
            temp_hx = hdeval[:, xidx[i]]

            ingrid = np.linspace(temp_inx.min(), temp_inx.max(), 1000)
            # ulgrid = np.linspace(temp_ulx.min(), temp_ulx.max(), 1000)
            lgrid = np.linspace(temp_lx.min(), temp_lx.max(), 1000)
            hgrid = np.linspace(temp_hx.min(), temp_hx.max(), 1000)

            inpdf = kde_scipy(temp_inx, ingrid)
            # ulpdf = kde_scipy(temp_ulx, ulgrid)
            lpdf = kde_scipy(temp_lx, lgrid)
            hpdf = kde_scipy(temp_hx, hgrid)

            ax.plot(ingrid, inpdf,
                    color='0', linestyle='solid',
                    label=r"$ID$")
            # ax.plot(ulgrid, ulpdf,
            #         color='0.5', linestyle='solid',
            #         label=r"$UL$")
            ax.plot(lgrid, lpdf,
                    color='b', linestyle='solid',
                    label=r"$UL^\mathrm{LPSC}_\mathrm{LDCP}$")
            ax.plot(hgrid, hpdf,
                    color='r', linestyle='solid',
                    label=r"$UL^\mathrm{LPSC}_\mathrm{HDCP}$")

            plt.xlabel(xlabel[i])
            plt.ylabel("Number density")
            plt.legend()

            temp_sn = sn + str(i)
            plt.savefig(temp_sn)
            plt.savefig(temp_sn+'.pdf', format='pdf', dpi=300)
            plt.close('all')
            logging.info("LPSC distribution is saved at %s" % temp_sn)


def RedshiftDistribution(zspec, zphot, opt, key='ID'):
    dn = opt.plot_fd
    if not os.path.exists(dn):
        os.makedirs(dn)
    fn = "redshift_distribution_%s" % key
    sn = os.path.join(dn, fn)

    zphot_univ = zphot[0][0]
    zphot_spec = zphot[0][1]

    zgrid = np.linspace(0, 2, 1000)
    zspec_pdf = kde_scipy(zspec.ravel(), zgrid)
    zphot_univ_pdf = kde_scipy(zphot_univ.ravel(), zgrid)
    zphot_spec_pdf = kde_scipy(zphot_spec.ravel(), zgrid)

    with plt.style.context(['science', 'ieee', 'high-vis', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots()
        ax.plot(zgrid, zspec_pdf,
                color=color_scheme[0],
                label=r"$z_{spec}$",
                linestyle='solid')
        ax.plot(zgrid, zphot_univ_pdf,
                color=color_scheme[1],
                label=r"$z^{hent}_{phot}$",
                linestyle='solid')
        ax.plot(zgrid, zphot_spec_pdf,
                color=color_scheme[2],
                label=r"$z^{lent}_{phot}$",
                linestyle='solid')
        ax.legend()
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Number density")
        fig.savefig(sn)
        logging.info("Redshift distribution is saved at %s" % sn)


def CalibrationPlot(probs, zcls, opt, key='universal'):
    pfn = os.path.join(opt.plot_fd, 'CalibrationPlot')
    pfn += '_'+key

    probs = probs.T
    confidence = np.max(probs, axis=0)
    pred = np.argmax(probs, axis=0)

    weights = np.ones(len(confidence))/len(confidence)
    acc = np.sum(pred == zcls)/len(pred)
    avc = np.mean(confidence)

    partial_acc = []
    conf_binc = []
    conf_bin = np.arange(0, 1.1, 0.1)
    CE_coeff = np.zeros(len(conf_bin)-1)
    nsamples = probs.shape[1]
    for ci in range(len(conf_bin)-1):
        mask1 = conf_bin[ci] <= confidence
        mask2 = conf_bin[ci+1] >= confidence
        mask = mask1*mask2

        nsamples_inbin = np.sum(mask)
        CE_coeff[ci] = nsamples_inbin/nsamples

        masked_pred = pred[mask]
        masked_zbin = zcls[mask]
        if np.sum(mask) == 0:
            masked_acc = 0
        else:
            masked_acc = np.sum(masked_pred == masked_zbin)/nsamples_inbin

        logging.info("acc: %s" % masked_acc)
        partial_acc.append(masked_acc)
        conf_binc.append((conf_bin[ci]+conf_bin[ci+1])/2.)
    conf_binc = np.array(conf_binc)
    partial_acc = np.array(partial_acc)

    with plt.style.context(['science', 'ieee', 'high-vis', 'grid', 'no-latex']):
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax = plt.subplots(2, 1)

        ax[0].hist(confidence, 100, weights=weights, color='b', alpha=0.5)
        ax[0].vlines(acc, 0, 1, linestyles='dashed',
                     color='k', label="Mean accuracy")
        ax[0].vlines(avc, 0, 1, linestyles='dashed',
                     color='r', label="Mean confidence")
        ax[0].set_ylabel("Number density")
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 0.05)
        ax[0].set_xticklabels(())
        ax[0].legend()

        ax[1].bar(conf_binc, conf_binc,
                  alpha=0.5, color='r',
                  width=0.1, edgecolor='k',
                  linewidth=1, label="Ideal")
        ax[1].bar(conf_binc, np.array(partial_acc),
                  alpha=0.5, color='b', width=0.1,
                  edgecolor='k', linewidth=1, label="MBRNN")
        ax[1].plot(np.arange(0, 1.1, 0.1),
                   np.arange(0, 1.1, 0.1),
                   color='k', ls='--')
        ax[1].set_ylim(0, 1)
        ax[1].set_xlim(0, 1)
        ax[1].set_yticks(np.arange(0, 0.9, 0.2))
        ax[1].set_xlabel("Confidence")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        fig.subplots_adjust(hspace=0.001)
        fig.savefig(pfn+'.%s' % 'png')
        fig.savefig(pfn+'.%s' % 'pdf')
        plt.close()

    ECE = np.sum(CE_coeff*np.abs(conf_binc-partial_acc))

    conf_binc = np.delete(conf_binc, np.where(partial_acc == 0))
    partial_acc = np.delete(partial_acc, np.where(partial_acc == 0))

    if len(conf_binc):
        MCE = np.max(np.abs(conf_binc-partial_acc))
    else:
        MCE = 0.

    logging.info("Calibration plot is saved at %s" % pfn)

    return [ECE, MCE]
