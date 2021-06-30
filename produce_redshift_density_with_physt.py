#!/usr/bin/env python3

import os
import sys
import numpy as np

import physt.binnings

try:
    infn = sys.argv[1]
    outfn_prefix = sys.argv[2]
except Exception:
    print('(usage) %s (input filename) (output filename prefix)' %
          (sys.argv[0]))
    sys.exit(1)

fsave = './bin_edges'
if not os.path.exists(fsave):
    os.makedirs(fsave)

# Non-uniform bin #
redshifts = np.loadtxt(infn, usecols=(2))
redshifts[redshifts < 0] = 0
num_redshifts = float(redshifts.size)
min_redshift = np.min(redshifts)
max_redshift = np.max(redshifts)
print("... min. and max. redshift: ", min_redshift, max_redshift)

num_bins = [32, 64, 128, 256, 512]


def create_nonuniform_bins(outfn, z, ncls):
    # quantile-based binning
    use_hist_bins = physt.binnings.quantile_binning(data=redshifts, bins=ncls)
    use_hist_bin_edges = use_hist_bins.numpy_bins
    # tweak the first and last bin edges

    prepend_value = 0.00010
    append_value = 0.00010

    use_hist_bin_edges[0] = use_hist_bin_edges[0] - prepend_value
    use_hist_bin_edges[-1] = use_hist_bin_edges[-1] + append_value

    print("... the number of bins: ", ncls)
    print("... the size of edges: ", use_hist_bin_edges.size)

    outfd = open(outfn, 'w')
    np.savetxt(outfd, use_hist_bin_edges, fmt='%.6f')
    outfd.close()


# Uniform bin #
def create_uniform_bins(outfn, z, ncls):
    prepend_value = 0.00010
    append_value = 0.00010

    minz, maxz = np.min(z)-prepend_value, np.max(z)+append_value

    bin_edges = np.linspace(minz, maxz, ncls+1)
    np.savetxt(outfn, np.array(bin_edges), fmt='%.6f')


for num_bin in num_bins:
    # nuni_outfn = os.path.join(fsave, outfn_prefix +
    #                           "%s-nonuniform.txt" % num_bin)
    # create_nonuniform_bins(nuni_outfn, redshifts, num_bin)

    uni_outfn = os.path.join(fsave, outfn_prefix +
                             "%s-uniform.txt" % num_bin)
    create_uniform_bins(uni_outfn, redshifts, num_bin)
