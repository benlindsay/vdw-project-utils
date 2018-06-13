# Copyright (c) 2018 Ben Lindsay <benjlindsay@gmail.com>

from glob import glob
from os.path import basename
from os.path import join
import numpy as np


def get_x_y_z_rho(fname):
    """Given a grid file with 4 columns, x, y, z, rho, returns x, y, z, and rho

    Input
    -----
    fname : string, path to input file

    Output
    ------
    x     : 1d np-array of length nx
    y     : 1d np-array of length ny
    z     : 1d np-array of length nz
    rho   : 3d np-array of shape (nx, ny, nz)
    """
    x, y, z, rho = np.loadtxt(fname).T
    x, y, z = np.unique(x), np.unique(y), np.unique(z)
    nx, ny, nz = len(x), len(y), len(z)
    rho = rho.reshape((nz, ny, nx)).T
    return x, y, z, rho


def save_dat(x, y, z, rho, fname_out):
    """Save grid data in the same format as comes out of DMFT simulations,
    i.e. x, y, z, rho columns, with x changing fastest and z changing slowest

    Inputs
    ------
    x         : 1d np-array of length nx
    y         : 1d np-array of length ny
    z         : 1d np-array of length nz
    rho       : 3d np-array of shape (nx, ny, nz)
    fname_out : string, path of file to which to save data
    """
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    data = np.array(
        [X.T.flatten(), Y.T.flatten(), Z.T.flatten(), rho.T.flatten()]
    ).T
    np.savetxt(fname_out, data)


def center_density_line(z, rho, initial_shift=0.0, shift_to_zero=True,
                        tol=1e-6):
    """
    Given 1d z and rho arrays, shifts and returns the arrays so that the center
    of mass of rho falls in the middle of the array

    Inputs
    ------
    z             : 1d np-array of length nz
    rho           : 1d np-array of length nz
    initial_shift : float, optional, default=0
                    Shifts the line by that amount, useful if the center of
                    mass is far from the center of the array
    shift_to_zero : boolean, optional, default=True
                    Ensures z[0] == 0 if True, meaning the center of mass
                    will be slightly off center
    tol           : float, optional, default=1e-6
                    Iterative tolerance for proximity to center of array.
                    Iterations stop when abs(center_of_mass/Lz-0.5) <= tol

    Outputs
    -------
    z   : 1d np-array of length nz, monotonically increasing
    rho : 1d np-array of length nz, center of mass now at or near
          the center of the array
    shift : float, the amount the arrays shifted. -Lz/2 < shift <= Lz/2
    """
    z, rho = z.copy(), rho.copy()
    Lz = z[1] + z[-1]
    error = 1000
    z += initial_shift
    z -= np.floor(z/Lz) * Lz
    z, rho = z[np.argsort(z)], rho[np.argsort(z)]
    shift = initial_shift
    while abs(error) > tol:
        center_of_mass = np.sum(z * rho) / np.sum(rho)
        error = center_of_mass / Lz - 0.5
        z -= error * Lz
        shift -= error * Lz
        z -= np.floor(z/Lz) * Lz
    z, rho = z[np.argsort(z)], rho[np.argsort(z)]
    if shift_to_zero:
        shift -= z[0]
        z -= z[0]
    while shift > Lz / 2:
        shift -= Lz
    while shift <= -Lz / 2:
        shift += Lz
    return z, rho, shift


def shift_3d_data(z, rho, initial_shift=0.0, shift=None, shift_to_zero=True):
    """
    Shifts 3d data so that center of mass is in the center along the
    z-dimension

    Inputs
    ------
    z     : 1d np-array of length nz
    rho   : 3d np-array with shape (nx, ny, nz)
    shift : float, optional, amount in z direction to shift rho and z.
            Calls center_density_line to determine shift if not provided

    Outputs
    -------
    z     : 1d np-array of length nz, shifted from input z
    rho   : 3d np-array with shape (nx, ny, nz), shifted along z-axis
    """
    z, rho = z.copy(), rho.copy()
    Lz = z[1] + z[-1] - 2 * z[0]
    # Shift by initial_shift
    # Get shift from center_density_line if not provided
    if shift is None:
        _, _, shift = center_density_line(
            z, np.mean(rho, axis=(0,1)),
            initial_shift=initial_shift, shift_to_zero=shift_to_zero
        )
    # Shift so that rho center of mass falls near center of z axis
    z += shift
    z -= np.floor(z/Lz) * Lz
    # Find all positions where z does not monotonically increase
    diffs = z[1:] - z[:-1]
    z_decrease_positions = np.argwhere(diffs<0)
    if len(z_decrease_positions) == 1:
        nroll = z_decrease_positions[0,0] + 1
        rho = np.roll(rho, -nroll, axis=2)
        z = np.roll(z, -nroll)
    elif len(z_decrease_positions) > 1:
        raise ValueError("There should be max 1 spot where z decreases")
    return z, rho


def generate_cum_avg_rho_files(simdir, prefixes=['avg_rhop'],
                               half_box_initial_shift=False, cum_freq=10,
                               n_skip=0):
    """
    Given a simulation directory `simdir`, accumulate average blocks for files
    with prefixes in `prefixes` and save to file every `cum_freq` blocks (and
    after the last block)

    Inputs
    ------
    simdir   : string, simulation directory
    prefixes : list, default=['avg_rhop'], data file prefixes. For example, if
               `prefixes=['avg_rhop', 'avg_rhoda']`, then accumulates files of
               type avg_rhop_*.dat and avg_rhoda_*.dat
    half_box_initial_shift : boolean, optional, default=False
                             If True, shifts the line by half the box length
                             before searching for center of mass
    cum_freq : int, default=10, frequency at which to save accumulated files.
               A frequency of 1 would save i.e. `cum_avg_rhop_1.dat`,
               `cum_avg_rhop_2.dat`, etc. while a frequency of 10 would save
               i.e. `cum_avg_rhop_10.dat`, `cum_avg_rhop_20.dat`, etc,
               assuming n_skip=0.  A file will always be written for the final
               block as well.
    n_skip   : int, default=0, number of blocks to skip before accumulating
    """
    for prefix in prefixes:
        # get a list of all avg_rhop filenames in simdir
        avg_rho_fnames = [
            basename(f) for f in glob(join(simdir, '{}_*.dat'.format(prefix)))
        ]
        # find the highest number block in that directory and assign that to
        # num_blocks
        if len(avg_rho_fnames) == 0:
            num_blocks = 0
        else:
            num_blocks = max(
                [int(f.split('_')[2].split('.')[0]) for f in avg_rho_fnames]
            )
        rho_list = []
        # Start after `n_skip` and generate cum_avg file every `cum_freq`
        # blocks plus the last block
        for i in range(n_skip + 1, num_blocks + 1):
            dat = join(simdir, '{}_{}.dat'.format(prefix, i))
            x, y, z, rho = get_x_y_z_rho(dat)
            Lz = z[1] + z[-1]
            if prefix != 'avg_rhop':
                avg_rhop_dat = join(simdir, 'avg_rhop_{}.dat'.format(i))
                _, _, _, rhop = get_x_y_z_rho(avg_rhop_dat)
                _, _, initial_shift = center_density_line(
                    z, np.mean(rhop, axis=(0,1))
                )
            else:
                initial_shift = 0.0
            if half_box_initial_shift:
                initial_shift += 0.5 * (z[1] + z[-1])
            if initial_shift >= Lz:
                initial_shift -= Lz
            if initial_shift < 0:
                initial_shift += Lz
            z, rho = shift_3d_data(z, rho, initial_shift=initial_shift)
            if i == n_skip + 1:
                rho_sum = rho
            else:
                rho_sum += rho
            if (i-n_skip) % cum_freq == 0 or i == num_blocks:
                cum_dat = join(simdir, 'cum_{}_{}.dat'.format(prefix, i))
                rho_avg = rho_sum / float(i-n_skip)
                print('Saving {}'.format(cum_dat))
                save_dat(x, y, z, rho_avg, cum_dat)
