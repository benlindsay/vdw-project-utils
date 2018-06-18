from os import rename
from os.path import basename
from os.path import dirname
from os.path import exists
from os.path import join
from os.path import realpath
from subprocess import check_output
import numpy as np
import pandas as pd

CODE_DIR_PATH = dirname(realpath(__file__))

def calc_b2(gr_fpath, Rp):
    r, gr = np.loadtxt(gr_fpath).T
    r *= 10
    dr = r[1] - r[0]
    b2 = -0.5 * np.sum(dr * (gr - 1) * 4 * np.pi * r * r)
    b2_hard_sphere = 16.0 / 3.0 * np.pi * Rp**3
    return b2 / b2_hard_sphere


def dir_list_to_sims_df(dir_list, col_dict_fn, sort_dirs=True):
    dict_list = []
    for d in dir_list:
        r = {}
        r['dir'] = d
        r['round_dir'], r['sim_dir'] = dirname(d), basename(d)
        col_dict = col_dict_fn(d)
        r.update(col_dict)
        dict_list.append(r)
    sims_df = pd.DataFrame(dict_list).sort_values(['round', 'sim_dir'])
    if not 'Rp' in sims_df.columns:
        raise ValueError("'Rp' is a required column")
    return sims_df


def calc_gr(sims_df, gr_exe_path=None, verbosity=0, gr_fname='gr_skp1.dat',
            new_gr_fname_fmt='gr_skp1_{:02d}.dat'):
    if gr_exe_path is None:
        gr_exe_path = join(CODE_DIR_PATH, '3d_gr', 'a.out')
    gr_calculations = 0
    for index, row in sims_df.iterrows():
        for i_frame in range(row['n_frames']):
            orig_fpath = gr_fname
            new_fpath = join(row['dir'], new_gr_fname_fmt.format(i_frame))
            if exists(new_fpath):
                if (verbosity > 1):
                    print('{} exists. Skipping.'.format(new_fpath))
                continue
            gr_calculations += 1
            kwargs = dict(
                gr_dr = 0.2 * row['Rg'] / 10.0, # divide by 10 to match scale in trajectory file
                traj_file = join(row['dir'], 'traj_np.gro'),
                box_col = 0, # I don't think this does anything...
                frmin = i_frame,
                frmax = i_frame+1,
                nP = row['np'],
                frs_skip = 1
            )
            cmd_str = (
                gr_exe_path +
                ' {gr_dr} {traj_file} {box_col} {frmin} {frmax} {nP} {frs_skip}'
                .format(**kwargs)
            )
            if verbosity > 1:
                print(cmd_str)
            output = check_output(cmd_str, shell=True)
            if verbosity > 1:
                print('Moving {} to {}'.format(orig_fpath, new_fpath))
            rename(orig_fpath, new_fpath)
        if verbosity > 0:
            print('Finished {}'.format(row['dir']))
    print('Finished g(r) calculations. Performed {} calculations.'.format(gr_calculations))


def sims_df_to_b2_df(sims_df, b2_df_index, keep_cols=None, gr_fname_fmt='gr_skp1_{:02d}.dat'):
    b2_df = pd.DataFrame(index=sims_df.groupby(b2_df_index).count().index)
    for group_name, group in sims_df.groupby(b2_df_index, sort=False):
        Rp = group['Rp'].values[0]
        if not np.allclose(group['Rp'], Rp):
            print('Rp values in group do not match')
            import pdb; pdb.set_trace()
        assert(np.allclose(group['Rp'], Rp))
        i_frame = 1
        for _, row in group.iterrows():
            for i_gr in range(1, row['n_frames']):
                gr_fname = join(row['dir'], gr_fname_fmt.format(i_gr))
                b2 = calc_b2(gr_fname, Rp)
                b2_df.loc[group_name, i_frame] = b2
                i_frame += 1
        if not keep_cols is None:
            for col in keep_cols:
                first_value = group[col].values[0]
                if not np.allclose(group[col], first_value):
                    print(col + ' values in group do not match')
                    import pdb; pdb.set_trace()
                assert(np.allclose(group[col], first_value))
                b2_df.loc[group_name, col] = first_value
    return b2_df


def b2_df_to_b2_agg_df(b2_df, b2_agg_df_index):
    int_column_names = [i for i in list(b2_df.columns) if isinstance(i, int)]
    b2_agg_df = (
        b2_df
        .groupby(b2_agg_df_index)[int_column_names]
        .agg([np.nanmean, np.nanstd, 'count'])
        .copy()
    )
    return b2_agg_df


def dir_list_to_dfs(dir_list, col_dict_fn, b2_df_index, b2_agg_df_index=None,
                    keep_cols=None, sort_dirs=True,
                    gr_fname_fmt='gr_skp1_{:02d}.dat', do_calc_gr=True,
                    gr_exe_path=None, verbosity=0, gr_fname='gr_skp1.dat'):
    if b2_agg_df_index is None:
        average_over = ['seed']
        b2_agg_df_index = [e for e in b2_df_index if e not in average_over]
    sims_df = dir_list_to_sims_df(dir_list, col_dict_fn, sort_dirs)
    if do_calc_gr:
        calc_gr(sims_df, gr_exe_path=gr_exe_path, new_gr_fname_fmt=gr_fname_fmt,
                verbosity=verbosity, gr_fname=gr_fname)
    b2_df = sims_df_to_b2_df(sims_df, b2_df_index, keep_cols, gr_fname_fmt)
    b2_agg_df = b2_df_to_b2_agg_df(b2_df, b2_agg_df_index)
    return sims_df, b2_df, b2_agg_df


def col_dict_from_files(d, r=None):
    if r is None:
        r = {}
    with open(join(d, 'dyft.input'), 'r') as f:
        lines = f.readlines()
    r['Na'], r['Nb'] = map(float, lines[0].split()[:2])
    r['phiP'] = float(lines[5].split()[0])
    r['Rp'] = float(lines[7].split()[0])
    r['a'] = float(lines[9].split()[0])
    r['rho0'] = float(lines[12].split()[0])
    r['chi'] = float(lines[13].split()[0])
    r['eps'] = float(lines[14].split()[0])
    r['Lx'], r['Ly'], r['Lz'] = map(float, lines[18].split()[:3])
    r['print_freq'] = int(lines[24].split()[0])
    r['seed'] = float(lines[29].split()[0])
    r['N'] = r['Na'] + r['Nb']
    r['fa'] = r['Na'] / r['N']
    r['fb'] = r['Nb'] / r['N']
    r['Rg'] = np.sqrt((r['N'] - 1) / 6.0)
    r['epsN'] = r['eps'] * r['N']
    with open(join(d, 'LOG'), 'r') as f:
        lines = f.readlines()
    r['n_frames'] = sum(1 for l in lines if l.startswith('step '))
    r['np'] = int([l for l in lines if l.startswith('nP: ')][0].split()[1])
    return r


def disc_col_dict_fn(d):
    sim_dir = basename(d)
    r = {}
    r['N'], r['epsN'], r['seed'] = [
        f(s.split('-')[1]) for f, s in zip([int, float, int], sim_dir.split('_'))
    ]
    r = col_dict_from_files(d, r)
    return r

    
def cubic_col_dict_fn(d):
    sim_dir = basename(d)
    r = {}
    r['eps'] = float(sim_dir.split('_')[1].split('-')[1])
    r['seed'] = int(sim_dir.split('_')[-1])
    r['N'] = 40
    r['epsN'] = r['eps'] * r['N']
    r = col_dict_from_files(d, r)
    return r
