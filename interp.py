#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Ben Lindsay <benjlindsay@gmail.com>

from os.path import dirname
from os.path import exists
from os.path import join
from scipy.interpolate import interp1d
from subprocess import check_output
from subprocess import Popen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def interp_chain(x, N_interp):
    index = np.linspace(0, 1, len(x))
    f = interp1d(index, x)
    index_new = np.linspace(0, 1, N_interp)
    x_new = f(index_new)
    return x_new

class TrajectoryInterpolator():
    col_names = [
        'molId', 'molName', 'siteName', 'siteId',
        'cur_x', 'cur_y', 'cur_z',
        'prev_x', 'prev_y', 'prev_z',
        'rand_x', 'rand_y', 'rand_z',
    ]
    widths = [5] * 4 + [8] * 9
    fmt = '%5d%-5s%5s%5d' + ''.join(['%8.4f']*9)
    converters = {
        0: int,
        3: int,
        4: float,
        5: float,
        6: float,
        7: float,
        8: float,
        9: float,
        10: float,
        11: float,
        12: float,
    }

    def read_single_gro_frame_file(self, file_path):
        data = pd.read_fwf(
            file_path,
            widths=self.widths,
            skiprows=2,
            skipfooter=1,
            header=None,
            names=self.col_names,
            converters=self.converters,
        )
        return data

    def read(self, file_path, single_frame=True):
        if not single_frame:
            raise ValueError("multiple frame files not supported yet.")
        self.file_path_tmp = file_path + '.tmp'
        if not exists(self.file_path_tmp):
            cmd = (
                # Handle the bug where we print a (god forbid...) 6 digit
                # instead of just a 5 digit number for an index. So convert
                # any 100000s that have a letter just before them into that
                # same letter (\1) with a properly padded 0
                "sed -E 's/([a-zA-Z])100000/\1    0/g' {} > {}"
                .format(file_path, self.file_path_tmp)
            )
            cmd_output = Popen(cmd, shell=True).wait()
        self.data = self.read_single_gro_frame_file(self.file_path_tmp)
        return self

    def get_interped_data(self, new_N):
        polymer_data = self.data[self.data['molName']=='BCP']
        self.n_chains = polymer_data['molId'].nunique()
        first_molId = polymer_data['molId'].min()
        self.old_N = len(polymer_data[polymer_data['molId']==first_molId])
        interped_polymer_data = pd.DataFrame()
        for col in polymer_data.columns[4:13]:
            x_array = (
                polymer_data[col]
                .values
                .reshape((self.n_chains, self.old_N))
            )
            interped_x = np.apply_along_axis(interp_chain, 1, x_array, new_N)
            interped_polymer_data[col] = interped_x.flatten()
        molIds = (
            np.ones((self.n_chains, new_N)) * np.arange(self.n_chains)[:, None]
        ).astype(int)
        interped_polymer_data['molId'] = molIds.flatten()
        interped_polymer_data['molName'] = 'BCP'
        interped_polymer_data['siteName'] = 'H'
        particle_data = self.data[self.data['molName']=='GP']
        interped_data = pd.concat([interped_polymer_data, particle_data])
        interped_data['siteId'] = (np.arange(len(interped_data)) + 1) % 100000
        interped_data = interped_data[self.col_names]
        return interped_data

    def interp(self, new_file_name, new_N):
        self.interped_data = self.get_interped_data(new_N)
        length_ratio = np.sqrt((new_N-1.0) / (self.old_N-1.0))
        self.interped_data.iloc[:, 4:10] *= length_ratio
        head = (
            check_output('head -1 {}'.format(self.file_path_tmp), shell=True)
            .decode("utf-8")
        )
        tail = (
            check_output('tail -1 {}'.format(self.file_path_tmp), shell=True)
            .decode("utf-8")
        )
        L_xyz = np.array([float(L) for L in tail.split()])
        L_xyz *= length_ratio
        new_tail = ' '.join(L_xyz.astype(str)) + '\n'
        with open(new_file_name, 'w') as f:
            f.write(head)
            f.write('{}\n'.format(len(self.interped_data)))
            np.savetxt(f, self.interped_data.values, fmt=self.fmt)
            f.write(new_tail)


if __name__ == "__main__":
    import sys
    minargs = 3
    if len(sys.argv[1:]) < minargs:
        print("Usage:", sys.argv[0], "old_traj_file new_traj_file new_N")
        exit(1)
    else:
        old_traj_file, new_traj_file = sys.argv[1:3]
        new_N = int(sys.argv[3])
        ti = TrajectoryInterpolator()
        ti.read(old_traj_file).interp(new_traj_file, new_N)
