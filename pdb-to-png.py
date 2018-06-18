#!/usr/bin/env ovitos
# -*- coding:utf-8 -*-
#
# pdb-to-png.py
# 
# Copyright (c) 2018 Ben Lindsay <benjlindsay@gmail.com>

from ovito import dataset
from ovito.io import import_file
from ovito.modifiers import SelectExpressionModifier
from ovito.modifiers import SliceModifier
from ovito.vis import Viewport
from ovito.vis import RenderSettings
from ovito.vis import TachyonRenderer
import os

def main(pdb_path, png_path):
    node = import_file(pdb_path)
    node.add_to_scene()
    vp = dataset.viewports.active_vp
    # node.modifiers.append(SelectExpressionModifier(expression = 'Position.X < Position.Y'))
    node.modifiers.append(SliceModifier(distance=0, normal=(1, -1, 0)))
    node.compute()
    particle_types = node.output.particle_properties.particle_type
    particle_types.get_type_by_name('H').color = (1, 0.5, 0.5)
    particle_types.get_type_by_name('He').color = (0.5, 0.5, 1)
    particle_types.get_type_by_name('O').color = (0.5, 0.5, 0.5)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    settings = RenderSettings(
        filename = png_path,
        size = (800, 600),
        renderer = TachyonRenderer(),
    )
    vp.render(settings)

if __name__ == '__main__':
    import sys
    usage = "Usage: {} sim_dir".format(sys.argv[0])
    n_args = len(sys.argv[1:])
    if n_args < 1:
        print(usage)
    else:
        sim_dir = sys.argv[1]
        pdb_path = os.path.join(sim_dir, 'pdbs', 'rst_coord_0001.pdb')
        png_path = os.path.join(sim_dir, 'images', 'rst_coord_0001.png')
        main(pdb_path, png_path)
