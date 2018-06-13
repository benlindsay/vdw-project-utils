#!/bin/bash
#
# split_gro.sh
#
# Copyright (c) 2018 Ben Lindsay <benjlindsay@gmail.com>

input_gro="$1"
base="$(basename $input_gro .gro)"
gro_output_dir="$(dirname $input_gro)/gros"
pdb_output_dir="$(dirname $input_gro)/pdbs"
fixed_gro="$(dirname $input_gro)/$base-fixed.gro"

if [ ! -f "$fixed_gro" ]; then
  echo "Correcting potential errors and writing to $fixed_gro:"

  cat $input_gro | sed -E 's/([a-zA-Z])100000/\1    0/g' \
    | sed 's/Writing frame/t=/g' > $fixed_gro
else
  echo "$input_gro already corrected and written to $fixed_gro"
fi

n_lines=$(head -2 $fixed_gro | tail -1 | awk '{print $1+3}')

if [ ! -d $gro_output_dir ] || [ -z "$(ls -A $gro_output_dir)" ]; then
  mkdir -p $gro_output_dir

  echo "Splitting $fixed_gro into individual frames in $gro_output_dir:"

  fmt=$gro_output_dir/$base"_%04d.gro"
  awk -v lines=$n_lines -v fmt="$fmt" \
    '{print>sprintf(fmt,1+int((NR-1)/lines))}' $fixed_gro

  first_file=$(printf "$fmt" 1)
  echo "First file: $first_file"
  n_files=$(find $gro_output_dir -maxdepth 1 -name $base"_*.gro" | wc -l)
  last_file=$(printf "$fmt" $n_files)
  echo "Last file: $last_file"

  n_lines_last=$(wc -l $last_file | awk '{print $1}')
  if [ "$n_lines_last" -ne "$n_lines" ]; then
    echo "$last_file is too short. Deleting."
    rm $last_file
  fi

else
  echo "$fixed_gro alread split into individual frames in $gro_output_dir"
fi

if [ ! -d $pdb_output_dir ] || [ -z "$(ls -A $pdb_output_dir)" ]; then
  echo "Converting individual gro frames into pdb frames"

  mkdir -p $pdb_output_dir

  for f in $gro_output_dir/*; do
    pdb_file="$pdb_output_dir/$(basename $f .gro).pdb"
    echo "Converting $f to $pdb_file:"
    gmx editconf -f $f -o $pdb_file
  done
else
  echo "Individual gro frames already converted to pdb frames in $pdb_output_dir"
fi
