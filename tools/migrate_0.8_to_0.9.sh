#!/bin/bash

cp control/datasets_initial.json control/datasets_initial.json.bak
mv control/datasets_initial.json control/initial_datasets.json

cp control/config.json control/config.json.bak
sed -i "s/\"ignore_first_n_frames\": 10/\"ignore_first_x_ps\": 0.5/g" control/config.json
sed -i "s/\"datasets_initial\"/\"initial_datasets\"/g" control/config.json
sed -i "s/, \"exploration_type\": \"lammps\"//g" control/config.json
sed -i "s/\"current_iteration\"/\"exploration_type\": \"lammps\",\"current_iteration\"/g" control/config.json

for f in control/training*.json; do
    cp "${f}" "${f/json/json.bak}"
    sed -i "s/avg_seconds_per_step/s_per_step/g" "${f}"
    sed -i "s/structures_trained_total/nb_trained/g" "${f}"
    sed -i "s/structures_initial_total/nb_initial/g" "${f}"
    sed -i "s/structures_added_nr_total/nb_added_nr/g" "${f}"
    sed -i "s/structures_added_r_total/nb_added_r/g" "${f}"
    sed -i "s/structures_added_nr_iter/nb_added_nr_iter/g" "${f}"
    sed -i "s/structures_added_r_iter/nb_added_r_iter/g" "${f}"
    sed -i "s/structures_extra_total/nb_extra/g" "${f}"
    sed -i "s/use_datasets_initial/use_initial_datasets/g" "${f}"
    sed -i "s/use_datasets_extra/use_extra_datasets/g" "${f}"
done

for f in inputs/*XXXX*.inp; do
    cp "${f}" "${f/.inp/.inp.bak}"
    sed -i "s/ _WALLTIME_/ _R_WALLTIME_/g" "${f}"
    sed -i "s/ _CELL_/ _R_CELL_/g" "${f}"
done

for f in inputs/*.in; do
    cp "${f}" "${f/.in/.in.bak}"
    sed -i "s/ _DATA_FILE_/ _R_DATA_FILE_/g" "${f}"
    sed -i "s/ _MODELS_LIST_/ _R_MODELS_LIST_/g" "${f}"
    sed -i "s/ _PRINT_FREQ_/ _R_PRINT_FREQ_/g" "${f}"
    sed -i "s/ _DEVI_OUT_/ _R_DEVI_OUT_/g" "${f}"
    sed -i "s/ _TIMESTEP_/ _R_TIMESTEP_/g" "${f}"
    sed -i "s/ _DCD_OUT_/ _R_DCD_OUT_/g" "${f}"
    sed -i "s/ _TEMPERATURE_/ _R_TEMPERATURE_/g" "${f}"
    sed -i "s/ _SEED_VEL_/ _R_SEED_VEL_/g" "${f}"
    sed -i "s/ _SEED_THER_/ _R_SEED_THER_/g" "${f}"
    sed -i "s/ _NUMBER_OF_STEPS_/ _R_NUMBER_OF_STEPS_/g" "${f}"
    sed -i "s/ _RESTART_OUT_/ _R_RESTART_OUT_/g" "${f}"
    sed -i "s/ _PLUMED_IN_/ _R_PLUMED_IN_/g" "${f}"
    sed -i "s/ _PLUMED_OUT_/ _R_PLUMED_OUT_/g" "${f}"
done

for f in inputs/*.dat; do
    cp "${f}" "${f/dat/dat.bak}"
    sed -i "s/=_PRINT_FREQ_/=_R_PRINT_FREQ_/g" "${f}"
done

for f in control/exploration*.json; do
    cp "${f}" "${f/json/json.bak}"
    sed -i "s/avg_seconds_per_step/s_per_step/g" "${f}"
    sed -i "s/\"ignore_first_n_frames\": 10/\"ignore_first_x_ps\": 0.5/g" "${f}"
    sed -i "s/, \"exploration_type\": \"lammps\"//g" "${f}"
    sed -i "s/\"subsys_nr\"/\"exploration_type\": \"lammps\",  \"subsys_nr\"/g" "${f}"
done

find ./*exploration/ -name "selection_candidates.json" -print0 | 
    while IFS= read -r -d $'\0' file; do mv "$file" "${file/selection_candidates.json/devi_info.json}"; done

find ./*exploration/ -name "selection_candidates_index.json" -print0 | 
    while IFS= read -r -d $'\0' file; do mv "$file" "${file/selection_candidates_index.json/devi_index.json}"; done

find ./*exploration/ -type f -name "devi_info.json" -exec sed -i "s/nb_selection_factor/selection_factor/g" {} +
find ./*exploration/ -type f -name "devi_info.json" -exec sed -i "s/nb_candidates_max_weighted/nb_candidates_max_local/g" {} +