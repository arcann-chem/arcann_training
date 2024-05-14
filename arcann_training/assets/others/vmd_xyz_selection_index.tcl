#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2023/08/16

# Load PDB file and clear all existing animations
mol new _R_PDB_FILE_
animate delete all

# Load XYZ file for trajectory and wait for it to finish loading
mol addfile _R_XYZ_FILE_ first 0 last -1 step 1 waitfor all

# Open frame index file and iterate through each line
set frame_index_file [open _R_FRAME_INDEX_FILE_ r]
while {[gets $frame_index_file line] >=0 } {
    set frame [expr {round(double($line))}]
    set j [format "%05g" $frame]
    
    # Write out a single frame of the trajectory for the selected atoms
    animate write xyz _R_XYZ_OUT__$j.xyz beg $frame end $frame skip 0 waitfor all
}

# Close frame index file and exit VMD
close $frame_index_file
quit