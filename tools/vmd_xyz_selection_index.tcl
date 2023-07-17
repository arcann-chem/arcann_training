#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Very fancy DeepMD-based semi-automatic highly-customizable iterative training procedure. #
#   Copyright 2022-2023 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#

mol new _R_PDB_FILE_
animate delete all
mol addfile _R_XYZ_FILE_ first 0 last -1 step 1 waitfor all
set selectfile [open _R_SELECTION_FILE_ r]
set i 0
while {[gets $selectfile line] >=0 } {
set a [expr round([lindex $line 0])]
set j [format {%05g} $a]
animate write xyz _R_XYZ_OUT_ beg $a end $a skip 0 waitfor all
incr i
}
quit