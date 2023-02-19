mol new _R_PDB_FILE_
animate delete all
mol addfile _R_DCD_FILE_ first 0 last -1 step 1 waitfor all
set selectfile [open _R_SELECTION_FILE_ r]
set i 0
while {[gets $selectfile line] >=0 } {
set a [round [lindex $line 0 ]]
set j [format {%05g} $i]
animate write xyz _R_XYZ_OUT_ beg $a end $a skip 0 waitfor all
incr i
}
quit