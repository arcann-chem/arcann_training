mol new _TOPO_FILE_
animate delete all
mol addfile _DCD_FILE_ first 0 last -1 step 1 waitfor all
set selectfile [open _SELECTION_FILE_ r]
set i 0
while {[gets $selectfile line] >=0 } {
set a [lindex $line 0  ]
set j [format {%05g} $i]
animate write xyz vmd_$j.xyz beg $a end $a skip 0 waitfor all
incr i
}
quit