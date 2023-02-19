import re

def analyze_plumed_file(plumed_lines):
    # Find if MOVINGRESTRAINT is present
    movres_found = False
    for line in plumed_lines:
        if 'MOVINGRESTRAINT' in line:
            movres_found = True
            break

    if movres_found:
        # Find the last value of the STEP keyword
        step_matches = re.findall(r'STEP\s*=\s*(\d+)', ''.join(plumed_lines))
        if len(step_matches) > 0:
            last_step = int(step_matches[-1])
            return 0, last_step
        else:
            
            return 1
    else:
        return 1