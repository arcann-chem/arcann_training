#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2024/05/26
# Last modified: 2024/05/26
#----------------------------------------------

# Read the file line by line
while IFS= read -r directory; do
    # Remove leading/trailing whitespaces
    directory=$(echo "$directory" | xargs)

    # Create the "force" file in the director
    touch "$directory"/force && echo "touch ${directory}/force"
done < failed_explorations.txt