#!/usr/bin/env bash
VENVNAME=cds-vis
source $VENVNAME/bin/activate
python -m ipykernel install --user --name $VENVNAME --display-name "$VENVNAME"