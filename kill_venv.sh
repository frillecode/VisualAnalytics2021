#!/usr/bin/env bash

VENVNAME=cds-vis
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME