#!/bin/bash
set -e

eval "$@"

# prevent docker exit
tail -f /dev/null
