#!/bin/sh

# Dont test code that is not being committed
git stash -q --keep-index

# Test the commit
make clean; make -j4 test
RESULT=$?

# Restore working dir
git stash pop -q

# Return exit status
[ $RESULT -ne 0 ] && exit 1
exit 0
