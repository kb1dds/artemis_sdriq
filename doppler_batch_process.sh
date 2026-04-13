#!/bin/bash

for fl in `find ~/20260402_ArtemisII/ -size +1G -name *airlie* | cut -d'.' -f 2
`
do
    echo $fl

    python qpsk_dopplerscan.py ~/20260402_ArtemisII/03-29-26_airlie_test.${fl}.sdriq --window_size 262144 --windows 128 --batches -1 --dopplersamples 1024 --device cuda --outfile ${fl}_dopplerscan.png
done
