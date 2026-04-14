#!/bin/bash

source .venv/bin/activate

for fl in `find ~/20260402_ArtemisII/ -size +1G -name "*2026-04*" | cut -d'.' -f 2`
do
    echo $fl

    python qpsk_dopplerscan.py ~/20260402_ArtemisII/03-29-26_airlie_test.${fl}.sdriq --window_size 65536 --windows 512 --batches 15 --dopplersamples 512 --dopplerstart 25000 --dopplerstop 45000 --bandpass 6000000 --device cuda --outfile ${fl}_dopplerscan_25_45kHz_6MHzBW.png

    python qpsk_dopplerscan.py ~/20260402_ArtemisII/03-29-26_airlie_test.${fl}.sdriq --window_size 65536 --windows 512 --batches 15 --dopplersamples 512 --dopplerstart -45000 --dopplerstop -25000 --bandpass 6000000 --device cuda --outfile ${fl}_dopplerscan_-45_-25kHz_6MHzBW.png
done
