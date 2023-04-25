@echo off

set ARGS_LIST=m1 m2 m3 m4 m5 m6 m7 ml1 ml2 ml3 ml4 ml5 ml6 ml7 n1 n2 n3 n4 n5 n6 n7

for %%a in (%ARGS_LIST%) do (
    start "%%a" python pulse_calculation.py OptikkLab/TekstData/Transmittans/%%a.txt resultater/%%a
)