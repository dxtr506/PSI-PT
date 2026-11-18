@echo off
rem FPR experiments (target coefficients beta0 = 0): PSI-PT, PSI-PT-oc, Bonferroni and naive (method si)
rem and data splitting (method split), for n0 = 40, 50, 60, 70; p = 500, K = 4 sources of size 200, h = 10.
rem   Chunks of 100 p-values; chunk c uses the seed seed0 + 4000 c, with seed0 = 400000 + 4000 (n0 - 40).
rem   si: 10 chunks from seed0 and 5 chunks from seed0 + 400000 (1500 p-values); split: 10 chunks (1000 p-values).
rem Run from anywhere:  experiments\run_fpr.bat
rem Environment: PY (python executable, default python), N_JOBS (processes per p-value, default 4).
rem Results: results\fpr\*.json; figure: results\fpr.pdf (+ .png)
setlocal EnableDelayedExpansion
cd /d "%~dp0.."
set "PYTHONPATH=%CD%"
set "OMP_NUM_THREADS=1" & set "MKL_NUM_THREADS=1" & set "OPENBLAS_NUM_THREADS=1"
if not defined PY set "PY=python"
if not defined N_JOBS set "N_JOBS=4"
set "OUT=results\fpr"
if not exist "%OUT%" mkdir "%OUT%"

for %%n in (40 50 60 70) do (
    set /a "s0=400000 + 4000 * (%%n - 40)"
    set /a "s1=s0 + 400000"
    call :fpr split %%n !s0! 9
    call :fpr si %%n !s0! 9
    call :fpr si %%n !s1! 4
)
"%PY%" experiments\plot.py fpr
goto :eof

rem :fpr <method> <n0> <seed0> <last chunk>
:fpr
for /L %%c in (0,1,%4) do (
    set /a seed=%3 + 4000 * %%c
    "%PY%" experiments\run_experiment.py --mode fpr --method %1 --nts %2 --H 10 ^
        --seed !seed! --n_rep 100000 --n_tests 100 --n_jobs %N_JOBS% --save_every 50 ^
        --out %OUT%\fpr_%1_h10_ng_nt%2_seed!seed!.json
)
exit /b 0
