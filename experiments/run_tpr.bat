@echo off
rem TPR experiments: PSI-PT, PSI-PT-oc and Bonferroni (method si) and data splitting (method split).
rem   TPR vs n0 (n0 = 40, 50, 60, 70, delta = 0.5) and TPR vs delta (n0 = 50, delta = 0.25, 0.5, 0.75, 1.0);
rem   p = 500, K = 4 sources of size 200, S = 10 true features, contrast size h = 10.
rem   Each setting: 5 chunks of 100 p-values; chunk c uses the seed seed0 + 4000 c.
rem Run from anywhere:  experiments\run_tpr.bat
rem Environment: PY (python executable, default python), N_JOBS (processes per p-value, default 4).
rem Results: results\tpr\*.json; figures: results\tpr_nt.pdf, results\tpr_delta_h10_nt50.pdf (+ .png)
setlocal EnableDelayedExpansion
cd /d "%~dp0.."
set "PYTHONPATH=%CD%"
set "OMP_NUM_THREADS=1" & set "MKL_NUM_THREADS=1" & set "OPENBLAS_NUM_THREADS=1"
if not defined PY set "PY=python"
if not defined N_JOBS set "N_JOBS=4"
set "OUT=results\tpr"
if not exist "%OUT%" mkdir "%OUT%"

for %%m in (split si) do (
    for %%n in (40 50 60 70) do call :tpr %%m %%n 0.5 320000
    call :tpr %%m 50 0.25 300000
    call :tpr %%m 50 0.75 340000
    call :tpr %%m 50 1.0 360000
)
"%PY%" experiments\plot.py tpr
goto :eof

rem :tpr <method> <n0> <delta> <seed0>
:tpr
for /L %%c in (0,1,4) do (
    set /a seed=%4 + 4000 * %%c
    "%PY%" experiments\run_experiment.py --mode tpr --method %1 --nts %2 --S 10 --H 10 --delta %3 ^
        --seed !seed! --n_rep 3000 --n_tests 100 --n_jobs %N_JOBS% --save_every 10 ^
        --out %OUT%\tpr_%1_s10_h10_ng_nt%2_delta%3_seed!seed!.json
)
exit /b 0
