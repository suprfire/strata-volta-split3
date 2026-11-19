@echo off
rem STRATA_ROOT = wherever the patched Strata checkout lives (serve/server.py inside)
set "STRATA_ROOT=D:\strata"
title Strata split3 LAN
cd /d "%STRATA_ROOT%"
set FNX_PF_ROWS=4096
set FNX_ARENA=0
set FNX_EXPERT_RAM_GIB=off
set FNX_PEER=1
set FNX_PF_PEER=1
set FNX_PF_A2A_ROWS=1024
set FNX_PF_PASS=4
set STRATA_SM70_TABLE=1
"%STRATA_ROOT%\.venv\Scripts\python.exe" "%STRATA_ROOT%\serve\server.py" --engine strata --config "%STRATA_ROOT%\configs\strata-nvfp4-split3-262k-4096p4.json" --port 8080 --host 0.0.0.0
if errorlevel 1 pause
