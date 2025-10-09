@echo off
setlocal EnableDelayedExpansion

rem List resolution and frame rate for every video file under a directory.
if "%~1"=="" (
  echo Usage: %~nx0 "path\to\folder"
  exit /b 1
)

set "ROOT=%~1"
if not exist "%ROOT%" (
  echo Target path not found: %ROOT%
  exit /b 1
)

set "FFPROBE=%~dp0ffprobe.exe"
if not exist "%FFPROBE%" set "FFPROBE=ffprobe"

for /R "%ROOT%" %%F in (*) do (
  set "VIDEO_EXT="
  for %%E in (.mp4 .mov .MOV .mkv .avi .flv .ts .m4v .wmv .webm) do (
    if /I "%%~xF"=="%%E" set "VIDEO_EXT=1"
  )
  if defined VIDEO_EXT (
    call :probe "%%~fF"
  )
)

exit /b

:probe
set "VIDFILE=%~1"

for /f "usebackq tokens=1-3 delims=," %%A in (`%FFPROBE% -v error -select_streams v:0 -show_entries stream^=width^,height^,r_frame_rate -of csv^=p^=0 "%VIDFILE%"`) do (
  set "WIDTH=%%A"
  set "HEIGHT=%%B"
  set "RATE=%%C"
  
  rem Calculate FPS from fraction (e.g., 30000/1001 -> 29.97)
  for /f "tokens=1,2 delims=/" %%X in ("!RATE!") do (
    set /a "FPS_INT=%%X / %%Y"
    set /a "REMAINDER=(%%X %% %%Y) * 100 / %%Y"
    if !REMAINDER! GTR 0 (
      echo !VIDFILE!    %%Ax%%B    !FPS_INT!.!REMAINDER! fps
    ) else (
      echo !VIDFILE!    %%Ax%%B    !FPS_INT! fps
    )
  )
  exit /b 0
)

echo !VIDFILE!    [no video stream]
exit /b 0