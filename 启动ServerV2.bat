@echo off
set PATH=%PATH%;%CD%;%CD%\ffmpeg-8.0-full_build\bin
call python-3.12.10-embed-amd64\python.exe serverV2.py

call pause