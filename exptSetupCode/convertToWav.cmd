@ECHO OFF
FOR %%f IN (*.mp3) DO (
    lame.exe --decode "%%f" "%%~nf.wav"
)
PAUSE