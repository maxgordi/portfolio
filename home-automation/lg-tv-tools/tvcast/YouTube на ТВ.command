#!/bin/bash
# Открывает на телевизоре YouTube с видео из буфера обмена (скопируй ссылку и запусти).
cd "$(dirname "$0")"
exec /usr/bin/python3 -c "import tvremote as t; ws=t.connect(); t.open_youtube(ws, t.clipboard() or input('Ссылка: '))"
