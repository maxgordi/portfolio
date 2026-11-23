#!/bin/bash
# Двойной клик — экран Mac на телевизоре LG. Закрой окно — трансляция остановится.
cd "$(dirname "$0")"
exec /usr/bin/python3 tvcast.py
