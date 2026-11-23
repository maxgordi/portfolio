#!/bin/bash
# Двойной клик: окно переводчика на Soniox (распознавание с переводом + голоса Soniox).
# При первом запуске программа собирается (~3 мин), потом стартует сразу.
# Окна Gemini и Soniox слушают одну клавишу: при запуске одно закрывает другое.
cd "$(dirname "$0")"
CARGO="$HOME/.cargo/bin/cargo"
if [ ! -x "$CARGO" ]; then
    echo "Не найден Rust ($CARGO). Установка: https://rustup.rs"
    read -n 1 -s -r -p "Нажмите любую клавишу…"; exit 1
fi
echo "Проверяю, свежая ли сборка (после изменений в коде пересборка ~1 мин, окно откроется само)…"
"$CARGO" build --release || { read -n 1 -s -r -p "Сборка не удалась. Нажмите любую клавишу…"; exit 1; }
echo "Открываю окно…"
exec ./target/release/pressvoice-soniox "$@"
