#!/bin/bash
# Собирает виртуальный микрофон «PressVoice Microphone» в driver/.build/PressVoice.driver.
#
# Это BlackHole (Existential Audio, GPL-3.0) со своими названиями, без Xcode — один файл
# на C и обычный clang. Два связанных устройства с общим кольцевым буфером:
#   «PressVoice Microphone»        — только вход, видно всем (его выбирают в Teams/Zoom);
#   «PressVoice Microphone (feed)» — только выход, скрыто; в него пишет PressVoice
#                                    (по UID «PressVoice_2_UID»).
# Громкость устройства отключена: автоподстройка микрофона в Teams не приглушит перевод.
# Подпись «местная» (ad-hoc): работает на этом Mac; для раздачи нужна подпись Developer ID.
set -euo pipefail
cd "$(dirname "$0")"

TAG=v0.7.2
SRC=.build/BlackHole
OUT=.build/PressVoice.driver

if [ ! -d "$SRC" ]; then
    git clone -q --depth 1 --branch "$TAG" https://github.com/ExistentialAudio/BlackHole.git "$SRC"
fi

# The one name the build flags don't reach: the "box" (seen in System Information).
sed 's/CFSTR("BlackHole Box")/CFSTR(kDriver_Name " Box")/' "$SRC/BlackHole/BlackHole.c" > .build/PressVoice.c
grep -q 'kDriver_Name " Box"' .build/PressVoice.c || { echo "box name patch failed"; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
clang -bundle -O2 -std=gnu11 -arch arm64 -arch x86_64 -mmacosx-version-min=12.0 \
    -DkDriver_Name='"PressVoice"' \
    -DkPlugIn_BundleID='"com.pressvoice.microphone"' \
    -DkPlugIn_Icon='"PressVoice.icns"' \
    -DkHas_Driver_Name_Format=false \
    -DkManufacturer_Name='"PressVoice"' \
    -DkDevice_Name='"PressVoice Microphone"' \
    -DkDevice_IsHidden=false -DkDevice_HasInput=true -DkDevice_HasOutput=false \
    -DkDevice2_Name='"PressVoice Microphone (feed)"' \
    -DkDevice2_IsHidden=true -DkDevice2_HasInput=false -DkDevice2_HasOutput=true \
    -DkEnableVolumeControl=false \
    -DkCanBeDefaultSystemDevice=false \
    -framework CoreAudio -framework CoreFoundation -framework Accelerate \
    -Wno-format-extra-args \
    -o "$OUT/Contents/MacOS/PressVoice" .build/PressVoice.c
cp Info.plist "$OUT/Contents/Info.plist"
cp ../assets/pressvoice.icns "$OUT/Contents/Resources/PressVoice.icns"
codesign --force --sign - "$OUT"
echo "Собрано: driver/$OUT"
