# Diction Fixer

Fixes "swallowed" consonants in AI-generated vocal stems (e.g. Suno → Kits).

1. Force-aligns the lyrics to the vocal track letter by letter (a Ukrainian phoneme recognizer + forced alignment).
2. Scores how clearly every consonant is heard.
3. For weak letters, finds a clear take of the same letter elsewhere in the song (or in your donor recordings) and blends it in: matched loudness, pitch for voiced sounds, smooth edges.
4. Writes a corrected WAV of the same length and format, a per-letter timing table, a list of fixes and DAW markers.

```bash
pip install -r requirements.txt
python diction_fix.py vocal.wav lyrics.txt
python diction_fix.py vocal.wav lyrics.txt --letters рдт --threshold 0.4
python bukvi_app.py        # GUI: waveform, letter list, manual fixes
```

Audio and lyrics are not included: use your own stem and text file.
