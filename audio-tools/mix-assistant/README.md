# Mix Assistant (Claude skill)

A Claude Code skill for mixing **vocals + acoustic guitar** in Cubase when the model "listens" through measurements instead of ears.

* `scripts/mix_analyzer.py` — LUFS, true peak, tonal balance, sibilance, plosives, vocal consistency, guitar masking, phase; comparison with reference tracks; PNG plots.
* `scripts/mix_engine.py` — offline mixing with real VST3 plugins driven by a JSON config (`mix_config.example.json`).
* `references/` — signal chains, target values, plugin notes and a guide for transferring settings into Cubase.

Install: copy the folder to `~/.claude/skills/acoustic-vocal-mix`, then `pip install -r scripts/requirements.txt`. See `SKILL.md` for the workflow.
