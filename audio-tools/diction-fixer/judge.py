import sys, warnings, numpy as np, soundfile as sf
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import diction_fix as dfx
MODELS = {}
def judge(wav, model, lyrics="test/lyrics.txt", words_of_interest=("град", "дорога"), letter="р"):
    lm = MODELS.setdefault(model, dfx.LetterModel(model, "cpu"))
    a, sr = sf.read(wav, dtype="float32", always_2d=True)
    m16 = dfx.to_mono16k(a, sr)
    words = dfx.parse_lyrics(open(lyrics, encoding="utf-8").read(), lm.letters)
    targets, meta = [], []
    for wi, (ln, w) in enumerate(words):
        if wi and lm.word_sep is not None: targets.append(lm.word_sep); meta.append(None)
        for ci, ch in enumerate(w): targets.append(lm.vocab[ch]); meta.append((wi, ch))
    lp, _ = lm.emissions_hires(m16)
    ft = dfx.ctc_force_align(lp, np.array(targets), lm.blank)
    pr = np.exp(lp)
    res, weak = [], []
    for k, mm in enumerate(meta):
        if not mm: continue
        fr = np.nonzero(ft == k)[0]
        c = float(pr[max(0, fr[0]-1):fr[-1]+2, targets[k]].max())
        w = words[mm[0]][1]
        if w in words_of_interest and mm[1] == letter: res.append(round(c, 2))
        if c < 0.35 and mm[1] in dfx.CONSONANTS and fr[0] > 3: weak.append(f"{w}:{mm[1]}")
    inv = {v: k for k, v in lm.vocab.items()}
    g = lp.argmax(1); txt = []; prev = None
    for i in g:
        if i != prev and i != lm.blank: txt.append(inv[int(i)])
        prev = i
    return res, weak, "".join(txt).replace("|", " ")
if __name__ == "__main__":
    for wav in sys.argv[1:]:
        for model in ["robinhad/wav2vec2-xls-r-300m-uk", "Yehor/w2v-bert-uk-v2.1"]:
            r, w, t = judge(wav, model)
            print(f"{wav.split('/')[-1][:28]:28s} {model.split('/')[-1][:10]:10s} р={r} слабые={w}")
            print(f"{'':39s} слышит: {t[:75]}")
