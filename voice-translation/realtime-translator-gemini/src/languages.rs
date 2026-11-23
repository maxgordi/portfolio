//! Languages Gemini 3.5 Live Translate speaks (`targetLanguageCode`), as
//! listed at https://ai.google.dev/gemini-api/docs/live-api/live-translate
//! (78 codes). The API accepts ANY code at setup — even a made-up one — so
//! only this list says which really work. The spoken language needs no
//! setting: the model detects it.

/// (code, name shown in the window)
pub const NAMES: [(&str, &str); 78] = [
    ("af", "африкаанс"),
    ("ak", "акан"),
    ("sq", "албанский"),
    ("am", "амхарский"),
    ("ar", "арабский"),
    ("hy", "армянский"),
    ("az", "азербайджанский"),
    ("eu", "баскский"),
    ("be", "белорусский"),
    ("bn", "бенгальский"),
    ("bg", "болгарский"),
    ("my", "бирманский"),
    ("ca", "каталанский"),
    ("zh-Hans", "китайский (упрощённый)"),
    ("zh-Hant", "китайский (традиционный)"),
    ("hr", "хорватский"),
    ("cs", "чешский"),
    ("da", "датский"),
    ("nl", "нидерландский"),
    ("en", "английский"),
    ("et", "эстонский"),
    ("fil", "филиппинский"),
    ("fi", "финский"),
    ("fr", "французский"),
    ("gl", "галисийский"),
    ("ka", "грузинский"),
    ("de", "немецкий"),
    ("el", "греческий"),
    ("gu", "гуджарати"),
    ("ha", "хауса"),
    ("he", "иврит"),
    ("hi", "хинди"),
    ("hu", "венгерский"),
    ("is", "исландский"),
    ("id", "индонезийский"),
    ("it", "итальянский"),
    ("ja", "японский"),
    ("jv", "яванский"),
    ("kn", "каннада"),
    ("kk", "казахский"),
    ("km", "кхмерский"),
    ("rw", "киньяруанда"),
    ("ko", "корейский"),
    ("lo", "лаосский"),
    ("lv", "латышский"),
    ("lt", "литовский"),
    ("mk", "македонский"),
    ("ms", "малайский"),
    ("ml", "малаялам"),
    ("mr", "маратхи"),
    ("mn", "монгольский"),
    ("ne", "непальский"),
    ("no", "норвежский"),
    ("fa", "персидский"),
    ("pl", "польский"),
    ("pt-BR", "португальский (Бразилия)"),
    ("pt-PT", "португальский (Португалия)"),
    ("pa", "панджаби"),
    ("ro", "румынский"),
    ("ru", "русский"),
    ("sr", "сербский"),
    ("sd", "синдхи"),
    ("si", "сингальский"),
    ("sk", "словацкий"),
    ("sl", "словенский"),
    ("es", "испанский"),
    ("su", "сунданский"),
    ("sw", "суахили"),
    ("sv", "шведский"),
    ("ta", "тамильский"),
    ("te", "телугу"),
    ("th", "тайский"),
    ("tr", "турецкий"),
    ("uk", "украинский"),
    ("ur", "урду"),
    ("uz", "узбекский"),
    ("vi", "вьетнамский"),
    ("zu", "зулу"),
];

/// The 16 most used, on top: English and Ukrainian first, then Europe, then Asia.
pub const POPULAR: [&str; 16] = [
    "en", "uk", "de", "fr", "es", "it", "pt-BR", "pl", "nl", "tr", "ru", "ar", "hi", "ko", "ja",
    "zh-Hans",
];

pub fn name(code: &str) -> Option<&'static str> {
    NAMES.iter().find(|(c, _)| *c == code).map(|(_, n)| *n)
}

/// `(code, name)`
pub type Language = (&'static str, &'static str);

/// (popular, the rest alphabetically).
pub fn ordered() -> (Vec<Language>, Vec<Language>) {
    let top = POPULAR.iter().map(|&c| (c, name(c).unwrap_or(c))).collect();
    let mut rest: Vec<_> = NAMES.iter().copied().filter(|(c, _)| !POPULAR.contains(c)).collect();
    rest.sort_by_key(|(_, n)| n.replace('ё', "е"));
    (top, rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_language_is_listed_once() {
        let (top, rest) = ordered();
        assert_eq!(top.len() + rest.len(), NAMES.len());
        assert!(POPULAR.iter().all(|c| name(c).is_some()));
        assert_eq!(top[0], ("en", "английский"));
    }

    #[test]
    fn the_rest_is_alphabetical() {
        let (_, rest) = ordered();
        let names: Vec<String> = rest.iter().map(|(_, n)| n.replace('ё', "е")).collect();
        assert!(names.windows(2).all(|w| w[0] <= w[1]));
    }
}
