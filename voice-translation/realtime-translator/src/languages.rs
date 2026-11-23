//! Languages Soniox recognises and translates (stt-rt-v5, 60 codes,
//! read from api.soniox.com/v1/models on 2026-10-10), with the names shown
//! in the window. Every Soniox voice speaks all of them.

/// (code, name shown in the window)
pub const NAMES: [(&str, &str); 60] = [
    ("af", "африкаанс"),
    ("sq", "албанский"),
    ("ar", "арабский"),
    ("az", "азербайджанский"),
    ("eu", "баскский"),
    ("be", "белорусский"),
    ("bn", "бенгальский"),
    ("bs", "боснийский"),
    ("bg", "болгарский"),
    ("ca", "каталанский"),
    ("zh", "китайский"),
    ("hr", "хорватский"),
    ("cs", "чешский"),
    ("da", "датский"),
    ("nl", "нидерландский"),
    ("en", "английский"),
    ("et", "эстонский"),
    ("fi", "финский"),
    ("fr", "французский"),
    ("gl", "галисийский"),
    ("de", "немецкий"),
    ("el", "греческий"),
    ("gu", "гуджарати"),
    ("he", "иврит"),
    ("hi", "хинди"),
    ("hu", "венгерский"),
    ("id", "индонезийский"),
    ("it", "итальянский"),
    ("ja", "японский"),
    ("kn", "каннада"),
    ("kk", "казахский"),
    ("ko", "корейский"),
    ("lv", "латышский"),
    ("lt", "литовский"),
    ("mk", "македонский"),
    ("ms", "малайский"),
    ("ml", "малаялам"),
    ("mr", "маратхи"),
    ("no", "норвежский"),
    ("fa", "персидский"),
    ("pl", "польский"),
    ("pt", "португальский"),
    ("pa", "панджаби"),
    ("ro", "румынский"),
    ("ru", "русский"),
    ("sr", "сербский"),
    ("sk", "словацкий"),
    ("sl", "словенский"),
    ("es", "испанский"),
    ("sw", "суахили"),
    ("sv", "шведский"),
    ("tl", "тагальский (филиппинский)"),
    ("ta", "тамильский"),
    ("te", "телугу"),
    ("th", "тайский"),
    ("tr", "турецкий"),
    ("uk", "украинский"),
    ("ur", "урду"),
    ("vi", "вьетнамский"),
    ("cy", "валлийский"),
];

/// The 16 most used, on top: English and Ukrainian first, then Europe, then Asia.
pub const POPULAR: [&str; 16] = [
    "en", "uk", "de", "fr", "es", "it", "pt", "pl", "nl", "tr", "ru", "ar", "hi", "ko", "ja", "zh",
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

/// A code saved by the Gemini version (pt-BR, zh-Hans…) in Soniox's form.
pub fn soniox_code(code: &str) -> String {
    code.split('-').next().unwrap_or(code).to_lowercase()
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
    fn gemini_codes_are_mapped() {
        assert_eq!(soniox_code("pt-BR"), "pt");
        assert_eq!(soniox_code("zh-Hans"), "zh");
        assert!(name(&soniox_code("pt-BR")).is_some());
    }
}
