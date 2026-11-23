//! Fonts for every language the translator speaks.
//!
//! The interface font is San Francisco (the macOS one). It has Latin, Cyrillic
//! and Greek only, so each other script comes from the macOS font made for it
//! — Georgian, Arabic, Chinese, Devanagari… — in a fallback chain: a letter
//! missing from one font is taken from the next. Arial Unicode closes the chain.
//! egui shapes text with HarfBuzz, so joined scripts (Arabic, Indic) come out right.
//!
//! The files are memory-mapped rather than read: the CJK and Korean fonts are
//! tens of MB, and only the pages actually drawn get loaded.

use std::fs::File;
use std::sync::Arc;

use eframe::egui::{self, FontData, FontDefinitions, FontFamily};
use eframe::epaint::text::VariationCoords;

const FONTS: &str = "/System/Library/Fonts";
const MORE: &str = "/System/Library/Fonts/Supplemental";

/// One face of a font file.
struct Face {
    name: &'static str,
    dir: &'static str,
    file: &'static str,
    /// Which face of a collection (.ttc) is the regular one.
    index: u32,
}

const fn face(name: &'static str, dir: &'static str, file: &'static str, index: u32) -> Face {
    Face { name, dir, file, index }
}

/// In order of preference.
const FACES: [Face; 29] = [
    face("San Francisco", FONTS, "SFNS.ttf", 0), // Latin, Cyrillic, Greek
    face("Apple Symbols", FONTS, "Apple Symbols.ttf", 0), // ⌥ ⌃ ⇧ ⌘
    face("SF Georgian", FONTS, "SFGeorgian.ttf", 0),
    face("SF Armenian", FONTS, "SFArmenian.ttf", 0),
    face("SF Hebrew", FONTS, "SFHebrew.ttf", 0),
    face("SF Arabic", FONTS, "SFArabic.ttf", 0), // Arabic, Persian, Urdu, Sindhi
    face("Hiragino Sans GB", FONTS, "Hiragino Sans GB.ttc", 0), // Chinese, Japanese kana
    face("Apple SD Gothic Neo", FONTS, "AppleSDGothicNeo.ttc", 0), // Korean
    face("Kohinoor Devanagari", FONTS, "Kohinoor.ttc", 0), // Hindi, Marathi, Nepali
    face("Kohinoor Bangla", FONTS, "KohinoorBangla.ttc", 0),
    face("Kohinoor Gujarati", FONTS, "KohinoorGujarati.ttc", 3),
    face("Kohinoor Telugu", FONTS, "KohinoorTelugu.ttc", 0),
    face("Gurmukhi MN", MORE, "Gurmukhi MN.ttc", 0), // Punjabi
    face("Noto Sans Kannada", FONTS, "NotoSansKannada.ttc", 3),
    face("Malayalam MN", MORE, "Malayalam MN.ttc", 0),
    face("Tamil MN", MORE, "Tamil MN.ttc", 0),
    face("Sinhala MN", MORE, "Sinhala MN.ttc", 0),
    face("Thonburi", FONTS, "ThonburiUI.ttc", 0), // Thai
    face("Lao MN", MORE, "Lao MN.ttc", 0),
    face("Khmer MN", MORE, "Khmer MN.ttc", 0),
    face("Noto Sans Myanmar", FONTS, "NotoSansMyanmar.ttc", 6), // Burmese
    face("Kefa", MORE, "Kefa.ttc", 0), // Amharic (Ethiopic)
    face("Noto Sans Mongolian", MORE, "NotoSansMongolian-Regular.ttf", 0),
    face("Oriya MN", MORE, "Oriya MN.ttc", 0),
    face("Kannada MN", MORE, "Kannada MN.ttc", 0),
    face("Telugu MN", MORE, "Telugu MN.ttc", 0),
    face("Gujarati MT", MORE, "GujaratiMT.ttc", 0),
    face("Myanmar MN", MORE, "Myanmar MN.ttc", 0),
    face("Arial Unicode", MORE, "Arial Unicode.ttf", 0), // everything else
];

/// San Francisco's regular weight (its file is a variable font).
const WEIGHT_REGULAR: f32 = 400.0;

/// The bytes of a system font file, mapped into memory for the program's
/// lifetime (system fonts don't change while it runs).
fn map(face: &Face) -> Option<&'static [u8]> {
    let file = File::open(format!("{}/{}", face.dir, face.file)).ok()?;
    let mapped = unsafe { memmap2::Mmap::map(&file) }.ok()?;
    Some(Box::leak(Box::new(mapped)))
}

pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let mut chain = Vec::new();
    for face in &FACES {
        let Some(bytes) = map(face) else {
            continue; // a font missing on this Mac: the next ones still cover most
        };
        let mut data = FontData::from_static(bytes);
        data.index = face.index;
        if face.name == "San Francisco" {
            data.tweak.coords = VariationCoords::new([("wght", WEIGHT_REGULAR)]);
        }
        fonts.font_data.insert(face.name.to_owned(), Arc::new(data));
        chain.push(face.name.to_owned());
    }
    // Ours first; egui's own fonts stay at the end of the chain.
    let proportional = fonts.families.entry(FontFamily::Proportional).or_default();
    proportional.splice(0..0, chain);
    ctx.set_fonts(fonts);
}

#[cfg(test)]
mod tests {
    use super::*;
    use skrifa::{FontRef, MetadataProvider};

    /// The name of each of the translator's 78 languages in its own script.
    const AUTONYMS: [&str; 78] = [
        "Afrikaans", "Akan ɛɔ", "Shqip ëç", "አማርኛ", "العربية", "Հայերեն", "Azərbaycan şğı",
        "Euskara", "Беларуская ў", "বাংলা", "Български", "မြန်မာ", "Català", "简体中文",
        "繁體中文", "Hrvatski", "Čeština ěř", "Dansk æøå", "Nederlands", "English", "Eesti õäöü",
        "Filipino", "Suomi", "Français œ", "Galego", "ქართული", "Deutsch ß", "Ελληνικά",
        "ગુજરાતી", "Hausa ɓɗƙ", "עברית", "हिन्दी", "Magyar őű", "Íslenska þð", "Indonesia",
        "Italiano", "日本語 ひらがな カタカナ", "Basa Jawa", "ಕನ್ನಡ", "Қазақ ғұ", "ខ្មែរ",
        "Kinyarwanda", "한국어", "ລາວ", "Latviešu", "Lietuvių", "Македонски ѓќ", "Melayu",
        "മലയാളം", "मराठी", "Монгол өү", "नेपाली", "Norsk", "فارسی", "Polski ąęłńśźż",
        "Português ãç", "Português", "ਪੰਜਾਬੀ", "Română ăâîșț", "Русский ё", "Српски ђћ",
        "سنڌي", "සිංහල", "Slovenčina", "Slovenščina", "Español ñ", "Basa Sunda", "Kiswahili",
        "Svenska", "தமிழ்", "తెలుగు", "ไทย", "Türkçe ğış", "Українська ґєії", "اردو",
        "Oʻzbek", "Tiếng Việt ế ệ", "isiZulu",
    ];

    #[test]
    fn every_letter_of_every_language_has_a_font() {
        let files: Vec<(&Face, Vec<u8>)> = FACES
            .iter()
            .filter_map(|f| std::fs::read(format!("{}/{}", f.dir, f.file)).ok().map(|bytes| (f, bytes)))
            .collect();
        assert!(files.len() >= FACES.len() - 2, "only {} of the fonts found", files.len());
        let fonts: Vec<FontRef> = files.iter().map(|(f, bytes)| FontRef::from_index(bytes, f.index).expect("a font")).collect();
        let missing: Vec<String> = AUTONYMS
            .iter()
            .flat_map(|text| text.chars())
            .filter(|c| !c.is_whitespace())
            .filter(|&c| !fonts.iter().any(|font| font.charmap().map(c).is_some()))
            .map(|c| format!("{c} (U+{:04X})", c as u32))
            .collect();
        assert!(missing.is_empty(), "no font for: {}", missing.join(", "));
    }

    /// The same, as egui itself sees it (no window needed).
    #[test]
    fn egui_draws_every_language() {
        let ctx = egui::Context::default();
        install(&ctx);
        let _ = ctx.run_ui(Default::default(), |_| {}); // fonts load on the first frame
        let font = egui::FontId::proportional(16.0);
        // The check itself works: a code point no font has is reported missing.
        assert!(!ctx.fonts_mut(|f| f.has_glyphs(&font, "\u{10FFFD}")));
        let missing: Vec<&str> = AUTONYMS
            .iter()
            .copied()
            .filter(|text| !ctx.fonts_mut(|f| f.has_glyphs(&font, &text.replace(' ', ""))))
            .collect();
        assert!(missing.is_empty(), "egui has no glyphs for: {missing:?}");
    }

    #[test]
    fn san_francisco_covers_the_interface() {
        let face = &FACES[0];
        let bytes = std::fs::read(format!("{}/{}", face.dir, face.file)).expect("SFNS.ttf");
        let font = FontRef::from_index(&bytes, 0).expect("a font");
        for c in "Язык перевода: English, Українська — ⌥".chars().filter(|c| *c != '⌥') {
            assert!(font.charmap().map(c).is_some(), "{c}");
        }
    }
}
