//! The Soniox balance and what the translation costs.
//!
//! Soniox gives no way to read the account's balance with an API key, so the
//! user enters it by hand (from the Soniox console). What is spent is counted
//! as it happens: recognition by the audio Soniox reports having processed
//! (`total_audio_proc_ms`), the voice by the seconds of speech it sends back
//! (and the text it was given). Silence the program doesn't send costs nothing.
//!
//! The conversation is counted only while translation is actually asked for:
//! while the key is held, and while the other side's speech is being sent.
//! Its stopwatch stands still otherwise, and only that counts towards its
//! cost. It starts at zero with every launch of the window and on «Сбросить».
//!
//! Kept in ~/.config/pressvoice/usage_soniox.json: the balance entered and
//! what was spent since, today and this month.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{Datelike, Local, NaiveDate};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::config;

/// Soniox prices (soniox.com/pricing, soniox.com/compare-translation):
/// real-time recognition with translation ≈ $0.18 per hour of audio;
/// voice $21.50 per 1M output audio tokens (≈ 30 000 per hour of speech)
/// and $4.00 per 1M input text tokens (≈ 0.3 per character).
pub const STT_PER_HOUR: f64 = 0.18;
pub const TTS_AUDIO_TOKENS_PER_SECOND: f64 = 30_000.0 / 3600.0;
pub const TTS_PRICE_PER_AUDIO_TOKEN: f64 = 21.50 / 1_000_000.0;
pub const TTS_TEXT_TOKENS_PER_CHAR: f64 = 0.3;
pub const TTS_PRICE_PER_TEXT_TOKEN: f64 = 4.00 / 1_000_000.0;
/// Translated speech is spoken at about this many characters a second
/// (for the estimate of time left).
const CHARS_PER_SECOND: f64 = 14.0;

/// Recognition (with translation) of `seconds` of audio.
pub fn stt_cost(seconds: f64) -> f64 {
    seconds * STT_PER_HOUR / 3600.0
}

/// `seconds` of synthesised voice.
pub fn tts_audio_cost(seconds: f64) -> f64 {
    seconds * TTS_AUDIO_TOKENS_PER_SECOND * TTS_PRICE_PER_AUDIO_TOKEN
}

/// The text given to the voice.
pub fn tts_text_cost(chars: usize) -> f64 {
    chars as f64 * TTS_TEXT_TOKENS_PER_CHAR * TTS_PRICE_PER_TEXT_TOKEN
}

/// What one second of a translation costs: recognition, plus the voice
/// when it is heard (`voice`), spoken at `speed` (faster = shorter audio =
/// cheaper) — for the estimate of time left.
pub fn cost_per_second(voice: bool, speed: f32) -> f64 {
    let voice_cost = (tts_audio_cost(1.0) + tts_text_cost(CHARS_PER_SECOND as usize)) / speed.max(0.1) as f64;
    stt_cost(1.0) + if voice { voice_cost } else { 0.0 }
}

/// Written to disk at most this often while money is being spent.
const SAVE_EVERY: Duration = Duration::from_secs(5);

/// The two channels that make a conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// My voice: from key down to the end of the release tail.
    Mine,
    /// The other side: from the start of their speech to the pause after it.
    Theirs,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct Stored {
    /// The balance the user entered, $; `None` until they do.
    balance: Option<f64>,
    /// Spent since it was entered, $.
    spent_since: f64,
    /// When it was entered (for the record).
    entered_at: Option<String>,
    day: Option<NaiveDate>,
    day_usd: f64,
    /// (year, month)
    month: Option<(i32, u32)>,
    month_usd: f64,
}

/// A stopwatch that runs only while a channel is active.
#[derive(Clone, Copy, Debug, Default)]
struct Conversation {
    /// Which channels are active now (`Channel as usize`).
    active: [bool; 2],
    /// Since when at least one is (the stopwatch is running).
    running_since: Option<Instant>,
    /// Time counted before that.
    counted: Duration,
    usd: f64,
}

impl Conversation {
    fn length(&self, now: Instant) -> Duration {
        self.counted + self.running_since.map_or(Duration::ZERO, |t| now.duration_since(t))
    }
}

struct State {
    stored: Stored,
    conversation: Conversation,
    /// Gemini refused for lack of money (cleared when a new balance is entered).
    exhausted: bool,
    dirty: bool,
    saved_at: Instant,
}

/// What the window shows.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// Left on the account, $ (None: no balance entered yet).
    pub balance: Option<f64>,
    /// The conversation since launch or «Сбросить»: the time translation was
    /// asked for, what it cost, and whether the stopwatch is running now.
    pub conversation: (Duration, f64, bool),
    pub today: f64,
    pub month: f64,
    pub exhausted: bool,
}

pub struct Meter {
    state: Mutex<State>,
    /// `None`: not saved (tests, --simulate).
    path: Option<PathBuf>,
}

impl Meter {
    /// The meter of the window, with what was saved last time.
    pub fn load() -> Arc<Self> {
        let path = config::config_dir().join("usage_soniox.json");
        let stored = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Arc::new(Self::with(stored, Some(path)))
    }

    /// Counts but saves nothing.
    pub fn detached() -> Arc<Self> {
        Arc::new(Self::with(Stored::default(), None))
    }

    fn with(stored: Stored, path: Option<PathBuf>) -> Self {
        Self {
            state: Mutex::new(State {
                stored,
                conversation: Conversation::default(),
                exhausted: false,
                dirty: false,
                saved_at: Instant::now(),
            }),
            path,
        }
    }

    /// A channel starts or stops being active (key down / end of the tail;
    /// the other side starts speaking / pauses).
    pub fn set_active(&self, channel: Channel, on: bool) {
        self.set_active_at(channel, on, Instant::now());
    }

    fn set_active_at(&self, channel: Channel, on: bool, now: Instant) {
        let mut st = self.state.lock();
        let c = &mut st.conversation;
        let was_running = c.running_since.is_some();
        c.active[channel as usize] = on;
        let running = c.active.iter().any(|&a| a);
        if running && !was_running {
            c.running_since = Some(now);
        } else if !running && was_running {
            c.counted += now.duration_since(c.running_since.take().expect("running"));
        }
    }

    /// «Сбросить»: the conversation's time and cost start again from zero
    /// (the balance, today and the month stay as they are).
    pub fn reset_conversation(&self) {
        self.reset_conversation_at(Instant::now());
    }

    fn reset_conversation_at(&self, now: Instant) {
        let mut st = self.state.lock();
        let c = &mut st.conversation;
        c.counted = Duration::ZERO;
        c.usd = 0.0;
        if c.running_since.is_some() {
            c.running_since = Some(now);
        }
    }

    /// Money spent just now on translation, $.
    pub fn charge(&self, usd: f64) {
        self.charge_at(usd, Local::now().date_naive());
    }

    fn charge_at(&self, usd: f64, today: NaiveDate) {
        if usd <= 0.0 {
            return;
        }
        let save = {
            let mut st = self.state.lock();
            let s = &mut st.stored;
            if s.day != Some(today) {
                s.day = Some(today);
                s.day_usd = 0.0;
            }
            let month = (today.year(), today.month());
            if s.month != Some(month) {
                s.month = Some(month);
                s.month_usd = 0.0;
            }
            s.day_usd += usd;
            s.month_usd += usd;
            s.spent_since += usd;
            st.conversation.usd += usd;
            st.dirty = true;
            st.saved_at.elapsed() > SAVE_EVERY
        };
        if save {
            self.save();
        }
    }

    /// The balance now on the account (from AI Studio), $.
    pub fn set_balance(&self, usd: f64) {
        {
            let mut st = self.state.lock();
            st.stored.balance = Some(usd.max(0.0));
            st.stored.spent_since = 0.0;
            st.stored.entered_at = Some(Local::now().format("%Y-%m-%d %H:%M").to_string());
            st.exhausted = false;
            st.dirty = true;
        }
        self.save();
    }

    /// Gemini said the money has run out.
    pub fn mark_exhausted(&self) {
        self.state.lock().exhausted = true;
    }

    pub fn snapshot(&self) -> Snapshot {
        self.snapshot_at(Instant::now(), Local::now().date_naive())
    }

    fn snapshot_at(&self, now: Instant, today: NaiveDate) -> Snapshot {
        let st = self.state.lock();
        let s = &st.stored;
        let balance = s.balance.map(|b| (b - s.spent_since).max(0.0));
        let c = st.conversation;
        let conversation = (c.length(now), c.usd, c.running_since.is_some());
        Snapshot {
            balance,
            conversation,
            today: if s.day == Some(today) { s.day_usd } else { 0.0 },
            month: if s.month == Some((today.year(), today.month())) { s.month_usd } else { 0.0 },
            exhausted: st.exhausted,
        }
    }

    /// Writes what changed (also called on exit).
    pub fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let stored = {
            let mut st = self.state.lock();
            if !st.dirty {
                return;
            }
            st.dirty = false;
            st.saved_at = Instant::now();
            st.stored.clone()
        };
        let _ = (|| -> std::io::Result<()> {
            std::fs::create_dir_all(config::config_dir())?;
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, serde_json::to_string_pretty(&stored)?)?;
            std::fs::rename(&tmp, path)
        })();
    }
}

/// "1 ч 58 мин 12 с"
pub fn duration_text(seconds: f64) -> String {
    let s = seconds.max(0.0).floor() as u64;
    let (h, m, s) = (s / 3600, s / 60 % 60, s % 60);
    match (h, m) {
        (0, 0) => format!("{s} с"),
        (0, _) => format!("{m} мин {s:02} с"),
        _ => format!("{h} ч {m:02} мин {s:02} с"),
    }
}

/// "00:12:34"
pub fn clock_text(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// "$4.37" (cents; tenths of a cent below a dollar so per-second changes show)
pub fn money_text(usd: f64) -> String {
    if usd < 1.0 { format!("${usd:.3}") } else { format!("${usd:.2}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 10, d).unwrap()
    }

    #[test]
    fn an_hour_costs_what_soniox_lists() {
        assert!((stt_cost(3600.0) - 0.18).abs() < 1e-9);
        assert!((tts_audio_cost(3600.0) - 0.645).abs() < 1e-9);
        // Voice ≈ $0.70/h with its text; recognition alone ≈ $0.18/h.
        assert!((cost_per_second(true, 1.0) * 3600.0 - 0.885).abs() < 0.02, "{}", cost_per_second(true, 1.0) * 3600.0);
        assert!((cost_per_second(false, 1.0) * 3600.0 - 0.18).abs() < 1e-9);
        // At 1.3× the voice is shorter, so cheaper; the recognition isn't.
        let fast = cost_per_second(true, 1.3) * 3600.0;
        assert!((fast - (0.18 + 0.705 / 1.3)).abs() < 0.02, "{fast}");
    }

    #[test]
    fn the_balance_goes_down_by_what_is_spent() {
        let m = Meter::detached();
        let t0 = Instant::now();
        assert_eq!(m.snapshot_at(t0, day(10)).balance, None);
        m.set_balance(1.0);
        m.set_active_at(Channel::Mine, true, t0);
        for _second in 0..120 {
            m.charge_at(stt_cost(1.0) + tts_audio_cost(1.0), day(10));
        }
        m.set_active_at(Channel::Mine, false, t0 + Duration::from_secs(120));
        let s = m.snapshot_at(t0 + Duration::from_secs(130), day(10));
        let spent = 120.0 * (stt_cost(1.0) + tts_audio_cost(1.0));
        assert!((s.balance.unwrap() - (1.0 - spent)).abs() < 1e-9);
        assert!((s.today - spent).abs() < 1e-9 && (s.month - spent).abs() < 1e-9);
        let (length, usd, running) = s.conversation;
        assert_eq!(length, Duration::from_secs(120), "the 10 s after the press don't count");
        assert!((usd - spent).abs() < 1e-9 && !running);
    }

    #[test]
    fn the_stopwatch_runs_only_while_a_channel_is_active() {
        let m = Meter::detached();
        let t0 = Instant::now();
        let at = |s| t0 + Duration::from_secs(s);
        assert_eq!(m.snapshot_at(t0, day(10)).conversation, (Duration::ZERO, 0.0, false), "zero at launch");
        m.set_active_at(Channel::Mine, true, at(0)); //   I speak 0–10
        m.set_active_at(Channel::Theirs, true, at(5)); // they overlap 5–20
        m.set_active_at(Channel::Mine, false, at(10));
        m.set_active_at(Channel::Theirs, false, at(20));
        // 20–60 quiet: the stopwatch stands at 20 s.
        assert_eq!(m.snapshot_at(at(60), day(10)).conversation.0, Duration::from_secs(20));
        m.set_active_at(Channel::Mine, true, at(60)); //  I speak again 60–63
        let (length, _, running) = m.snapshot_at(at(62), day(10)).conversation;
        assert_eq!(length, Duration::from_secs(22));
        assert!(running);
        m.set_active_at(Channel::Mine, false, at(63));
        // An hour of quiet changes nothing: it is still the same conversation.
        assert_eq!(m.snapshot_at(at(3663), day(10)).conversation.0, Duration::from_secs(23));
    }

    #[test]
    fn reset_starts_the_conversation_over_but_keeps_the_money() {
        let m = Meter::detached();
        let t0 = Instant::now();
        let at = |s| t0 + Duration::from_secs(s);
        m.set_balance(2.0);
        m.set_active_at(Channel::Mine, true, at(0));
        m.charge_at(stt_cost(10.0), day(10));
        m.reset_conversation_at(at(10)); // pressed while I'm speaking
        assert_eq!(m.snapshot_at(at(10), day(10)).conversation, (Duration::ZERO, 0.0, true));
        m.set_active_at(Channel::Mine, false, at(13));
        let s = m.snapshot_at(at(20), day(10));
        assert_eq!(s.conversation.0, Duration::from_secs(3), "counts on from the reset");
        assert!((s.balance.unwrap() - (2.0 - stt_cost(10.0))).abs() < 1e-12, "the balance is untouched");
        assert!((s.today - stt_cost(10.0)).abs() < 1e-12);
    }

    #[test]
    fn a_new_balance_starts_over() {
        let m = Meter::detached();
        m.set_balance(1.0);
        m.charge_at(1.0, day(10));
        m.mark_exhausted();
        m.set_balance(10.0);
        let s = m.snapshot_at(Instant::now(), day(10));
        assert_eq!(s.balance, Some(10.0));
        assert!(!s.exhausted);
        assert!(s.today > 0.0, "the day's spending is kept");
    }

    #[test]
    fn days_and_months_roll_over() {
        let m = Meter::detached();
        let t0 = Instant::now();
        m.charge_at(stt_cost(1.0), day(10));
        m.charge_at(stt_cost(1.0), day(11));
        let s = m.snapshot_at(t0, day(11));
        assert!((s.today - stt_cost(1.0)).abs() < 1e-12, "a new day starts at zero");
        assert!((s.month - 2.0 * stt_cost(1.0)).abs() < 1e-12, "the month goes on");
    }

    #[test]
    fn texts() {
        assert_eq!(duration_text(7092.4), "1 ч 58 мин 12 с");
        assert_eq!(duration_text(125.0), "2 мин 05 с");
        assert_eq!(duration_text(9.9), "9 с");
        assert_eq!(clock_text(Duration::from_secs(754)), "00:12:34");
        assert_eq!(money_text(4.371), "$4.37");
        assert_eq!(money_text(0.3812), "$0.381");
    }
}
