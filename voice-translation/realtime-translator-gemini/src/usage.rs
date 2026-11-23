//! The Gemini balance and what the translation costs.
//!
//! Google gives no way to read the account's balance with an API key (only
//! AI Studio's Billing page shows it), so the user enters it by hand. What is
//! spent comes from Gemini itself: every session reports the tokens it used
//! about once a second (`usageMetadata`, in increments — measured: 25 tokens
//! in and 25 out per second of audio sent). Silence the program doesn't send
//! costs nothing.
//!
//! The conversation is counted only while translation is actually asked for:
//! while the key is held, and while the other side's speech is being sent.
//! Its stopwatch stands still otherwise, and only that audio counts towards
//! its cost. It starts at zero with every launch of the window and on «Сбросить».
//! Service audio — a new session's warm-up, a spare's — is billed by Google
//! too, so it comes off the balance, but apart from the conversation.
//!
//! Kept in ~/.config/pressvoice/usage.json: the balance entered and what was
//! spent since, today and this month.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{Datelike, Local, NaiveDate};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::config;

/// Gemini 3.5 Live Translate prices (ai.google.dev/gemini-api/docs/pricing):
/// $3.50 per 1M audio tokens in, $21.00 per 1M audio tokens out.
pub const PRICE_IN_PER_TOKEN: f64 = 3.50 / 1_000_000.0;
pub const PRICE_OUT_PER_TOKEN: f64 = 21.00 / 1_000_000.0;
/// Audio tokens per second, each way.
pub const TOKENS_PER_SECOND: f64 = 25.0;
/// What one second of translation costs (≈ $0.0368 a minute).
pub const COST_PER_SECOND: f64 = TOKENS_PER_SECOND * (PRICE_IN_PER_TOKEN + PRICE_OUT_PER_TOKEN);
/// Written to disk at most this often while money is being spent.
const SAVE_EVERY: Duration = Duration::from_secs(5);

/// What spent tokens were for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Translating a press or the other side's speech.
    Talk,
    /// A session's warm-up (or an unused spare's).
    Service,
}

/// The two channels that make a conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// My voice: from key down to the end of the release tail.
    Mine,
    /// The other side: from the start of their speech to the pause after it.
    Theirs,
}

pub fn cost(input_tokens: u64, output_tokens: u64) -> f64 {
    input_tokens as f64 * PRICE_IN_PER_TOKEN + output_tokens as f64 * PRICE_OUT_PER_TOKEN
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct Stored {
    /// The balance the user entered, $; `None` until they do.
    balance: Option<f64>,
    /// Spent since it was entered, $.
    spent_since: f64,
    /// …of which on service audio (warm-ups), $.
    service_since: f64,
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
    /// Seconds of ONE translation (one channel) that money buys.
    pub seconds_left: Option<f64>,
    /// The conversation since launch or «Сбросить»: the time translation was
    /// asked for, what it cost, and whether the stopwatch is running now.
    pub conversation: (Duration, f64, bool),
    /// Spent on service audio (warm-ups) since the balance was entered, $.
    pub service: f64,
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
        let path = config::config_dir().join("usage.json");
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

    /// Tokens a session reported spending (an increment).
    pub fn charge(&self, input_tokens: u64, output_tokens: u64, kind: Kind) {
        self.charge_at(input_tokens, output_tokens, kind, Local::now().date_naive());
    }

    fn charge_at(&self, input_tokens: u64, output_tokens: u64, kind: Kind, today: NaiveDate) {
        let usd = cost(input_tokens, output_tokens);
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
            match kind {
                Kind::Service => s.service_since += usd,
                // Reports come ~1 s late: the end of a phrase still belongs to it.
                Kind::Talk => st.conversation.usd += usd,
            }
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
            st.stored.service_since = 0.0;
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
            seconds_left: balance.map(|b| b / COST_PER_SECOND),
            conversation,
            today: if s.day == Some(today) { s.day_usd } else { 0.0 },
            month: if s.month == Some((today.year(), today.month())) { s.month_usd } else { 0.0 },
            exhausted: st.exhausted,
            service: s.service_since,
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

/// Gemini's refusal looks like the account is out of money (HTTP 402, or 429
/// "prepayment credits are depleted").
pub fn is_out_of_money(error: &str) -> bool {
    let e = error.to_lowercase();
    e.contains("402") || e.contains("payment") || e.contains("prepay") || e.contains("credits") || e.contains("billing")
}

/// Gemini refused for quota (429 / RESOURCE_EXHAUSTED) — limits, or money.
pub fn is_quota(error: &str) -> bool {
    let e = error.to_lowercase();
    e.contains("429") || e.contains("resource_exhausted") || e.contains("quota")
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
    fn a_minute_costs_what_google_lists() {
        // 60 s × 25 tokens each way.
        let minute = cost(1500, 1500);
        assert!((minute - 0.03675).abs() < 1e-9, "{minute}");
        assert!((COST_PER_SECOND * 60.0 - minute).abs() < 1e-12);
    }

    #[test]
    fn the_balance_goes_down_by_what_is_reported() {
        let m = Meter::detached();
        let t0 = Instant::now();
        assert_eq!(m.snapshot_at(t0, day(10)).balance, None);
        m.set_balance(5.0);
        m.set_active_at(Channel::Mine, true, t0);
        for _second in 0..120 {
            // two minutes of reports, one a second
            m.charge_at(25, 25, Kind::Talk, day(10));
        }
        m.set_active_at(Channel::Mine, false, t0 + Duration::from_secs(120));
        let s = m.snapshot_at(t0 + Duration::from_secs(130), day(10));
        let spent = 2.0 * 0.03675;
        assert!((s.balance.unwrap() - (5.0 - spent)).abs() < 1e-9);
        assert!((s.seconds_left.unwrap() - (5.0 - spent) / COST_PER_SECOND).abs() < 1e-6);
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
        m.charge_at(250, 250, Kind::Talk, day(10));
        m.reset_conversation_at(at(10)); // pressed while I'm speaking
        assert_eq!(m.snapshot_at(at(10), day(10)).conversation, (Duration::ZERO, 0.0, true));
        m.set_active_at(Channel::Mine, false, at(13));
        let s = m.snapshot_at(at(20), day(10));
        assert_eq!(s.conversation.0, Duration::from_secs(3), "counts on from the reset");
        assert!((s.balance.unwrap() - (2.0 - cost(250, 250))).abs() < 1e-12, "the balance is untouched");
        assert!((s.today - cost(250, 250)).abs() < 1e-12);
    }

    #[test]
    fn warm_ups_come_off_the_balance_but_not_the_conversation() {
        let m = Meter::detached();
        m.set_balance(1.0);
        m.charge_at(100, 100, Kind::Service, day(10));
        m.set_active_at(Channel::Mine, true, Instant::now());
        m.charge_at(25, 25, Kind::Talk, day(10));
        let s = m.snapshot_at(Instant::now(), day(10));
        assert!((s.balance.unwrap() - (1.0 - cost(125, 125))).abs() < 1e-12);
        assert!((s.service - cost(100, 100)).abs() < 1e-12);
        assert!((s.conversation.1 - cost(25, 25)).abs() < 1e-12);
    }

    #[test]
    fn a_new_balance_starts_over() {
        let m = Meter::detached();
        m.set_balance(1.0);
        m.charge_at(25_000, 25_000, Kind::Talk, day(10));
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
        m.charge_at(25, 25, Kind::Talk, day(10));
        m.charge_at(25, 25, Kind::Service, day(11));
        let s = m.snapshot_at(t0, day(11));
        assert!((s.today - cost(25, 25)).abs() < 1e-12, "a new day starts at zero");
        assert!((s.month - 2.0 * cost(25, 25)).abs() < 1e-12, "the month goes on");
    }

    #[test]
    fn refusals_are_recognised() {
        assert!(is_out_of_money("Gemini refused: HTTP 402 Payment Required"));
        assert!(is_out_of_money("Your prepayment credits are depleted"));
        assert!(is_quota("429 RESOURCE_EXHAUSTED"));
        assert!(!is_out_of_money("connection reset") && !is_quota("connection reset"));
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
