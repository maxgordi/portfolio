//! The window. On top: my language and the other side's. Below, two boxed
//! sections, each a channel of its own:
//! - «Я → собеседник»: hold the key and speak; my voice, translated, goes to
//!   the headphones or into the call (the virtual microphone);
//! - «Собеседник → я»: the call app's sound, translated into my language,
//!   plays in my headphones (switched on and off with its checkbox).
//!
//! At the bottom, the live text of all four: what each side says and hears.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui::{self, RichText};
use parking_lot::Mutex;

use crate::app_audio::{self, App, Source};
use crate::audio;
use crate::config::WindowSettings;
use crate::hotkey::{Hotkey, KEYS};
use crate::incoming::{Incoming, IncomingOptions, State};
use crate::journal;
use crate::languages;
use crate::translator::{Options, Pane, TextSink, Translator};
use crate::usage::{self, Meter};
use crate::virtual_mic;
use crate::voice_mix::MixSettings;

const TITLE: &str = "PressVoice Live — синхронный переводчик";
const MODEL_LABEL: &str = "Gemini 3.5 Live Translate";
/// (label, semitones the voice is lowered by before the model hears it)
const DEEPER: [(&str, i32); 5] =
    [("выключена", 0), ("лёгкая", 2), ("средняя", 3), ("сильная", 4), ("максимальная", 6)];
const DEEPER_TIP: &str = "Голос немного понижается перед отправкой в модель. Переводчик повторяет \
                          тембр говорящего, поэтому отвечает более низким и ровным голосом. \
                          Задержку и качество перевода не меняет.";
const SYSTEM_DEVICE: &str = "Системный по умолчанию";
const ALL_APPS: &str = "Все программы";
/// (label, % of a voice under the translation)
const VOICE_LEVELS: [(&str, i32); 11] = [
    ("выключен — только перевод", 0), ("10%", 10), ("20%", 20), ("30%", 30), ("40%", 40), ("50%", 50),
    ("60%", 60), ("70%", 70), ("80%", 80), ("90%", 90), ("100%", 100),
];
/// (label, % of the voice left while the translation sounds)
const DUCK_LEVELS: [(&str, i32); 9] = [
    ("10%", 10), ("20%", 20), ("30%", 30), ("40%", 40), ("50%", 50), ("60%", 60), ("70%", 70), ("80%", 80), ("90%", 90),
];
/// (label, id, how long after the translation the voice stays down)
const DUCK_SPEEDS: [(&str, &str, Duration); 3] = [
    ("быстро — 0,3 с", "fast", Duration::from_millis(300)),
    ("средне — 0,7 с", "medium", Duration::from_millis(700)),
    ("плавно — 1,5 с", "smooth", Duration::from_millis(1500)),
];
const INTO_CALL_TIP: &str = "Работает, когда «Вывод перевода» — PressVoice Microphone: собеседник слышит \
                             ваш живой голос, пока вы держите клавишу, а под переводом — тише.";
const SOURCE_TIP: &str = "Откуда брать голос собеседника. В самой программе звонка ничего настраивать не нужно. \
                          «Все программы» — любой звук Mac: музыка и видео тоже будут переводиться и приглушаться, \
                          поэтому лучше выбрать программу звонка.";
const HIS_VOICE_TIP: &str = "Оригинальный голос собеседника в ваших наушниках. Под переводом — тише \
                             (автоприглушение); 0 — слышен только перевод.";
const DUCK_SPEED_TIP: &str = "Через сколько голос возвращается к обычной громкости, когда перевод замолчал.";
/// dBFS shown as an empty meter.
const METER_FLOOR: f32 = -60.0;
/// A new paragraph after this long a pause in the text.
const PARAGRAPH_GAP: Duration = Duration::from_secs(4);
const PANE_MAX_LINES: usize = 500;
/// The running apps are listed afresh at most this often.
const APPS_REFRESH: Duration = Duration::from_secs(3);
const LABEL_WIDTH: f32 = 150.0;

pub fn run(api_key: String) -> eframe::Result {
    let mut viewport = egui::ViewportBuilder::default()
        .with_title(TITLE)
        .with_inner_size([1080.0, 880.0])
        .with_min_inner_size([960.0, 760.0]);
    if let Ok(icon) = eframe::icon_data::from_png_bytes(include_bytes!("../assets/pressvoice.png")) {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native(TITLE, options, Box::new(move |cc| Ok(Box::new(LiveWindow::new(cc, api_key)))))
}

/// What the key handlers share with the window.
struct Link {
    translator: Mutex<Option<Arc<Translator>>>,
    /// The key is held down.
    held: AtomicBool,
    ctx: egui::Context,
}

impl Link {
    fn current(&self) -> Option<Arc<Translator>> {
        self.translator.lock().clone()
    }
}

enum Connection {
    Off,
    Connecting,
    Ready,
    Failed(String),
}

/// The live text of one side: a new paragraph after a pause in the text,
/// not at the key press — the previous phrase is often still being
/// finished then.
#[derive(Default)]
struct PaneText {
    text: String,
    last: Option<Instant>,
}

impl PaneText {
    fn append(&mut self, piece: &str) {
        let now = Instant::now();
        let paused = self.last.is_some_and(|t| now - t > PARAGRAPH_GAP);
        self.last = Some(now);
        if self.text.trim().is_empty() {
            self.text = piece.trim_start().to_owned();
        } else if paused {
            self.text.push_str("\n\n");
            self.text.push_str(piece.trim_start());
        } else {
            self.text.push_str(piece);
        }
        let lines = self.text.matches('\n').count();
        if lines > PANE_MAX_LINES {
            let cut = self.text.match_indices('\n').nth(lines - PANE_MAX_LINES).map_or(0, |(i, _)| i + 1);
            self.text.drain(..cut);
        }
    }
}

/// A channel being (re)built in the background: which restart it belongs to.
type Ready<T> = (u64, Result<Arc<T>, String>);

struct LiveWindow {
    api_key: String,
    /// The Gemini balance and spending (shared by both channels).
    meter: Arc<Meter>,
    /// The «Указать баланс» dialog's text while it is open.
    topup: Option<String>,
    settings: WindowSettings,
    inputs: Vec<String>,
    outputs: Vec<String>,
    headphones: Vec<String>,
    apps: Vec<App>,
    apps_listed: Instant,
    link: Arc<Link>,
    hotkey: Hotkey,
    // My voice → the other side.
    generation: u64,
    connection: Connection,
    ready_tx: Sender<Ready<Translator>>,
    ready_rx: Receiver<Ready<Translator>>,
    // The other side → me.
    incoming: Option<Arc<Incoming>>,
    in_generation: u64,
    in_connection: Connection,
    in_ready_tx: Sender<Ready<Incoming>>,
    in_ready_rx: Receiver<Ready<Incoming>>,
    // Live text.
    text_tx: Sender<(Pane, String)>,
    text_rx: Receiver<(Pane, String)>,
    said: PaneText,
    heard: PaneText,
    they_said: PaneText,
    i_hear: PaneText,
}

impl LiveWindow {
    fn new(cc: &eframe::CreationContext<'_>, api_key: String) -> Self {
        crate::fonts::install(&cc.egui_ctx);
        cc.egui_ctx.all_styles_mut(|style| {
            for font in style.text_styles.values_mut() {
                font.size *= 1.05;
            }
        });

        // The virtual microphone is never a source (that would translate the
        // translation); as an output it sends my translation into the call.
        let inputs: Vec<String> = audio::device_names(true).into_iter().filter(|n| n != virtual_mic::NAME).collect();
        let headphones = audio::device_names(false);
        let mut outputs = headphones.clone();
        if virtual_mic::feed_device().is_some() {
            outputs.insert(0, virtual_mic::NAME.to_owned());
        }
        let mut settings = WindowSettings::load();
        let defaults = WindowSettings::default();
        // A saved choice that is gone (an unplugged device) falls back to the default.
        if settings.mic.as_ref().is_some_and(|m| !inputs.contains(m)) {
            settings.mic = None;
        }
        if settings.out.as_ref().is_some_and(|o| !outputs.contains(o)) {
            settings.out = None;
        }
        if settings.headphones.as_ref().is_some_and(|h| !headphones.contains(h)) {
            settings.headphones = None;
        }
        if languages::name(&settings.direction).is_none() {
            settings.direction = defaults.direction.clone();
        }
        if languages::name(&settings.my_lang).is_none() {
            settings.my_lang = defaults.my_lang.clone();
        }
        if !KEYS.iter().any(|(_, id)| *id == settings.key) {
            settings.key = defaults.key;
        }

        let link = Arc::new(Link { translator: Mutex::new(None), held: AtomicBool::new(false), ctx: cc.egui_ctx.clone() });
        let (down, up) = (link.clone(), link.clone());
        let hotkey = Hotkey::install(
            &settings.key,
            move || {
                down.held.store(true, Ordering::Release);
                if let Some(translator) = down.current() {
                    translator.press();
                }
                down.ctx.request_repaint();
            },
            move || {
                up.held.store(false, Ordering::Release);
                if let Some(translator) = up.current() {
                    translator.release();
                }
                up.ctx.request_repaint();
            },
        );
        journal::log(&format!("window: hotkey = {}", settings.key));

        let (ready_tx, ready_rx) = channel();
        let (in_ready_tx, in_ready_rx) = channel();
        let (text_tx, text_rx) = channel();
        let mut window = Self {
            api_key,
            meter: Meter::load(),
            topup: None,
            settings,
            inputs,
            outputs,
            headphones,
            apps: app_audio::running_apps(),
            apps_listed: Instant::now(),
            link,
            hotkey,
            generation: 0,
            connection: Connection::Connecting,
            ready_tx,
            ready_rx,
            incoming: None,
            in_generation: 0,
            in_connection: Connection::Off,
            in_ready_tx,
            in_ready_rx,
            text_tx,
            text_rx,
            said: PaneText::default(),
            heard: PaneText::default(),
            they_said: PaneText::default(),
            i_hear: PaneText::default(),
        };
        window.restart();
        window.restart_incoming();
        window
    }

    fn text_sink(&self) -> TextSink {
        let (texts, ctx) = (self.text_tx.clone(), self.link.ctx.clone());
        Arc::new(move |pane: Pane, piece: &str| {
            let _ = texts.send((pane, piece.to_owned()));
            ctx.request_repaint();
        })
    }

    /// My channel's settings changed: build a fresh translator in the background.
    fn restart(&mut self) {
        self.generation += 1;
        let generation = self.generation;
        self.connection = Connection::Connecting;
        let old = self.link.translator.lock().take();
        let s = &self.settings;
        let options = Options {
            target: s.direction.clone(),
            mic: s.mic.clone(),
            out: s.out.clone(),
            deeper: s.deeper,
            echo: s.echo,
            mix: mix_settings(s.voice_level, s.auto_duck, s.duck_to, &s.duck_speed),
            meter: self.meter.clone(),
        };
        let (key, text, ready, ctx) = (self.api_key.clone(), self.text_sink(), self.ready_tx.clone(), self.link.ctx.clone());
        thread::spawn(move || {
            if let Some(old) = old {
                old.close(); // frees the microphone before it is opened again
            }
            let result = Translator::start(&key, &options, text).map_err(|e| format!("{e:#}"));
            let _ = ready.send((generation, result));
            ctx.request_repaint();
        });
    }

    /// The other side's channel: (re)built when on, closed when off.
    fn restart_incoming(&mut self) {
        self.in_generation += 1;
        let generation = self.in_generation;
        let old = self.incoming.take();
        if !self.settings.his_on {
            self.in_connection = Connection::Off;
            if let Some(old) = old {
                thread::spawn(move || old.close());
                journal::log("=== [собеседник] перевод выключен");
            }
            return;
        }
        self.in_connection = Connection::Connecting;
        let s = &self.settings;
        let options = IncomingOptions {
            target: s.my_lang.clone(),
            source: s.his_source.clone().map_or(Source::AllApps, Source::App),
            headphones: s.headphones.clone(),
            mix: mix_settings(s.his_level, s.his_auto_duck, s.his_duck_to, &s.his_duck_speed),
            meter: self.meter.clone(),
        };
        let (key, text, ready, ctx) = (self.api_key.clone(), self.text_sink(), self.in_ready_tx.clone(), self.link.ctx.clone());
        thread::spawn(move || {
            if let Some(old) = old {
                old.close(); // gives the app its sound back first
            }
            let result = Incoming::start(&key, &options, text).map_err(|e| format!("{e:#}"));
            let _ = ready.send((generation, result));
            ctx.request_repaint();
        });
    }

    /// Results of background work: channels connected, live text.
    fn poll(&mut self) {
        while let Ok((generation, result)) = self.ready_rx.try_recv() {
            if generation != self.generation {
                // Settings changed again while this one was connecting: a newer
                // one is on its way, this one must not keep running unseen.
                if let Ok(stale) = result {
                    thread::spawn(move || stale.close());
                }
                continue;
            }
            match result {
                Ok(translator) => {
                    *self.link.translator.lock() = Some(translator);
                    self.connection = Connection::Ready;
                }
                Err(e) => {
                    journal::log(&format!("error: не удалось подключиться: {e}"));
                    self.connection = Connection::Failed(e);
                }
            }
        }
        while let Ok((generation, result)) = self.in_ready_rx.try_recv() {
            if generation != self.in_generation {
                if let Ok(stale) = result {
                    thread::spawn(move || stale.close());
                }
                continue;
            }
            match result {
                Ok(incoming) => {
                    self.incoming = Some(incoming);
                    self.in_connection = Connection::Ready;
                }
                Err(e) => {
                    journal::log(&format!("error: [собеседник] не удалось подключиться: {e}"));
                    self.in_connection = Connection::Failed(e);
                }
            }
        }
        while let Ok((pane, piece)) = self.text_rx.try_recv() {
            match pane {
                Pane::Said => self.said.append(&piece),
                Pane::Heard => self.heard.append(&piece),
                Pane::TheySaid => self.they_said.append(&piece),
                Pane::IHear => self.i_hear.append(&piece),
            }
        }
        if self.apps_listed.elapsed() > APPS_REFRESH {
            self.apps = app_audio::running_apps();
            self.apps_listed = Instant::now();
        }
    }

    fn status(&self) -> String {
        match &self.connection {
            Connection::Off | Connection::Connecting => "Подключение к переводчику…".into(),
            Connection::Failed(e) => format!("Не удалось подключиться: {e}"),
            Connection::Ready if self.link.held.load(Ordering::Acquire) => {
                "● Идёт перевод — отпустите клавишу, когда закончите".into()
            }
            Connection::Ready => {
                let label = KEYS.iter().find(|(_, id)| *id == self.settings.key).map_or("клавишу", |(l, _)| l);
                let short = label.split(" (").next().unwrap_or(label);
                format!("Готово · удерживайте {short}, пока говорите")
            }
        }
    }

    fn source_name(&self) -> String {
        match &self.settings.his_source {
            None => ALL_APPS.to_lowercase(),
            Some(bundle) => self.apps.iter().find(|a| &a.bundle == bundle).map_or_else(|| bundle.clone(), |a| a.name.clone()),
        }
    }

    fn in_status(&self) -> String {
        match (&self.in_connection, &self.incoming) {
            (Connection::Off, _) => "Перевод собеседника выключен".into(),
            (Connection::Connecting, _) => "Подключение…".into(),
            (Connection::Failed(e), _) => format!("Не удалось: {e}"),
            (Connection::Ready, Some(incoming)) => match incoming.state() {
                State::Waiting => format!("Жду звук из «{}» — начните звонок", self.source_name()),
                State::Listening => format!("Слушаю «{}»", self.source_name()),
                State::Speaking => "● Собеседник говорит — перевод идёт".into(),
            },
            (Connection::Ready, None) => String::new(),
        }
    }

    /// The Gemini balance: what is left, how long it lasts, what this
    /// conversation, today and the month cost.
    fn balance_row(&mut self, ui: &mut egui::Ui) {
        let snap = self.meter.snapshot();
        let (yellow, red) = (egui::Color32::from_rgb(200, 140, 0), egui::Color32::from_rgb(210, 50, 50));
        // With the other side translated too, both translations are counted as
        // running at once: the time left halves (on the safe side).
        let both = self.settings.his_on;
        let channels = if both { 2.0 } else { 1.0 };
        ui.horizontal_wrapped(|ui| {
            match (snap.balance, snap.seconds_left) {
                (Some(balance), Some(one)) => {
                    let seconds = one / channels;
                    let color = if snap.exhausted || seconds < 120.0 {
                        Some(red)
                    } else if seconds < 600.0 {
                        Some(yellow)
                    } else {
                        None
                    };
                    let what = if both { "мой перевод + собеседника" } else { "только мой перевод" };
                    let mut money = RichText::new(format!("Баланс: {}", usage::money_text(balance))).strong().size(15.0);
                    let mut left =
                        RichText::new(format!("хватит ≈ на {} ({what})", usage::duration_text(seconds))).size(15.0);
                    if let Some(color) = color {
                        money = money.color(color);
                        left = left.color(color);
                    }
                    ui.label(money);
                    ui.label("·");
                    ui.label(left).on_hover_text(format!(
                        "Один синхронный перевод ≈ {} в минуту ({} в час). С переводом собеседника \
                         считается, что работают оба перевода сразу — время делится на два: только мой \
                         перевод — {}, с собеседником — {}. Когда говорят по очереди, хватит дольше: \
                         платятся только секунды, когда звук реально идёт в Gemini, тишина бесплатна.\n\n\
                         Из потраченного с последнего пополнения {} — служебное: «разогрев» новых \
                         сессий (Google считает и его).",
                        usage::money_text(usage::COST_PER_SECOND * 60.0),
                        usage::money_text(usage::COST_PER_SECOND * 3600.0),
                        usage::duration_text(one),
                        usage::duration_text(one / 2.0),
                        usage::money_text(snap.service)
                    ));
                }
                _ => {
                    ui.label(RichText::new("Баланс не указан").size(15.0).weak()).on_hover_text(
                        "Google не даёт узнать остаток по ключу: впишите сумму из AI Studio (Billing) — дальше программа считает сама.",
                    );
                }
            }
            let label = if snap.balance.is_some() { "Пополнить / сверить…" } else { "Указать баланс…" };
            if ui.button(label).clicked() {
                self.topup = Some(snap.balance.map_or(String::new(), |b| format!("{b:.2}")));
            }
        });
        let (length, usd, running) = snap.conversation;
        ui.horizontal_wrapped(|ui| {
            let mut time = RichText::new(format!(
                "{}Время разговора: {} · {}",
                if running { "● " } else { "" },
                usage::clock_text(length),
                usage::money_text(usd)
            ));
            if !running {
                time = time.weak();
            }
            ui.label(time).on_hover_text(
                "Идёт только пока вы держите клавишу и пока переводится голос собеседника. \
                 С каждым запуском окна — с нуля.",
            );
            if ui.small_button("Сбросить").on_hover_text("Обнулить время и стоимость разговора (баланс не меняется)").clicked() {
                self.meter.reset_conversation();
                journal::log("window: время разговора сброшено");
            }
        });
        if snap.exhausted {
            ui.label(RichText::new("Баланс Gemini закончился — пополните его в AI Studio и нажмите «Пополнить / сверить…»").color(red));
        }
    }

    /// «Сколько сейчас на счёте Gemini?»
    fn topup_dialog(&mut self, ctx: &egui::Context) {
        let Some(text) = self.topup.as_mut() else {
            return;
        };
        let (mut save, mut cancel) = (false, false);
        egui::Window::new("Баланс Gemini")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label("Сколько сейчас на счёте Gemini, в долларах?");
                ui.label(RichText::new("Сумму видно в AI Studio → Billing (или Dashboard → Usage and Limits).").weak());
                ui.horizontal(|ui| {
                    ui.label("$");
                    let field = ui.add(egui::TextEdit::singleline(text).desired_width(120.0).hint_text("5.00"));
                    field.request_focus();
                    if field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        save = true;
                    }
                });
                ui.horizontal(|ui| {
                    save |= ui.button("Сохранить").clicked();
                    cancel |= ui.button("Отмена").clicked();
                });
            });
        if save {
            if let Ok(usd) = text.trim().replace(',', ".").trim_start_matches('$').parse::<f64>() {
                self.meter.set_balance(usd);
                journal::log(&format!("window: баланс Gemini = ${usd:.2}"));
                self.topup = None;
            }
        } else if cancel {
            self.topup = None;
        }
    }

    /// My language and the other side's, side by side.
    fn languages_row(&mut self, ui: &mut egui::Ui) {
        let idle = !matches!(self.connection, Connection::Connecting);
        let in_idle = !matches!(self.in_connection, Connection::Connecting);
        let (mut mine_changed, mut his_changed) = (false, false);
        ui.horizontal(|ui| {
            ui.label("Мой язык:")
                .on_hover_text("На этот язык переводится собеседник — его перевод вы слышите в наушниках.");
            ui.add_enabled_ui(in_idle, |ui| mine_changed = language_combo(ui, "my_lang", &mut self.settings.my_lang));
            ui.add_space(24.0);
            ui.label("Язык собеседника:")
                .on_hover_text("На этот язык переводится ваша речь — её перевод слышит собеседник.");
            ui.add_enabled_ui(idle, |ui| his_changed = language_combo(ui, "direction", &mut self.settings.direction));
            ui.add_space(24.0);
            ui.label(RichText::new(format!("Модель: {MODEL_LABEL}")).weak());
        });
        if mine_changed || his_changed {
            self.settings.save();
        }
        if his_changed {
            self.restart();
        }
        if mine_changed && self.settings.his_on {
            self.restart_incoming();
        }
    }

    /// «Я → собеседник»: my voice and where its translation goes.
    fn my_section(&mut self, ui: &mut egui::Ui) {
        let idle = !matches!(self.connection, Connection::Connecting);
        let field = field_width(ui);
        let (mut reconnect, mut changed, mut mix_changed) = (false, false, false);
        ui.label(RichText::new("Я → собеседник").strong().size(16.0));
        ui.add_space(4.0);
        egui::Grid::new("mine").num_columns(2).spacing([10.0, 7.0]).min_col_width(LABEL_WIDTH).show(ui, |ui| {
            ui.label("Микрофон:");
            ui.add_enabled_ui(idle, |ui| reconnect |= device_combo(ui, "mic", field, &self.inputs, &mut self.settings.mic));
            ui.end_row();

            ui.label("Вывод перевода:");
            ui.add_enabled_ui(idle, |ui| reconnect |= device_combo(ui, "out", field, &self.outputs, &mut self.settings.out));
            ui.end_row();

            ui.label("Клавиша:");
            let mut key = KEYS.iter().map(|(_, id)| *id).find(|id| *id == self.settings.key).unwrap_or(KEYS[0].1);
            if choice_combo(ui, "key", field, &KEYS, &mut key) {
                self.settings.key = key.to_owned();
                self.hotkey.set_key(key);
                journal::log(&format!("window: hotkey = {}", self.settings.key));
                changed = true;
            }
            ui.end_row();

            ui.label("Коррекция тембра:").on_hover_text(DEEPER_TIP);
            if choice_combo(ui, "deeper", field, &DEEPER, &mut self.settings.deeper) {
                if let Some(translator) = self.link.current() {
                    translator.set_deeper(self.settings.deeper);
                }
                journal::log(&format!("window: голос ниже на {} полутона", self.settings.deeper));
                changed = true;
            }
            ui.end_row();

            ui.label("");
            ui.add_enabled_ui(idle, |ui| {
                reconnect |= ui.checkbox(&mut self.settings.echo, "Подавление эха (наушники в микрофоне)").changed();
            });
            ui.end_row();

            let into_call = self.settings.out.as_deref() == Some(virtual_mic::NAME);
            let s = &mut self.settings;
            mix_changed = voice_rows(ui, "my", field, into_call, INTO_CALL_TIP, "Мой голос в звонке:",
                                     "Автоприглушение: мой голос тише под переводом",
                                     &mut s.voice_level, &mut s.auto_duck, &mut s.duck_to, &mut s.duck_speed);

            let translator = self.link.current();
            ui.label("Уровни:");
            meters(ui, field, translator.as_ref().map(|t| (t.mic_level_db(), t.out_level_db())));
            ui.end_row();
        });
        ui.add_space(6.0);
        ui.label(RichText::new(self.status()).size(15.0));

        if mix_changed {
            let s = &self.settings;
            let mix = mix_settings(s.voice_level, s.auto_duck, s.duck_to, &s.duck_speed);
            if let Some(translator) = self.link.current() {
                translator.set_voice_mix(mix);
            }
            journal::log(&format!("window:{}", crate::translator::mix_summary(&mix).trim_start_matches(',')));
        }
        if reconnect || changed || mix_changed {
            self.settings.save();
        }
        if reconnect {
            self.restart();
        }
    }

    /// «Собеседник → я»: the call app's sound, translated into my language.
    fn his_section(&mut self, ui: &mut egui::Ui) {
        let idle = !matches!(self.in_connection, Connection::Connecting);
        let field = field_width(ui);
        let (mut reconnect, mut mix_changed) = (false, false);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Собеседник → я").strong().size(16.0));
            ui.add_space(12.0);
            ui.add_enabled_ui(idle, |ui| {
                reconnect |= ui
                    .checkbox(&mut self.settings.his_on, "Переводить собеседника")
                    .on_hover_text("Выключено — модель работает только на ваш голос (и не тратит деньги на собеседника).")
                    .changed();
            });
        });
        ui.add_space(4.0);
        let on = self.settings.his_on;
        egui::Grid::new("his").num_columns(2).spacing([10.0, 7.0]).min_col_width(LABEL_WIDTH).show(ui, |ui| {
            ui.label("Звук собеседника из:").on_hover_text(SOURCE_TIP);
            ui.add_enabled_ui(idle, |ui| reconnect |= app_combo(ui, field, &self.apps, &mut self.settings.his_source));
            ui.end_row();

            ui.label("Мои наушники:");
            ui.add_enabled_ui(idle, |ui| {
                reconnect |= device_combo(ui, "headphones", field, &self.headphones, &mut self.settings.headphones);
            });
            ui.end_row();

            let s = &mut self.settings;
            mix_changed = voice_rows(ui, "his", field, true, HIS_VOICE_TIP, "Его голос (оригинал):",
                                     "Автоприглушение: оригинал тише под переводом",
                                     &mut s.his_level, &mut s.his_auto_duck, &mut s.his_duck_to, &mut s.his_duck_speed);

            ui.label("Уровни:");
            meters(ui, field, self.incoming.as_ref().map(|i| (i.level_db(), i.out_level_db())));
            ui.end_row();
        });
        ui.add_space(6.0);
        ui.add_enabled_ui(on, |ui| ui.label(RichText::new(self.in_status()).size(15.0)));

        if mix_changed {
            let s = &self.settings;
            let mix = mix_settings(s.his_level, s.his_auto_duck, s.his_duck_to, &s.his_duck_speed);
            if let Some(incoming) = &self.incoming {
                incoming.set_voice_mix(mix);
            }
            journal::log(&format!("window: [собеседник]{}", crate::translator::mix_summary(&mix).replace("мой голос в звонке", "оригинал")));
        }
        if reconnect || mix_changed {
            self.settings.save();
        }
        if reconnect {
            self.restart_incoming();
        }
    }

    /// Two rows of two panes — my side, then the other side's — with my
    /// language on the left (what I say, what I hear) and theirs on the right.
    fn panes(&self, ui: &mut egui::Ui) {
        let row_height = ((ui.available_height() - 2.0 * 30.0) / 2.0).max(80.0);
        for (heading, panes) in [
            ("Я → собеседник", [("Я говорю", &self.said), ("Собеседник слышит (перевод)", &self.heard)]),
            // Left: my language; right: the other side's.
            ("Собеседник → я", [("Я слышу (перевод)", &self.i_hear), ("Собеседник говорит", &self.they_said)]),
        ] {
            ui.label(RichText::new(heading).strong());
            ui.columns(2, |columns| {
                for (column, (title, pane)) in columns.iter_mut().zip(panes) {
                    column.label(RichText::new(title).weak());
                    egui::Frame::new()
                        .fill(column.visuals().extreme_bg_color)
                        .stroke(column.visuals().widgets.noninteractive.bg_stroke)
                        .corner_radius(4.0)
                        .inner_margin(8.0)
                        .show(column, |ui| {
                            egui::ScrollArea::vertical()
                                .id_salt(title)
                                .max_height(row_height - 40.0)
                                .auto_shrink([false, false])
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                    ui.add(egui::Label::new(RichText::new(&pane.text).size(16.0)).wrap());
                                });
                        });
                }
            });
            ui.add_space(4.0);
        }
    }

    fn busy(&self) -> bool {
        let sounding = |levels: (f32, f32)| levels.0 > METER_FLOOR || levels.1 > METER_FLOOR;
        self.link.current().is_some_and(|t| sounding((t.mic_level_db(), t.out_level_db())))
            || self.incoming.as_ref().is_some_and(|i| sounding((i.level_db(), i.out_level_db())))
            || self.link.held.load(Ordering::Acquire)
            || matches!(self.connection, Connection::Connecting)
            || matches!(self.in_connection, Connection::Connecting)
    }
}

impl eframe::App for LiveWindow {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll();
        egui::CentralPanel::default().show(ui, |ui| {
            self.balance_row(ui);
            ui.add_space(4.0);
            self.languages_row(ui);
            ui.add_space(8.0);
            ui.columns(2, |columns| {
                let [left, right] = columns else { unreachable!("two columns") };
                section(left, |ui| self.my_section(ui));
                section(right, |ui| self.his_section(ui));
            });
            ui.add_space(10.0);
            self.panes(ui);
        });
        // The meters move at 20 frames a second while there is sound;
        // otherwise a few redraws a second are enough (and spare the CPU).
        let ctx = ui.ctx().clone();
        self.topup_dialog(&ctx);
        let every = if self.busy() { 50 } else { 250 };
        ui.ctx().request_repaint_after(Duration::from_millis(every));
    }

    fn on_exit(&mut self) {
        let incoming = self.incoming.take();
        let translator = self.link.translator.lock().take();
        let closing = thread::spawn(move || {
            if let Some(incoming) = incoming {
                incoming.close();
            }
        });
        if let Some(translator) = translator {
            translator.close();
        }
        let _ = closing.join();
        self.meter.save(); // after the sessions' last reports
    }
}

/// A boxed category of settings.
fn section(ui: &mut egui::Ui, contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::group(ui.style()).inner_margin(10.0).corner_radius(6.0).show(ui, |ui| {
        ui.set_width(ui.available_width());
        contents(ui);
    });
}

fn field_width(ui: &egui::Ui) -> f32 {
    (ui.available_width() - LABEL_WIDTH - 16.0).max(180.0)
}

/// A voice under the translation: its volume, auto-ducking, how far and how
/// fast. True if something changed.
#[allow(clippy::too_many_arguments)]
fn voice_rows(
    ui: &mut egui::Ui,
    id: &str,
    field: f32,
    enabled: bool,
    tip: &str,
    level_label: &str,
    duck_label: &str,
    level: &mut i32,
    auto_duck: &mut bool,
    duck_to: &mut i32,
    duck_speed: &mut String,
) -> bool {
    let mut changed = false;
    ui.label(level_label).on_hover_text(tip);
    ui.add_enabled_ui(enabled, |ui| changed |= choice_combo(ui, &format!("{id}_level"), field, &VOICE_LEVELS, level))
        .response
        .on_disabled_hover_text(tip);
    ui.end_row();

    let voice_on = enabled && *level > 0;
    ui.label("");
    ui.add_enabled_ui(voice_on, |ui| changed |= ui.checkbox(auto_duck, duck_label).changed());
    ui.end_row();

    let ducking = voice_on && *auto_duck;
    ui.label("Приглушать до:").on_hover_text(DUCK_SPEED_TIP);
    ui.add_enabled_ui(ducking, |ui| {
        ui.horizontal(|ui| {
            changed |= choice_combo(ui, &format!("{id}_duck_to"), 80.0, &DUCK_LEVELS, duck_to);
            ui.label("возврат:");
            let speeds = DUCK_SPEEDS.map(|(label, speed_id, _)| (label, speed_id));
            let mut speed = speeds.iter().map(|(_, s)| *s).find(|s| *s == duck_speed.as_str()).unwrap_or("medium");
            if choice_combo(ui, &format!("{id}_duck_speed"), (field - 160.0).max(120.0), &speeds, &mut speed) {
                *duck_speed = speed.to_owned();
                changed = true;
            }
        });
    });
    ui.end_row();
    changed
}

fn mix_settings(level: i32, auto_duck: bool, duck_to: i32, speed: &str) -> MixSettings {
    MixSettings {
        level: level as f32 / 100.0,
        auto_duck,
        duck_to: duck_to as f32 / 100.0,
        hold: DUCK_SPEEDS.iter().find(|(_, id, _)| *id == speed).map_or(DUCK_SPEEDS[1].2, |(_, _, hold)| *hold),
    }
}

fn language_combo(ui: &mut egui::Ui, id: &str, code: &mut String) -> bool {
    let (popular, rest) = languages::ordered();
    let before = code.clone();
    egui::ComboBox::from_id_salt(id)
        .width(200.0)
        .height(520.0)
        .selected_text(languages::name(code).unwrap_or(code.as_str()))
        .show_ui(ui, |ui| {
            for (c, name) in popular {
                ui.selectable_value(code, c.to_owned(), name);
            }
            ui.separator();
            for (c, name) in rest {
                ui.selectable_value(code, c.to_owned(), name);
            }
        });
    *code != before
}

fn device_label(name: &str) -> String {
    if name == virtual_mic::NAME { format!("{name} — в звонок") } else { name.to_owned() }
}

fn device_combo(ui: &mut egui::Ui, id: &str, width: f32, names: &[String], chosen: &mut Option<String>) -> bool {
    let before = chosen.clone();
    egui::ComboBox::from_id_salt(id)
        .width(width)
        .selected_text(chosen.as_deref().map_or(SYSTEM_DEVICE.to_owned(), device_label))
        .show_ui(ui, |ui| {
            ui.selectable_value(chosen, None, SYSTEM_DEVICE);
            for name in names {
                ui.selectable_value(chosen, Some(name.clone()), device_label(name));
            }
        });
    *chosen != before
}

/// «Все программы» or one running app (a saved app that isn't running stays listed).
fn app_combo(ui: &mut egui::Ui, width: f32, apps: &[App], chosen: &mut Option<String>) -> bool {
    let before = chosen.clone();
    let name_of = |bundle: &str| apps.iter().find(|a| a.bundle == bundle).map_or_else(|| bundle.to_owned(), |a| a.name.clone());
    egui::ComboBox::from_id_salt("his_source")
        .width(width)
        .height(420.0)
        .selected_text(chosen.as_deref().map_or(ALL_APPS.to_owned(), name_of))
        .show_ui(ui, |ui| {
            ui.selectable_value(chosen, None, ALL_APPS);
            ui.separator();
            if let Some(saved) = before.as_ref().filter(|b| !apps.iter().any(|a| &a.bundle == *b)) {
                ui.selectable_value(chosen, Some(saved.clone()), format!("{} (не запущена)", name_of(saved)));
            }
            for app in apps {
                ui.selectable_value(chosen, Some(app.bundle.clone()), &app.name);
            }
        });
    *chosen != before
}

fn choice_combo<T: Clone + PartialEq>(ui: &mut egui::Ui, id: &str, width: f32, choices: &[(&str, T)], value: &mut T) -> bool {
    let before = value.clone();
    let shown = choices.iter().find(|(_, v)| v == value).map_or("", |(label, _)| label);
    egui::ComboBox::from_id_salt(id).width(width).selected_text(shown).show_ui(ui, |ui| {
        for (label, choice) in choices {
            ui.selectable_value(value, choice.clone(), *label);
        }
    });
    *value != before
}

/// In and out levels side by side.
fn meters(ui: &mut egui::Ui, width: f32, levels_db: Option<(f32, f32)>) {
    let fill = |db: f32| ((db - METER_FLOOR) / -METER_FLOOR).clamp(0.0, 1.0);
    let (input, output) = levels_db.map_or((0.0, 0.0), |(i, o)| (fill(i), fill(o)));
    let bar = ((width - 90.0) / 2.0).max(60.0);
    ui.horizontal(|ui| {
        ui.label(RichText::new("вход").weak());
        ui.add(egui::ProgressBar::new(input).desired_width(bar).desired_height(12.0).corner_radius(2.0));
        ui.label(RichText::new("выход").weak());
        ui.add(egui::ProgressBar::new(output).desired_width(bar).desired_height(12.0).corner_radius(2.0));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_flows_then_breaks_after_a_pause() {
        let mut pane = PaneText::default();
        pane.append("  Привет,");
        pane.append(" как дела?");
        assert_eq!(pane.text, "Привет, как дела?");
        pane.last = Some(Instant::now() - Duration::from_secs(5));
        pane.append(" Новая фраза");
        assert_eq!(pane.text, "Привет, как дела?\n\nНовая фраза");
    }

    #[test]
    fn old_lines_are_dropped() {
        let mut pane = PaneText::default();
        for i in 0..(PANE_MAX_LINES + 50) {
            pane.append(&format!("строка {i}\n"));
        }
        assert!(pane.text.matches('\n').count() <= PANE_MAX_LINES);
        assert!(pane.text.ends_with(&format!("строка {}\n", PANE_MAX_LINES + 49)));
    }

    #[test]
    fn mix_settings_follow_the_window() {
        let mix = mix_settings(50, true, 30, "smooth");
        assert_eq!(mix.level, 0.5);
        assert_eq!(mix.duck_to, 0.3);
        assert_eq!(mix.hold, Duration::from_millis(1500));
        assert_eq!(mix_settings(0, false, 30, "unknown").hold, Duration::from_millis(700));
    }
}
