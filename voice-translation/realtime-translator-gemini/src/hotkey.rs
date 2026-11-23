//! The push-to-talk key, seen system-wide (macOS).
//!
//! A global NSEvent monitor sees the key while other apps are in front; a
//! local one is needed too, because a global monitor misses keys sent to
//! this app's own window. Modifier keys (right ⌥ and friends) come as
//! "flags changed" events; F18/F19 as key down/up. A release can be missed
//! (e.g. while a system dialog has the keyboard), so while a modifier is
//! held its real state is also polled.
//!
//! Needs the Accessibility / Input Monitoring permission of the app that
//! starts the program (Terminal, when launched with the .command file).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_app_kit::{NSEvent, NSEventMask, NSEventModifierFlags, NSEventType};
use parking_lot::Mutex;

/// (label, id) of the keys offered in the window.
pub const KEYS: [(&str, &str); 6] = [
    ("правый ⌥ Option", "alt_r"),
    ("правый ⌃ Ctrl", "ctrl_r"),
    ("правый ⇧ Shift", "shift_r"),
    ("правый ⌘ Command (если основной PressVoice выключен)", "cmd_r"),
    ("F18", "f18"),
    ("F19", "f19"),
];

/// The macOS key code and, for a modifier, its flag.
fn key_info(id: &str) -> Option<(u16, Option<NSEventModifierFlags>)> {
    Some(match id {
        "alt_r" => (61, Some(NSEventModifierFlags::Option)),
        "ctrl_r" => (62, Some(NSEventModifierFlags::Control)),
        "shift_r" => (60, Some(NSEventModifierFlags::Shift)),
        "cmd_r" => (54, Some(NSEventModifierFlags::Command)),
        "f18" => (79, None),
        "f19" => (80, None),
        _ => return None,
    })
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    /// Modifier flags held right now (same bit values as NSEvent's).
    fn CGEventSourceFlagsState(state_id: i32) -> u64;
}
const COMBINED_SESSION_STATE: i32 = 0;
const POLL_EVERY: Duration = Duration::from_millis(150);

type Callback = Box<dyn Fn() + Send + Sync>;

struct State {
    key: Mutex<String>,
    down: AtomicBool,
    polling: AtomicBool,
    on_down: Callback,
    on_up: Callback,
}

impl State {
    fn handle(self: &Arc<Self>, event: &NSEvent) {
        let key = self.key.lock().clone();
        let Some((code, flag)) = key_info(&key) else {
            return;
        };
        if event.keyCode() != code {
            return;
        }
        let pressed = match flag {
            Some(flag) => event.modifierFlags().contains(flag),
            None => match event.r#type() {
                NSEventType::KeyDown => true,
                NSEventType::KeyUp => false,
                _ => return,
            },
        };
        self.set_down(pressed, flag);
    }

    fn set_down(self: &Arc<Self>, pressed: bool, flag: Option<NSEventModifierFlags>) {
        if pressed {
            // Auto-repeat sends more key-downs while held: only the first counts.
            if !self.down.swap(true, Ordering::AcqRel) {
                (self.on_down)();
                if let Some(flag) = flag {
                    self.watch_release(flag);
                }
            }
        } else if self.down.swap(false, Ordering::AcqRel) {
            (self.on_up)();
        }
    }

    /// Polls the modifier while it is held, in case its release is missed.
    fn watch_release(self: &Arc<Self>, flag: NSEventModifierFlags) {
        if self.polling.swap(true, Ordering::AcqRel) {
            return;
        }
        let state = self.clone();
        thread::spawn(move || {
            while state.down.load(Ordering::Acquire) {
                thread::sleep(POLL_EVERY);
                let held = unsafe { CGEventSourceFlagsState(COMBINED_SESSION_STATE) } & flag.0 as u64 != 0;
                if !held {
                    state.set_down(false, None);
                }
            }
            state.polling.store(false, Ordering::Release);
        });
    }
}

/// Listens for the key until dropped.
pub struct Hotkey {
    state: Arc<State>,
    monitors: Vec<Retained<AnyObject>>,
}

impl Hotkey {
    /// Must run on the main thread once the app is running. The callbacks
    /// are called on the main thread or on the release poller.
    pub fn install(key: &str, on_down: impl Fn() + Send + Sync + 'static, on_up: impl Fn() + Send + Sync + 'static) -> Self {
        let state = Arc::new(State {
            key: Mutex::new(key.to_owned()),
            down: AtomicBool::new(false),
            polling: AtomicBool::new(false),
            on_down: Box::new(on_down),
            on_up: Box::new(on_up),
        });
        let mask = NSEventMask::FlagsChanged | NSEventMask::KeyDown | NSEventMask::KeyUp;
        let mut monitors = Vec::new();

        let global_state = state.clone();
        let global = RcBlock::new(move |event: NonNull<NSEvent>| {
            // AppKit calls this; a panic must not unwind into it.
            let _ = catch_unwind(AssertUnwindSafe(|| global_state.handle(unsafe { event.as_ref() })));
        });
        monitors.extend(NSEvent::addGlobalMonitorForEventsMatchingMask_handler(mask, &global));

        let local_state = state.clone();
        let local = RcBlock::new(move |event: NonNull<NSEvent>| -> *mut NSEvent {
            let _ = catch_unwind(AssertUnwindSafe(|| local_state.handle(unsafe { event.as_ref() })));
            event.as_ptr() // a local monitor must hand the event on
        });
        monitors.extend(unsafe { NSEvent::addLocalMonitorForEventsMatchingMask_handler(mask, &local) });

        Self { state, monitors }
    }

    /// Listen to another key from now on (a held one counts as released).
    pub fn set_key(&self, key: &str) {
        *self.state.key.lock() = key.to_owned();
        self.state.set_down(false, None);
    }
}

impl Drop for Hotkey {
    fn drop(&mut self) {
        for monitor in self.monitors.drain(..) {
            unsafe { NSEvent::removeMonitor(&monitor) };
        }
    }
}
