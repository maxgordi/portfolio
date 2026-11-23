//! The other side of the call: the sound of another app — Teams, Zoom,
//! Telegram, WhatsApp, a browser… — taken straight from it with a Core Audio
//! process tap (macOS 14.2+). No virtual speaker: the app keeps its usual
//! output, nothing in it has to be set up.
//!
//! The tap goes into a private aggregate device, which is then read like a
//! microphone. While tapped, the app's own sound is muted: the program plays
//! it back itself, at the volume the user chose, under the translation.
//! macOS asks once for the "System Audio Recording" permission.

use std::ffi::{CStr, c_void};
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Result, anyhow, bail};
use coreaudio::audio_unit::macos_helpers::audio_unit_from_device_id_uninitialized;
use coreaudio::audio_unit::render_callback::{self, data};
use coreaudio::audio_unit::{AudioUnit, Element, SampleFormat, Scope, StreamFormat};
use coreaudio::audio_unit::audio_format::LinearPcmFlags;
use objc2::AllocAnyThread;
use objc2::rc::Retained;
use objc2_app_kit::{NSApplicationActivationPolicy, NSWorkspace};
use objc2_core_audio::{
    AudioHardwareCreateAggregateDevice, AudioHardwareCreateProcessTap, AudioHardwareDestroyAggregateDevice,
    AudioHardwareDestroyProcessTap, AudioObjectID, CATapDescription, CATapMuteBehavior,
    kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceNameKey, kAudioAggregateDeviceTapAutoStartKey,
    kAudioAggregateDeviceTapListKey, kAudioAggregateDeviceUIDKey, kAudioDevicePropertyNominalSampleRate,
    kAudioHardwarePropertyProcessObjectList, kAudioHardwarePropertyTranslatePIDToProcessObject,
    kAudioProcessPropertyBundleID, kAudioSubTapDriftCompensationKey, kAudioSubTapUIDKey,
};
use objc2_core_foundation::{
    CFArray, CFBoolean, CFDictionary, CFMutableDictionary, CFRetained, CFString, kCFAllocatorDefault,
    kCFTypeArrayCallBacks, kCFTypeDictionaryKeyCallBacks, kCFTypeDictionaryValueCallBacks,
};
use objc2_foundation::{NSArray, NSNumber, NSString};

use crate::audio::Worker;
use crate::core_audio;
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::{HeapProd, HeapRb};

/// Whose sound is the other side of the call.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// Every app except this one (whatever the call is in).
    AllApps,
    /// One app (and its helper processes), by bundle ID.
    App(String),
}

/// An app the user can pick.
#[derive(Clone, Debug, PartialEq)]
pub struct App {
    pub bundle: String,
    pub name: String,
}

/// The apps in the Dock right now (the ones a call can be in), by name.
pub fn running_apps() -> Vec<App> {
    let mine = std::process::id() as i32;
    let mut apps: Vec<App> = NSWorkspace::sharedWorkspace()
        .runningApplications()
        .iter()
        .filter(|app| app.activationPolicy() == NSApplicationActivationPolicy::Regular && app.processIdentifier() != mine)
        .filter_map(|app| {
            let bundle = app.bundleIdentifier()?.to_string();
            let name = app.localizedName().map_or_else(|| bundle.clone(), |n| n.to_string());
            Some(App { bundle, name })
        })
        .collect();
    apps.sort_by_key(|a| a.name.to_lowercase());
    apps.dedup_by(|a, b| a.bundle == b.bundle);
    apps
}

/// The display name of an app by bundle ID, if it is running.
pub fn app_name(bundle: &str) -> Option<String> {
    running_apps().into_iter().find(|a| a.bundle == bundle).map(|a| a.name)
}

/// Core Audio's process objects of an app: the app itself and its helpers
/// (browsers and Teams play sound from helper processes), matched by bundle
/// ID prefix. Sorted, so two lists can be compared.
pub fn process_objects(bundle: &str) -> Vec<AudioObjectID> {
    let mut objects: Vec<AudioObjectID> = core_audio::get_list::<AudioObjectID>(core_audio::SYSTEM, kAudioHardwarePropertyProcessObjectList)
        .into_iter()
        .filter(|&object| {
            core_audio::get_string(object, kAudioProcessPropertyBundleID)
                .is_some_and(|id| id == bundle || id.starts_with(&format!("{bundle}.")))
        })
        .collect();
    objects.sort_unstable();
    objects
}

/// This program's own process object (left out of an "all apps" tap: its
/// translation and replayed voice must not be captured again).
fn own_process_object() -> Option<AudioObjectID> {
    let pid = std::process::id() as i32;
    core_audio::get::<AudioObjectID>(core_audio::SYSTEM, kAudioHardwarePropertyTranslatePIDToProcessObject, &pid.to_ne_bytes())
        .filter(|&object| object != 0)
}

/// Called on a worker thread with the captured sound (mono, 16-bit scale)
/// and its rate.
pub type AudioHandler = Box<dyn FnMut(&[f32], u32) + Send>;

static INSTANCES: AtomicU32 = AtomicU32::new(0);

/// A running capture; stops (and gives the app its sound back) when dropped.
pub struct AppAudio {
    pub rate: u32,
    unit: Option<AudioUnit>,
    aggregate: AudioObjectID,
    tap: AudioObjectID,
    worker: Option<Worker>,
}

impl AppAudio {
    pub fn open(source: &Source, on_audio: AudioHandler) -> Result<Self> {
        let description = tap_description(source)?;
        let mut tap: AudioObjectID = 0;
        let status = unsafe { AudioHardwareCreateProcessTap(Some(&description), &mut tap) };
        if status != 0 || tap == 0 {
            bail!("macOS не дал записать звук программы (код {status}): проверьте «Запись системного звука» в Конфиденциальности");
        }
        let mut capture = Self { rate: 0, unit: None, aggregate: 0, tap, worker: None };
        let tap_uid = unsafe { description.UUID().UUIDString() }.to_string();
        let instance = INSTANCES.fetch_add(1, Ordering::Relaxed);
        let properties = aggregate_properties(
            &tap_uid,
            &format!("com.pressvoice.call-audio.{}.{instance}", std::process::id()),
            "PressVoice — звук собеседника",
        );
        let status = unsafe { AudioHardwareCreateAggregateDevice(&properties, std::ptr::NonNull::from(&mut capture.aggregate)) };
        if status != 0 {
            bail!("не удалось подключиться к звуку программы (код {status})");
        }
        let rate = core_audio::get::<f64>(capture.aggregate, kAudioDevicePropertyNominalSampleRate, &[]).unwrap_or(48000.0);
        capture.rate = rate as u32;
        // The Core Audio callback only fills a ring; the handler runs on a worker.
        let (producer, mut consumer) = HeapRb::<f32>::new(rate as usize).split();
        let mut on_audio = on_audio;
        let worker = Worker::spawn("call audio", move |stop| {
            let mut block = vec![0.0f32; rate as usize / 50];
            while !stop.load(Ordering::Acquire) {
                std::thread::park_timeout(std::time::Duration::from_millis(50));
                loop {
                    let n = consumer.pop_slice(&mut block);
                    if n == 0 {
                        break;
                    }
                    on_audio(&block[..n], rate as u32);
                }
            }
        })?;
        capture.unit = Some(start_unit(capture.aggregate, rate, producer, worker.thread())?);
        capture.worker = Some(worker);
        Ok(capture)
    }
}

impl Drop for AppAudio {
    fn drop(&mut self) {
        drop(self.unit.take()); // stop reading before the device goes
        drop(self.worker.take());
        unsafe {
            if self.aggregate != 0 {
                AudioHardwareDestroyAggregateDevice(self.aggregate);
            }
            AudioHardwareDestroyProcessTap(self.tap);
        }
    }
}

fn tap_description(source: &Source) -> Result<Retained<CATapDescription>> {
    let numbers = |objects: &[AudioObjectID]| -> Retained<NSArray<NSNumber>> {
        let numbers: Vec<Retained<NSNumber>> = objects.iter().map(|&o| NSNumber::new_u32(o)).collect();
        NSArray::from_retained_slice(&numbers)
    };
    let description = match source {
        Source::AllApps => {
            let mine: Vec<AudioObjectID> = own_process_object().into_iter().collect();
            unsafe { CATapDescription::initMonoGlobalTapButExcludeProcesses(CATapDescription::alloc(), &numbers(&mine)) }
        }
        Source::App(bundle) => {
            let objects = process_objects(bundle);
            if objects.is_empty() {
                let name = app_name(bundle).unwrap_or_else(|| bundle.clone());
                return Err(anyhow!("«{name}» пока не воспроизводит звук — начните звонок, перевод подключится сам"));
            }
            unsafe { CATapDescription::initMonoMixdownOfProcesses(CATapDescription::alloc(), &numbers(&objects)) }
        }
    };
    unsafe {
        description.setName(&NSString::from_str("PressVoice — звук собеседника"));
        description.setPrivate(true);
        // The app goes quiet while tapped; the program replays it at the chosen volume.
        description.setMuteBehavior(CATapMuteBehavior::MutedWhenTapped);
    }
    Ok(description)
}

fn cf_key(key: &CStr) -> CFRetained<CFString> {
    CFString::from_str(key.to_str().expect("Core Audio keys are ASCII"))
}

fn set(dict: &CFMutableDictionary, key: &CStr, value: *const c_void) {
    let key = cf_key(key);
    unsafe { CFMutableDictionary::set_value(Some(dict), (&*key as *const CFString).cast(), value) };
}

/// { name, uid, private, tap auto-start, taps: [{ uid, drift compensation }] }
/// (after cpal's loopback code).
fn aggregate_properties(tap_uid: &str, uid: &str, name: &str) -> CFRetained<CFDictionary> {
    let new_dict = || unsafe {
        CFMutableDictionary::new(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks)
            .expect("a dictionary")
    };
    let yes = CFBoolean::new(true);
    let yes_ptr: *const c_void = (yes as *const CFBoolean).cast();

    let tap = new_dict();
    let tap_uid = CFString::from_str(tap_uid);
    set(&tap, kAudioSubTapUIDKey, (&*tap_uid as *const CFString).cast());
    set(&tap, kAudioSubTapDriftCompensationKey, yes_ptr);
    let taps_items = [(&*tap as *const CFMutableDictionary).cast::<c_void>()];
    let taps = unsafe {
        CFArray::new(kCFAllocatorDefault, taps_items.as_ptr() as *mut *const c_void, 1, &kCFTypeArrayCallBacks)
            .expect("an array")
    };

    let properties = new_dict();
    let (name, uid) = (CFString::from_str(name), CFString::from_str(uid));
    set(&properties, kAudioAggregateDeviceNameKey, (&*name as *const CFString).cast());
    set(&properties, kAudioAggregateDeviceUIDKey, (&*uid as *const CFString).cast());
    set(&properties, kAudioAggregateDeviceTapListKey, (&*taps as *const CFArray).cast());
    set(&properties, kAudioAggregateDeviceTapAutoStartKey, yes_ptr);
    set(&properties, kAudioAggregateDeviceIsPrivateKey, yes_ptr);
    unsafe { CFRetained::cast_unchecked::<CFDictionary>(properties) }
}

/// Reads the aggregate device like a microphone (mono, 32-bit float) into `ring`.
fn start_unit(device: AudioObjectID, rate: f64, mut ring: HeapProd<f32>, wake: std::thread::Thread) -> Result<AudioUnit> {
    fn err(what: &'static str) -> impl Fn(coreaudio::Error) -> anyhow::Error {
        move |e| anyhow!("звук собеседника: {what}: {e}")
    }
    let mut unit = audio_unit_from_device_id_uninitialized(device, true).map_err(err("устройство"))?;
    let format = StreamFormat {
        sample_rate: rate,
        sample_format: SampleFormat::F32,
        flags: LinearPcmFlags::IS_FLOAT | LinearPcmFlags::IS_PACKED | LinearPcmFlags::IS_NON_INTERLEAVED,
        channels: 1,
    };
    unit.set_stream_format(format, Scope::Output, Element::Input).map_err(err("формат"))?;
    unit.initialize().map_err(err("запуск"))?;
    type Args = render_callback::Args<data::NonInterleaved<f32>>;
    unit.set_input_callback(move |args: Args| {
        if let Some(channel) = args.data.channels().next() {
            ring.push_iter(channel.iter().map(|&x| x * 32768.0));
            wake.unpark();
        }
        Ok(())
    })
    .map_err(err("приём"))?;
    unit.start().map_err(err("старт"))?;
    Ok(unit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::sync::Arc;
    use std::time::Duration;

    /// Needs the "System Audio Recording" permission: `cargo test -- --ignored`.
    /// `afplay` plays a sound; the "all apps" tap must hear it.
    #[test]
    #[ignore]
    fn hears_another_app() {
        let loudest = Arc::new(parking_lot::Mutex::new(-120.0f32));
        let seen = loudest.clone();
        let capture = AppAudio::open(
            &Source::AllApps,
            Box::new(move |samples, _| {
                let db = crate::dsp::dbfs(samples);
                let mut l = seen.lock();
                *l = l.max(db);
            }),
        )
        .expect("tap");
        println!("tap at {} Hz", capture.rate);
        let status = Command::new("afplay").args(["-t", "1.5", "/System/Library/Sounds/Submarine.aiff"]).status().expect("afplay");
        assert!(status.success());
        std::thread::sleep(Duration::from_millis(200));
        drop(capture);
        let loudest = *loudest.lock();
        println!("another app's sound: loudest {loudest:.1} dBFS");
        assert!(loudest > -50.0, "heard nothing ({loudest:.1} dBFS)");
    }

    #[test]
    #[ignore]
    fn lists_running_apps() {
        let apps = running_apps();
        for app in &apps {
            println!("{} — {} — {} process objects", app.name, app.bundle, process_objects(&app.bundle).len());
        }
        assert!(!apps.is_empty());
    }
}
