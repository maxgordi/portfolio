//! Reading Core Audio object properties (devices, processes, taps) — the few
//! raw calls the device and app-audio code need, in one safe place.

use std::ptr::NonNull;

use objc2_core_audio::{
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectID, AudioObjectPropertyAddress,
    AudioObjectPropertySelector, kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal,
    kAudioObjectSystemObject,
};
use objc2_core_foundation::{CFRetained, CFString};

pub const SYSTEM: AudioObjectID = kAudioObjectSystemObject as AudioObjectID;

fn address(selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

/// A fixed-size property (`T` must be the property's exact C type).
/// `qualifier`: the bytes some properties need to answer (e.g. a PID).
pub fn get<T: Copy + Default>(object: AudioObjectID, selector: AudioObjectPropertySelector, qualifier: &[u8]) -> Option<T> {
    let address = address(selector);
    let mut value = T::default();
    let mut size = size_of::<T>() as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            qualifier.len() as u32,
            if qualifier.is_empty() { std::ptr::null() } else { qualifier.as_ptr().cast() },
            NonNull::from(&mut size),
            NonNull::from(&mut value).cast(),
        )
    };
    (status == 0).then_some(value)
}

/// An array property (e.g. the list of process objects).
pub fn get_list<T: Copy + Default>(object: AudioObjectID, selector: AudioObjectPropertySelector) -> Vec<T> {
    let address = address(selector);
    let mut size = 0u32;
    let status = unsafe { AudioObjectGetPropertyDataSize(object, NonNull::from(&address), 0, std::ptr::null(), NonNull::from(&mut size)) };
    if status != 0 || size == 0 {
        return Vec::new();
    }
    let mut items = vec![T::default(); size as usize / size_of::<T>()];
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            std::ptr::null(),
            NonNull::from(&mut size),
            NonNull::new(items.as_mut_ptr()).expect("non-empty").cast(),
        )
    };
    if status != 0 {
        return Vec::new();
    }
    items.truncate(size as usize / size_of::<T>());
    items
}

/// A string property (the caller of Core Audio owns the returned CFString).
pub fn get_string(object: AudioObjectID, selector: AudioObjectPropertySelector) -> Option<String> {
    let raw: *const CFString = get(object, selector, &[])?;
    let string = unsafe { CFRetained::from_raw(NonNull::new(raw.cast_mut())?) };
    Some(string.to_string())
}

/// The object for a string key (a device UID, …), via a "translate" property.
pub fn translate(selector: AudioObjectPropertySelector, key: &str) -> Option<AudioObjectID> {
    let key = CFString::from_str(key);
    let key_ref: *const CFString = &*key;
    // The qualifier is the CFStringRef itself.
    let qualifier = unsafe { std::slice::from_raw_parts((&key_ref as *const *const CFString).cast::<u8>(), size_of::<*const CFString>()) };
    get::<AudioObjectID>(SYSTEM, selector, qualifier).filter(|&id| id != 0)
}
