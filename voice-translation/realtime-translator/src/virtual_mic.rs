//! «PressVoice Microphone»: the virtual microphone from `driver/`, seen from the program.
//!
//! The driver makes two linked devices sharing one buffer: the microphone that
//! everyone sees (Teams/Zoom pick it), and a hidden output. Whatever is played
//! into the hidden output is what the microphone "hears". Hidden devices are not
//! listed, so the output is found by its UID.

use objc2_core_audio::{AudioDeviceID, kAudioHardwarePropertyTranslateUIDToDevice};

use crate::core_audio;

/// The name Teams/Zoom show; also how the window's settings refer to it.
pub const NAME: &str = "PressVoice Microphone";
/// The hidden output's UID (set by the driver build: kDriver_Name + "_2_UID").
const FEED_UID: &str = "PressVoice_2_UID";

/// The hidden output to play into, if the driver is installed.
pub fn feed_device() -> Option<AudioDeviceID> {
    core_audio::translate(kAudioHardwarePropertyTranslateUIDToDevice, FEED_UID)
}

#[cfg(test)]
mod tests {
    /// Needs the driver installed (`driver/install.sh`): `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn the_hidden_feed_is_found() {
        assert!(super::feed_device().is_some(), "PressVoice Microphone is not installed");
    }
}
