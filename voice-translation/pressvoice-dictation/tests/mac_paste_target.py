"""Helper for tests/test_e2e_macos.py: a separate app with a frontmost
window and a text field — the "other app" PressVoice pastes into. Every
0.1 s it writes {"active": ..., "text": ...} to the JSON file given as
argv[1]; it quits after argv[2] seconds."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AppKit import (  # noqa: E402
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSBackingStoreBuffered,
    NSMakeRect,
    NSTextField,
    NSTimer,
    NSWindow,
    NSWindowStyleMaskTitled,
)
from Foundation import NSObject  # noqa: E402

import mac_support  # noqa: E402

OUT, LIFETIME = sys.argv[1], float(sys.argv[2])


class Target(NSObject):
    def setUp(self):
        self.win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(300, 300, 480, 60), NSWindowStyleMaskTitled,
            NSBackingStoreBuffered, False)
        self.win.setTitle_("PressVoice paste target")
        self.field = NSTextField.alloc().initWithFrame_(NSMakeRect(12, 18, 456, 24))
        self.win.contentView().addSubview_(self.field)
        self.win.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.win.makeFirstResponder_(self.field)
        self.ticks = 0
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.1, self, "tick:", None, True)

    def tick_(self, timer):
        app = NSApplication.sharedApplication()
        if not app.isActive():
            app.activateIgnoringOtherApps_(True)
            self.win.makeKeyAndOrderFront_(None)
            self.win.makeFirstResponder_(self.field)
        state = {"active": bool(app.isActive() and self.win.isKeyWindow()),
                 "text": str(self.field.stringValue())}
        with open(OUT + ".tmp", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(OUT + ".tmp", OUT)
        self.ticks += 1
        if self.ticks * 0.1 >= LIFETIME:
            app.terminate_(None)


app = NSApplication.sharedApplication()
app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
mac_support.install_edit_menu()  # so the field accepts Cmd+V
target = Target.alloc().init()
target.setUp()
app.run()
