//! macOS Accessibility API text insertion.
//!
//! Streams transcription text directly into the focused text field using the AX API,
//! bypassing the clipboard entirely. Falls back gracefully when the focused element
//! doesn't support text attributes (e.g. terminal emulators).

use accessibility_sys::*;
use core_foundation::base::{CFRelease, CFTypeRef, TCFType};
use core_foundation::string::CFString;
use core_foundation_sys::base::{CFIndex, CFRange};
use log::{debug, error, info, warn};
use std::ptr;

/// Result of attempting to begin an AX text insertion session.
pub enum SessionResult {
    /// Session created successfully — AX streaming is available.
    Active(TextInsertionSession),
    /// The focused element doesn't support AX text attributes; caller should
    /// fall back to clipboard paste on release.
    FallbackNeeded,
}

/// Result of a single `update_text` call.
pub enum InsertResult {
    /// Text was inserted/replaced successfully.
    Ok,
    /// Transient failure (e.g. app busy) — safe to retry next cycle.
    Retry,
    /// Permanent failure — stop using AX for this session.
    Failed,
}

/// An active accessibility text insertion session.
///
/// Captures the focused AX element and cursor offset when recording begins,
/// then streams text by selecting-and-replacing the previously inserted range.
pub struct TextInsertionSession {
    focused_element: AXUIElementRef,
    insertion_offset: CFIndex,
    inserted_length: CFIndex, // in UTF-16 code units (what AX uses)
    verified: bool,           // true after readback confirms text was actually inserted
}

// AXUIElementRef is a CFType — safe to send across threads once captured.
unsafe impl Send for TextInsertionSession {}

impl TextInsertionSession {
    /// Attempt to start an AX text insertion session.
    ///
    /// Queries the system-wide AX element for the currently focused UI element,
    /// verifies it supports the text attributes we need, and records the cursor
    /// position. Returns `FallbackNeeded` if any step fails.
    pub fn begin() -> SessionResult {
        unsafe {
            // 1. Get the system-wide AX element
            let system = AXUIElementCreateSystemWide();
            if system.is_null() {
                warn!("AX: failed to create system-wide element");
                return SessionResult::FallbackNeeded;
            }

            // 2. Get the focused UI element
            let mut focused_raw: CFTypeRef = ptr::null_mut();
            let attr = CFString::new(kAXFocusedUIElementAttribute);
            let err = AXUIElementCopyAttributeValue(
                system,
                attr.as_concrete_TypeRef(),
                &mut focused_raw,
            );
            CFRelease(system as CFTypeRef);

            if err != kAXErrorSuccess || focused_raw.is_null() {
                debug!("AX: no focused element (error {})", err);
                return SessionResult::FallbackNeeded;
            }

            let focused = focused_raw as AXUIElementRef;

            // Set a messaging timeout so we don't block forever on hung apps
            AXUIElementSetMessagingTimeout(focused, 1.0);

            // 3. Check that the element supports the attributes we need
            let selected_range_attr = CFString::new(kAXSelectedTextRangeAttribute);
            let selected_text_attr = CFString::new(kAXSelectedTextAttribute);

            let mut range_settable: bool = false;
            let mut text_settable: bool = false;

            let err1 = AXUIElementIsAttributeSettable(
                focused,
                selected_range_attr.as_concrete_TypeRef(),
                &mut range_settable,
            );
            let err2 = AXUIElementIsAttributeSettable(
                focused,
                selected_text_attr.as_concrete_TypeRef(),
                &mut text_settable,
            );

            if err1 != kAXErrorSuccess || err2 != kAXErrorSuccess
                || !range_settable || !text_settable
            {
                debug!(
                    "AX: focused element doesn't support text attributes \
                     (range_err={}, text_err={}, range_settable={}, text_settable={})",
                    err1, err2, range_settable, text_settable
                );
                CFRelease(focused as CFTypeRef);
                return SessionResult::FallbackNeeded;
            }

            // 4. Record the current cursor position (start of selected text range)
            let insertion_offset = match Self::get_cursor_offset(focused) {
                Some(offset) => offset,
                None => {
                    debug!("AX: could not read cursor offset");
                    CFRelease(focused as CFTypeRef);
                    return SessionResult::FallbackNeeded;
                }
            };

            info!("AX: session started at offset {}", insertion_offset);

            // Retain the element (CopyAttributeValue already gave us an owning ref)
            SessionResult::Active(TextInsertionSession {
                focused_element: focused,
                insertion_offset,
                inserted_length: 0,
                verified: false,
            })
        }
    }

    /// Replace all previously inserted text with `full_text`.
    ///
    /// Each call selects the range `[insertion_offset .. insertion_offset + inserted_length)`
    /// and replaces it with the new text. This handles ASR hypothesis revisions cleanly.
    pub fn update_text(&mut self, full_text: &str) -> InsertResult {
        if full_text.is_empty() {
            return InsertResult::Ok;
        }

        unsafe {
            // 1. Select the range we previously inserted
            let select_range = CFRange {
                location: self.insertion_offset,
                length: self.inserted_length,
            };

            let range_attr = CFString::new(kAXSelectedTextRangeAttribute);
            let range_value = AXValueCreate(
                kAXValueTypeCFRange,
                &select_range as *const CFRange as *const std::ffi::c_void,
            );

            if range_value.is_null() {
                error!("AX: failed to create CFRange value");
                return InsertResult::Failed;
            }

            let err = AXUIElementSetAttributeValue(
                self.focused_element,
                range_attr.as_concrete_TypeRef(),
                range_value as CFTypeRef,
            );
            CFRelease(range_value as CFTypeRef);

            if let Some(result) = Self::classify_error(err, "set selected range") {
                return result;
            }

            // 2. Replace the selection with the new text
            let text_attr = CFString::new(kAXSelectedTextAttribute);
            let cf_text = CFString::new(full_text);

            let err = AXUIElementSetAttributeValue(
                self.focused_element,
                text_attr.as_concrete_TypeRef(),
                cf_text.as_CFTypeRef(),
            );

            if let Some(result) = Self::classify_error(err, "set selected text") {
                return result;
            }

            // 3. Update our bookkeeping with the new UTF-16 length
            let new_len = cf_text.char_len() as CFIndex;
            self.inserted_length = new_len;

            // 4. On the first successful write, verify by reading cursor position back.
            //    If the cursor didn't move to where we expect, the app silently ignored
            //    the write (common with web content areas, some Electron apps, etc.).
            if !self.verified {
                let expected_cursor = self.insertion_offset + new_len;
                match Self::get_cursor_offset(self.focused_element) {
                    Some(actual) if actual == expected_cursor => {
                        info!("AX: verified — cursor at expected position {}", actual);
                        self.verified = true;
                    }
                    Some(actual) => {
                        warn!(
                            "AX: verification failed — cursor at {} but expected {} \
                             (app likely ignored the write)",
                            actual, expected_cursor
                        );
                        return InsertResult::Failed;
                    }
                    None => {
                        warn!("AX: verification failed — could not read cursor back");
                        return InsertResult::Failed;
                    }
                }
            }

            debug!(
                "AX: updated text ({} UTF-16 units at offset {})",
                self.inserted_length, self.insertion_offset
            );
            InsertResult::Ok
        }
    }

    /// Read the current cursor offset from the focused element's selected text range.
    unsafe fn get_cursor_offset(element: AXUIElementRef) -> Option<CFIndex> {
        let attr = CFString::new(kAXSelectedTextRangeAttribute);
        let mut value_raw: CFTypeRef = ptr::null_mut();

        let err = AXUIElementCopyAttributeValue(
            element,
            attr.as_concrete_TypeRef(),
            &mut value_raw,
        );

        if err != kAXErrorSuccess || value_raw.is_null() {
            return None;
        }

        let mut range = CFRange { location: 0, length: 0 };
        let ok = AXValueGetValue(
            value_raw as AXValueRef,
            kAXValueTypeCFRange,
            &mut range as *mut CFRange as *mut std::ffi::c_void,
        );
        CFRelease(value_raw);

        if ok {
            // Use the start of the selection (or cursor position if no selection)
            Some(range.location)
        } else {
            None
        }
    }

    /// Classify an AX error code into an `InsertResult`.
    /// Returns `None` for success.
    fn classify_error(err: AXError, context: &str) -> Option<InsertResult> {
        if err == kAXErrorSuccess {
            None
        } else if err == kAXErrorCannotComplete {
            warn!("AX: transient error in {}: {}", context, err);
            Some(InsertResult::Retry)
        } else if err == kAXErrorInvalidUIElement {
            error!("AX: element invalidated in {}: {}", context, err);
            Some(InsertResult::Failed)
        } else if err == kAXErrorAttributeUnsupported {
            error!("AX: attribute unsupported in {}: {}", context, err);
            Some(InsertResult::Failed)
        } else {
            error!("AX: unexpected error in {}: {}", context, err);
            Some(InsertResult::Failed)
        }
    }
}

impl Drop for TextInsertionSession {
    fn drop(&mut self) {
        unsafe {
            if !self.focused_element.is_null() {
                CFRelease(self.focused_element as CFTypeRef);
            }
        }
        debug!("AX: session dropped");
    }
}
