use global_hotkey::{GlobalHotKeyManager, GlobalHotKeyEvent, hotkey::{Code, HotKey, Modifiers}, HotKeyState};
use log::{debug, info, error, warn};
use tauri::{Manager, Emitter};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use arboard::Clipboard;
use enigo::{Enigo, Key, Keyboard, Settings as EnigoSettings};

mod settings;
mod asr;
#[cfg(target_os = "macos")]
mod ax_text;

pub use settings::Settings;
use asr::NemoTranscriber;

/// Check if accessibility permissions are granted on macOS
#[cfg(target_os = "macos")]
fn check_accessibility_permission() -> bool {
    debug!("Checking accessibility permissions...");
    
    let has_permission = unsafe {
        accessibility_sys::AXIsProcessTrustedWithOptions(std::ptr::null())
    };
    
    debug!("Accessibility permission check result: {}", has_permission);
    has_permission
}

/// Check if microphone permissions are granted on macOS
#[cfg(target_os = "macos")]
fn check_microphone_permission() -> bool {
    use std::process::Command;
    
    debug!("Checking microphone permissions...");
    
    // Use osascript to check microphone permission status
    // This is a workaround since Rust doesn't have direct AVFoundation bindings
    let output = Command::new("osascript")
        .arg("-e")
        .arg("tell application \"System Events\" to return true")
        .output();
    
    // If we can run osascript, we assume permissions are OK for now
    // A more robust check would require Objective-C bindings
    let result = output.is_ok();
    debug!("Microphone permission check result: {}", result);
    result
}

/// Application state shared across Tauri commands and event handlers
pub struct AppState {
    /// The NeMo transcriber instance (created once, reused)
    pub transcriber: Arc<Mutex<Option<NemoTranscriber>>>,

    /// Flag indicating whether recording is currently active
    pub is_recording: Arc<AtomicBool>,

    /// Application settings
    pub settings: Arc<Mutex<Option<Settings>>>,

    /// Global hotkey manager (kept alive for app lifetime)
    pub hotkey_manager: Arc<Mutex<Option<GlobalHotKeyManager>>>,

    /// Whether the current recording session is using AX text insertion
    pub ax_insertion_active: Arc<AtomicBool>,

    /// Latest transcription text (for clipboard fallback when AX is unavailable)
    pub last_transcription: Arc<Mutex<String>>,
}

impl AppState {
    /// Create a new empty AppState
    pub fn new() -> Self {
        Self {
            transcriber: Arc::new(Mutex::new(None)),
            is_recording: Arc::new(AtomicBool::new(false)),
            settings: Arc::new(Mutex::new(None)),
            hotkey_manager: Arc::new(Mutex::new(None)),
            ax_insertion_active: Arc::new(AtomicBool::new(false)),
            last_transcription: Arc::new(Mutex::new(String::new())),
        }
    }

    /// Initialize the transcriber and settings
    pub fn initialize_with_settings(&self, settings: Settings) -> anyhow::Result<()> {
        info!("Initializing AppState with transcriber");

        // Always store settings first so user can change them via UI
        {
            let mut settings_guard = self.settings.lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock settings: {}", e))?;
            *settings_guard = Some(settings.clone());
        }

        // Create transcriber config from settings
        let config = asr::TranscriberConfig {
            model_path: settings.model_path.clone().into(),
            mode_idx: settings.mode_idx,
        };

        // Create transcriber once
        let transcriber = NemoTranscriber::new(config)?;

        info!("Transcriber created successfully");

        {
            let mut transcriber_guard = self.transcriber.lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock transcriber: {}", e))?;
            *transcriber_guard = Some(transcriber);
        }

        info!("AppState initialized successfully");
        Ok(())
    }

    /// Update the settings in the AppState
    pub fn update_settings(&self, new_settings: Settings) -> anyhow::Result<()> {
        info!("Updating AppState settings");

        // Create transcriber config from settings
        let config = asr::TranscriberConfig {
            model_path: new_settings.model_path.clone().into(),
            mode_idx: new_settings.mode_idx,
        };

        // Create a new transcriber
        let transcriber = NemoTranscriber::new(config)?;

        {
            let mut transcriber_guard = self.transcriber.lock()
                .map_err(|e|  anyhow::anyhow!("Failed to lock transcriber: {}", e))?;
            *transcriber_guard = Some(transcriber);
        }
        {
            let mut settings_guard = self.settings.lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock settings: {}", e))?;
            *settings_guard = Some(new_settings);
        }

        info!("Settings updated successfully in AppState");
        Ok(())
    }
    
}

/// Spawn a background thread to poll for transcription updates.
///
/// On macOS, attempts to create an AX text insertion session to stream text
/// directly into the focused app. Falls back to accumulating text for
/// clipboard paste if AX is unavailable.
fn spawn_transcription_polling_thread(
    state: &AppState,
    _app: tauri::AppHandle,
) {
    let transcriber = state.transcriber.clone();
    let is_recording = state.is_recording.clone();
    let ax_insertion_active = state.ax_insertion_active.clone();
    let last_transcription = state.last_transcription.clone();

    thread::Builder::new()
        .name("transcription-poller".to_string())
        .spawn(move || {
            info!("Transcription polling thread started");

            // Try to open an AX text insertion session
            #[cfg(target_os = "macos")]
            let mut ax_session = match ax_text::TextInsertionSession::begin() {
                ax_text::SessionResult::Active(session) => {
                    info!("AX text insertion session created");
                    ax_insertion_active.store(true, Ordering::SeqCst);
                    Some(session)
                }
                ax_text::SessionResult::FallbackNeeded => {
                    info!("AX text insertion unavailable, will use clipboard fallback");
                    ax_insertion_active.store(false, Ordering::SeqCst);
                    None
                }
            };

            #[cfg(not(target_os = "macos"))]
            {
                ax_insertion_active.store(false, Ordering::SeqCst);
            }

            loop {
                // Check if still recording
                if !is_recording.load(Ordering::SeqCst) {
                    info!("Recording stopped, exiting transcription polling thread");
                    break;
                }

                // Try to get transcription update
                let transcription_text = {
                    let mut transcriber_guard = match transcriber.lock() {
                        Ok(guard) => guard,
                        Err(e) => {
                            error!("Failed to lock transcriber: {}", e);
                            thread::sleep(Duration::from_millis(100));
                            continue;
                        }
                    };

                    if let Some(transcriber) = transcriber_guard.as_mut() {
                        transcriber.try_next_transcription().map(|result| result.text)
                    } else {
                        None
                    }
                };

                if let Some(text) = transcription_text {
                    if !text.is_empty() {
                        // Always store the latest text for clipboard fallback
                        if let Ok(mut last) = last_transcription.lock() {
                            *last = text.clone();
                        }

                        #[cfg(target_os = "macos")]
                        if let Some(ref mut session) = ax_session {
                            match session.update_text(&text) {
                                ax_text::InsertResult::Ok => {
                                    debug!("AX: streamed text update");
                                }
                                ax_text::InsertResult::Retry => {
                                    debug!("AX: transient error, will retry");
                                }
                                ax_text::InsertResult::Failed => {
                                    warn!("AX: permanent failure, disabling AX for this session");
                                    ax_session = None;
                                    ax_insertion_active.store(false, Ordering::SeqCst);
                                }
                            }
                        }
                    }
                }

                // Sleep briefly to avoid busy-waiting
                thread::sleep(Duration::from_millis(100));
            }

            info!("Transcription polling thread stopped");
        })
        .expect("Failed to spawn transcription polling thread");
}

/// Stop recording and return final transcription text
fn stop_recording(state: &AppState, _app: &tauri::AppHandle) -> Result<String, String> {
    info!("Stopping recording");
    
    // Check if actually recording
    if !state.is_recording.load(Ordering::SeqCst) {
        warn!("Not recording, ignoring stop request");
        return Err("Not recording".to_string());
    }
    
    // Get transcriber and stop it
    let mut transcriber_guard = state.transcriber.lock()
        .map_err(|e| format!("Failed to lock transcriber: {}", e))?;
    
    let transcriber = transcriber_guard.as_mut()
        .ok_or_else(|| "Transcriber not initialized".to_string())?;
    
    // Stop the transcriber
    transcriber.stop();
    
    // Try to get any remaining transcription results
    let mut final_text = String::new();
    while let Some(result) = transcriber.try_next_transcription() {
        if !result.text.is_empty() {
            final_text = result.text;
        }
    }
    
    drop(transcriber_guard);
    
    // Set recording flag to false
    state.is_recording.store(false, Ordering::SeqCst);
    
    // Check if no speech was detected
    if final_text.trim().is_empty() {
        warn!("No speech detected in recording");
    }
    
    info!("Recording stopped, final text: {}", final_text);
    Ok(final_text)
}

/// Copy text to clipboard and simulate Cmd+V paste
fn copy_and_paste(text: &str, _app: &tauri::AppHandle) -> Result<(), String> {
    info!("Copying and pasting text: {}", text);
    
    // Handle empty text
    if text.is_empty() {
        warn!("Empty text provided, skipping paste");
        return Ok(());
    }
    
    // Copy text to clipboard
    let mut clipboard = Clipboard::new()
        .map_err(|e| {
            error!("Failed to access clipboard: {}", e);
            format!("Failed to access clipboard: {}", e)
        })?;
    
    clipboard.set_text(text)
        .map_err(|e| {
            error!("Failed to copy text to clipboard: {}", e);
            format!("Failed to copy text to clipboard: {}", e)
        })?;
    
    info!("Text copied to clipboard successfully");
    
    // Small delay to ensure clipboard is ready
    thread::sleep(Duration::from_millis(50));
    
    // Simulate Cmd+V keypress
    let mut enigo = Enigo::new(&EnigoSettings::default())
        .map_err(|e| {
            error!("Failed to create keyboard controller: {}", e);
            format!("Failed to create keyboard controller: {}", e)
        })?;
    
    // Press Cmd+V (Meta key is Cmd on macOS)
    enigo.key(Key::Meta, enigo::Direction::Press)
        .map_err(|e| {
            error!("Failed to press Cmd key: {}", e);
            format!("Failed to press Cmd key: {}", e)
        })?;
    enigo.key(Key::Unicode('v'), enigo::Direction::Click)
        .map_err(|e| {
            error!("Failed to press V key: {}", e);
            format!("Failed to press V key: {}", e)
        })?;
    enigo.key(Key::Meta, enigo::Direction::Release)
        .map_err(|e| {
            error!("Failed to release Cmd key: {}", e);
            format!("Failed to release Cmd key: {}", e)
        })?;
    
    info!("Paste command simulated successfully");
    
    Ok(())
}

/// Start recording and transcription
fn start_recording(state: &AppState, app: &tauri::AppHandle) -> anyhow::Result<()> {
    info!("Starting recording");
    debug!("=== START RECORDING CALLED ===");

    // Check if already recording
    let is_recording = state.is_recording.load(Ordering::SeqCst);
    debug!("Current recording state: {}", is_recording);
    
    if is_recording {
        warn!("Already recording, ignoring start request");
        anyhow::bail!("Already recording");
    }

    /// // Check microphone permissions
    /// debug!("Checking microphone permissions...");
    /// if !check_microphone_permission() {
    ///     error!("Microphone permission denied - please grant microphone access in System Settings > Privacy & Security > Microphone");
    ///     anyhow::bail!("Microphone permission denied");
    /// }
    /// debug!("Microphone permissions OK");

    // Get device_id and device_name from settings
    let (device_id, device_name) = {
        let settings = state.settings.lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock settings: {}", e))?;
        let settings = settings.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Settings not initialized"))?;
        (settings.device_id.clone(), settings.get_device_name())
    };
    
    info!("Attempting to use audio device: {}", device_name);

    // Create recorder with selected device, with fallback to system default
    let recorder = match asr::audio::AudioRecorder::from_device_id(&device_id) {
        Ok(rec) => {
            info!("Successfully initialized device: {}", device_name);
            rec
        }
        Err(e) => {
            warn!("Failed to use selected device '{}': {}. Falling back to system default.", device_name, e);
            
            // Emit device-fallback event to notify user
            let fallback_message = format!("Selected device '{}' unavailable. Using system default.", device_name);
            if let Err(emit_err) = app.emit("device-fallback", fallback_message.clone()) {
                error!("Failed to emit device-fallback event: {}", emit_err);
            }
            
            // Try to use system default
            match asr::audio::AudioRecorder::from_device_id(&asr::audio::DeviceId::SystemDefault) {
                Ok(rec) => {
                    info!("Successfully fell back to system default device");
                    rec
                }
                Err(fallback_err) => {
                    error!("Failed to initialize system default device: {}", fallback_err);
                    
                    // Emit device-error event
                    let error_message = format!("Failed to initialize audio device: {}", fallback_err);
                    if let Err(emit_err) = app.emit("device-error", error_message.clone()) {
                        error!("Failed to emit device-error event: {}", emit_err);
                    }
                    
                    anyhow::bail!("Failed to initialize any audio device: {}", fallback_err);
                }
            }
        }
    };

    // Lazily initialize transcriber if it wasn't created at startup
    {
        let needs_init = state.transcriber.lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock transcriber: {}", e))?
            .is_none();
        if needs_init {
            info!("Transcriber not yet initialized, attempting lazy init...");
            let settings = state.settings.lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock settings: {}", e))?;
            let settings = settings.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Settings not initialized"))?;
            let config = asr::TranscriberConfig {
                model_path: settings.model_path.clone().into(),
                mode_idx: settings.mode_idx,
            };
            let transcriber = asr::NemoTranscriber::new(config)?;
            let mut guard = state.transcriber.lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock transcriber: {}", e))?;
            *guard = Some(transcriber);
            info!("Transcriber lazily initialized");
        }
    }

    // Get transcriber and start it with the recorder
    debug!("Acquiring transcriber lock...");
    let mut transcriber_guard = state.transcriber.lock()
        .map_err(|e| {
            error!("Failed to lock transcriber: {}", e);
            anyhow::anyhow!("Failed to lock transcriber: {}", e)
        })?;
    debug!("Transcriber lock acquired");

    let transcriber = transcriber_guard.as_mut()
        .ok_or_else(|| anyhow::anyhow!("Transcriber not initialized"))?;

    // Start transcriber with custom recorder
    debug!("Starting transcriber with custom recorder...");
    transcriber.start_with_recorder(recorder)?;
    debug!("Transcriber started successfully");

    drop(transcriber_guard);

    // Set recording flag
    debug!("Setting recording flag to true...");
    state.is_recording.store(true, Ordering::SeqCst);
    debug!("Recording flag set to true");

    // Clear last transcription for this session
    if let Ok(mut last) = state.last_transcription.lock() {
        last.clear();
    }

    // Spawn transcription polling thread (will attempt AX text insertion)
    debug!("Spawning transcription polling thread...");
    spawn_transcription_polling_thread(state, app.clone());
    debug!("Transcription polling thread spawned");
    
    info!("Recording started successfully");
    debug!("=== START RECORDING COMPLETED ===");
    Ok(())
}

/// Handle the "Start Recording" menu item
fn handle_start_recording(app: &tauri::AppHandle) {
    info!("Start Recording triggered from menu");
    debug!("=== TRAY MENU: START RECORDING CLICKED ===");

    // Get the app state
    debug!("Getting app state...");
    let state = app.state::<AppState>();
    debug!("App state acquired");

    // Start recording
    debug!("Calling start_recording...");
    if let Err(e) = start_recording(&state, app) {
        error!("Failed to start recording from menu: {:?}", e);
        debug!("=== TRAY MENU: START RECORDING FAILED ===");
    } else {
        debug!("=== TRAY MENU: START RECORDING SUCCESS ===");
    }
}

/// Handle the "Settings" menu item
fn handle_settings(app: &tauri::AppHandle) {
    info!("Settings triggered from menu");
    debug!("=== TRAY MENU: SETTINGS CLICKED ===");

    // Temporarily show app in dock when settings window is open
    app.set_activation_policy(tauri::ActivationPolicy::Regular);

    // Show and focus the settings window
    if let Some(window) = app.get_webview_window("main") {
        if let Err(e) = window.show() {
            error!("Failed to show settings window: {}", e);
        }
        if let Err(e) = window.set_focus() {
            error!("Failed to focus settings window: {}", e);
        }
        debug!("=== TRAY MENU: SETTINGS SUCCESS ===");
    } else {
        error!("Settings window not found");
        debug!("=== TRAY MENU: SETTINGS FAILED ===");
    }
}

/// Handle the "Quit" menu item
fn handle_quit(_app: &tauri::AppHandle) {
    info!("Quit triggered from menu");
    
    // Exit the application
    _app.exit(0);
}

/// Handle global hotkey events (press and release)
fn handle_hotkey_event(app: &tauri::AppHandle, event: GlobalHotKeyEvent) {
    let state = app.state::<AppState>();
    
    match event.state {
        HotKeyState::Pressed => {
            info!("Hotkey pressed - starting recording");
            
            // Start recording
            if let Err(e) = start_recording(&state, app) {
                error!("Failed to start recording: {:?}", e);
            }
        }
        HotKeyState::Released => {
            info!("Hotkey released - stopping recording");

            let ax_was_active = state.ax_insertion_active.load(Ordering::SeqCst);

            // Stop recording (drains remaining results)
            match stop_recording(&state, app) {
                Ok(final_text) => {
                    info!("Recording stopped with text: {}", final_text);

                    if ax_was_active {
                        // Text was already streamed into the app via AX — nothing to do
                        info!("AX insertion was active, text already in place");
                    } else {
                        // Fallback: get the latest transcription and clipboard-paste it
                        let text = {
                            let last = state.last_transcription.lock()
                                .map(|t| t.clone())
                                .unwrap_or_default();
                            if last.is_empty() { final_text } else { last }
                        };

                        if let Err(e) = copy_and_paste(&text, app) {
                            error!("Failed to copy and paste: {}", e);
                        }
                    }

                    // Reset for next session
                    state.ax_insertion_active.store(false, Ordering::SeqCst);
                }
                Err(e) => {
                    error!("Failed to stop recording: {}", e);
                    state.ax_insertion_active.store(false, Ordering::SeqCst);
                }
            }
        }
    }
}

/// Spawn a background thread to poll for hotkey events
fn spawn_hotkey_polling_thread(app: tauri::AppHandle) {
    thread::Builder::new()
        .name("hotkey-poller".to_string())
        .spawn(move || {
            info!("Hotkey polling thread started");
            
            let receiver = GlobalHotKeyEvent::receiver();
            loop {
                // Use blocking recv() to wait for events without busy-waiting
                match receiver.recv() {
                    Ok(event) => {
                        debug!("Hotkey event received: {:?}", event);
                        handle_hotkey_event(&app, event);
                    }
                    Err(e) => {
                        error!("Hotkey receiver error: {}", e);
                        break;
                    }
                }
            }
            
            info!("Hotkey polling thread stopped");
        })
        .expect("Failed to spawn hotkey polling thread");
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Get current settings from AppState
#[tauri::command]
fn get_settings(state: tauri::State<AppState>) -> Result<Settings, String> {
    info!("Getting current settings");
    
    let settings_guard = state.settings.lock()
        .map_err(|e| format!("Failed to lock settings: {}", e))?;
    
    let settings = settings_guard.as_ref()
        .ok_or_else(|| "Settings not initialized".to_string())?
        .clone();
    
    Ok(settings)
}

/// Update settings in AppState and save to config file
#[tauri::command]
fn update_settings(state: tauri::State<AppState>, settings: Settings) -> Result<(), String> {
    info!("Updating settings: {:?}", settings);
    
    // Update settings in AppState (this recreates the transcriber)
    state.update_settings(settings.clone())
        .map_err(|e| format!("Failed to update settings: {}", e))?;
    
    // Save settings to config file
    settings.save()
        .map_err(|e| format!("Failed to save settings: {}", e))?;
    
    info!("Settings updated and saved successfully");
    Ok(())
}

/// Get list of available audio input devices
#[tauri::command]
fn get_audio_devices() -> Vec<asr::audio::AudioDeviceInfo> {
    info!("Getting available audio input devices");
    
    match asr::audio::AudioRecorder::list_input_devices_with_info() {
        Ok(devices) => {
            info!("Successfully enumerated {} audio devices", devices.len());
            devices
        }
        Err(e) => {
            error!("Failed to enumerate audio devices: {}", e);
            // Return empty list on error as per requirements 8.1
            Vec::new()
        }
    }
}

/// Manually start recording (for UI control)
#[tauri::command]
fn manual_start_recording(state: tauri::State<AppState>, app: tauri::AppHandle) -> Result<(), String> {
    info!("Manual start recording triggered");
    start_recording(&state, &app).map_err(|e| e.to_string())
}

/// Manually stop recording (for UI control)
#[tauri::command]
fn manual_stop_recording(state: tauri::State<AppState>, app: tauri::AppHandle) -> Result<String, String> {
    info!("Manual stop recording triggered");
    stop_recording(&state, &app)
}

/// Hide app from dock (macOS only)
#[tauri::command]
fn hide_from_dock(app: tauri::AppHandle) -> Result<(), String> {
    info!("Hiding app from dock");
    app.set_activation_policy(tauri::ActivationPolicy::Prohibited);
    Ok(())
}

pub fn run() {
    eprintln!("🦉🦉🦉🦉🦉🦉🦉🦉 UNTER WHISPER STARTING 🦉🦉🦉🦉🦉🦉🦉🦉");
    eprintln!("Debug logging enabled");
    
    // Print log file location
    let log_dir = dirs::data_local_dir()
        .map(|p| p.join("unterwhisper").join("logs"))
        .unwrap_or_else(|| std::path::PathBuf::from("logs"));
    eprintln!("Log file location: {:?}/app.log", log_dir);
    eprintln!("You can tail the logs with: tail -f {:?}/app.log", log_dir);

    // Initialize empty application state
    eprintln!("Initializing empty application state...");
    let app_state = AppState::new();
    eprintln!("Empty application state created");

    eprintln!("Building Tauri application...");
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_notification::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            greet,
            get_settings,
            update_settings,
            get_audio_devices,
            manual_start_recording,
            manual_stop_recording,
            hide_from_dock
        ])
        .setup(move |app| {
            // Set up logging with file output
            let log_dir = dirs::data_local_dir()
                .map(|p| p.join("unterwhisper").join("logs"))
                .unwrap_or_else(|| std::path::PathBuf::from("logs"));
            
            // Create log directory if it doesn't exist
            if let Err(e) = std::fs::create_dir_all(&log_dir) {
                eprintln!("Failed to create log directory: {}", e);
            }
            
            let log_file = log_dir.join("app.log");
            
            eprintln!("=== UNTERWHISPER SETUP STARTING ===");
            eprintln!("Logging to file: {:?}", log_file);
            
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Debug)  // Always use Debug level
                    .format(|out, message, record| {
                        out.finish(format_args!(
                            "[{} {} {}] {}",
                            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                            record.level(),
                            record.target(),
                            message
                        ))
                    })
                    .target(tauri_plugin_log::Target::new(
                        tauri_plugin_log::TargetKind::LogDir { file_name: Some("app".to_string()) }
                    ))
                    .build(),
            )?;

            info!("=== UNTERWHISPER SETUP: Logging initialized ===");
            debug!("Log directory: {:?}", log_dir);
            debug!("Log file: {:?}", log_file);

            // Load settings from config file
            info!("Loading settings...");
            let settings = Settings::load().unwrap_or_else(|e| {
                warn!("Failed to load settings: {}. Using defaults.", e);
                Settings::default()
            });
            info!("Settings loaded: model_path={}, mode_idx={}", settings.model_path, settings.mode_idx);
            
            // Initialize application state with transcriber
            info!("Initializing transcriber with settings...");
            let state = app.state::<AppState>();
            match state.initialize_with_settings(settings) {
                Ok(()) => info!("Transcriber initialized successfully"),
                Err(e) => {
                    error!("Failed to initialize transcriber: {}. App will start without transcriber — fix the model path in Settings.", e);
                    // Store settings even if transcriber fails, so user can fix via UI
                }
            }

            // Configure app to not show in dock on macOS
            app.set_activation_policy(tauri::ActivationPolicy::Prohibited);

            info!("Setting up system tray...");
            debug!("Creating tray menu...");
            
            // Create system tray menu
            debug!("Building menu items...");
            let menu = tauri::menu::MenuBuilder::new(app)
                .item(
                    &tauri::menu::MenuItemBuilder::with_id("start_recording", "Start Recording")
                        .build(app)?,
                )
                .item(
                    &tauri::menu::MenuItemBuilder::with_id("settings", "Settings")
                        .build(app)?,
                )
                .separator()
                .item(
                    &tauri::menu::MenuItemBuilder::with_id("quit", "Quit")
                        .build(app)?,
                )
                .build()?;
            
            debug!("Menu built successfully");
            debug!("Creating tray icon...");

            // Create system tray with custom icon
            let icon_bytes = include_bytes!("../icons/OwlHead-EyesHatOnly.png");
            let _tray = tauri::tray::TrayIconBuilder::new()
                .icon(tauri::image::Image::from_bytes(icon_bytes)?)
                .menu(&menu)
                .on_menu_event(|app, event| {
                    debug!("Tray menu event received: {:?}", event.id());
                    match event.id().as_ref() {
                        "start_recording" => {
                            debug!("Tray menu: 'start_recording' selected");
                            handle_start_recording(app);
                        }
                        "settings" => {
                            debug!("Tray menu: 'settings' selected");
                            handle_settings(app);
                        }
                        "quit" => {
                            debug!("Tray menu: 'quit' selected");
                            handle_quit(app);
                        }
                        _ => {
                            debug!("Tray menu: unknown event {:?}", event.id());
                        }
                    }
                })
                .build(app)?;

            info!("System tray created successfully");

            // Register global hotkey (Option+V)
            info!("Registering global hotkey (Option+V)...");
            
            // Check accessibility permissions first
            if !check_accessibility_permission() {
                warn!("Accessibility permissions not granted - please grant Accessibility permission in System Settings > Privacy & Security > Accessibility to enable global hotkeys. You can still use the tray menu to start recording.");
            }
            
            let hotkey_manager = match GlobalHotKeyManager::new() {
                Ok(manager) => manager,
                Err(e) => {
                    error!("Failed to create hotkey manager: {} - please check that Accessibility permissions are granted in System Settings. You can still use the tray menu to start recording.", e);
                    warn!("Continuing without hotkey support");
                    return Ok(());
                }
            };
            
            let hotkey = HotKey::new(Some(Modifiers::ALT), Code::KeyV);
            
            if let Err(e) = hotkey_manager.register(hotkey) {
                error!("Failed to register hotkey: {} - the hotkey may be in use by another application, or Accessibility permissions may not be granted. You can still use the tray menu to start recording.", e);
                warn!("Continuing without hotkey support");
                return Ok(());
            }

            info!("Global hotkey registered successfully");
            
            // Store hotkey manager in AppState to keep it alive for app lifetime
            let state = app.state::<AppState>();
            *state.hotkey_manager.lock().unwrap() = Some(hotkey_manager);
            
            // Spawn background thread to poll for hotkey events
            spawn_hotkey_polling_thread(app.handle().clone());
            
            info!("Unterwhisper started - press Option+V to start recording, or use the tray menu");
            info!("Tauri setup completed successfully");

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
