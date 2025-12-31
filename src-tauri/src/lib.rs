use global_hotkey::{GlobalHotKeyManager, GlobalHotKeyEvent, hotkey::{Code, HotKey, Modifiers}, HotKeyState};
use log::{info, error, warn};
use tauri::{Manager, Emitter};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use arboard::Clipboard;
use enigo::{Enigo, Key, Keyboard, Settings as EnigoSettings};

mod settings;
mod asr;

pub use settings::Settings;
use asr::RealtimeTranscriber;

/// Application state shared across Tauri commands and event handlers
pub struct AppState {
    /// The real-time transcriber instance (None when not recording)
    pub transcriber: Arc<Mutex<Option<RealtimeTranscriber>>>,
    
    /// Flag indicating whether recording is currently active
    pub is_recording: Arc<AtomicBool>,
    
    /// Application settings
    pub settings: Arc<Mutex<Settings>>,
}

impl AppState {
    /// Create a new AppState with default settings
    pub fn new() -> Self {
        Self {
            transcriber: Arc::new(Mutex::new(None)),
            is_recording: Arc::new(AtomicBool::new(false)),
            settings: Arc::new(Mutex::new(Settings::default())),
        }
    }
    
    /// Create a new AppState with loaded settings
    pub fn with_settings(settings: Settings) -> Self {
        Self {
            transcriber: Arc::new(Mutex::new(None)),
            is_recording: Arc::new(AtomicBool::new(false)),
            settings: Arc::new(Mutex::new(settings)),
        }
    }
}

/// Spawn a background thread to poll for transcription updates
fn spawn_transcription_polling_thread(
    state: &AppState,
    app: tauri::AppHandle,
) {
    let transcriber = state.transcriber.clone();
    let is_recording = state.is_recording.clone();
    
    thread::Builder::new()
        .name("transcription-poller".to_string())
        .spawn(move || {
            info!("Transcription polling thread started");
            
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
                    
                    if let Some(ref mut t) = *transcriber_guard {
                        // Try to get next transcription (non-blocking)
                        t.try_next_transcription().map(|result| result.text)
                    } else {
                        None
                    }
                };
                
                // Emit transcription update if we got one
                if let Some(text) = transcription_text {
                    if !text.is_empty() {
                        info!("Emitting transcription update: {}", text);
                        if let Err(e) = app.emit("transcription-update", text) {
                            error!("Failed to emit transcription-update event: {}", e);
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
    
    // Get and stop transcriber
    let mut transcriber_guard = state.transcriber.lock()
        .map_err(|e| format!("Failed to lock transcriber: {}", e))?;
    
    let final_text = if let Some(ref mut transcriber) = *transcriber_guard {
        // Stop the transcriber
        transcriber.stop();
        
        // Try to get any remaining transcription results
        let mut text = String::new();
        while let Some(result) = transcriber.try_next_transcription() {
            if !result.text.is_empty() {
                text = result.text;
            }
        }
        
        text
    } else {
        String::new()
    };
    
    // Remove transcriber from state
    *transcriber_guard = None;
    drop(transcriber_guard);
    
    // Set recording flag to false
    state.is_recording.store(false, Ordering::SeqCst);
    
    info!("Recording stopped, final text: {}", final_text);
    Ok(final_text)
}

/// Show a window by label
fn show_window(app: &tauri::AppHandle, label: &str) -> Result<(), String> {
    info!("Showing window: {}", label);
    
    // Get window by label
    let window = app.get_webview_window(label)
        .ok_or_else(|| format!("Window not found: {}", label))?;
    
    // Show the window
    window.show()
        .map_err(|e| format!("Failed to show window {}: {}", label, e))?;
    
    info!("Window {} shown successfully", label);
    Ok(())
}

/// Hide a window by label
fn hide_window(app: &tauri::AppHandle, label: &str) -> Result<(), String> {
    info!("Hiding window: {}", label);
    
    // Get window by label
    let window = app.get_webview_window(label)
        .ok_or_else(|| format!("Window not found: {}", label))?;
    
    // Hide the window
    window.hide()
        .map_err(|e| format!("Failed to hide window {}: {}", label, e))?;
    
    info!("Window {} hidden successfully", label);
    Ok(())
}

/// Copy text to clipboard and simulate Cmd+V paste
fn copy_and_paste(text: &str) -> Result<(), String> {
    info!("Copying and pasting text: {}", text);
    
    // Handle empty text
    if text.is_empty() {
        warn!("Empty text provided, skipping paste");
        return Ok(());
    }
    
    // Copy text to clipboard
    let mut clipboard = Clipboard::new()
        .map_err(|e| format!("Failed to access clipboard: {}", e))?;
    
    clipboard.set_text(text)
        .map_err(|e| format!("Failed to copy text to clipboard: {}", e))?;
    
    info!("Text copied to clipboard successfully");
    
    // Small delay to ensure clipboard is ready
    thread::sleep(Duration::from_millis(50));
    
    // Simulate Cmd+V keypress
    let mut enigo = Enigo::new(&EnigoSettings::default())
        .map_err(|e| format!("Failed to create keyboard controller: {}", e))?;
    
    // Press Cmd+V (Meta key is Cmd on macOS)
    enigo.key(Key::Meta, enigo::Direction::Press)
        .map_err(|e| format!("Failed to press Cmd key: {}", e))?;
    enigo.key(Key::Unicode('v'), enigo::Direction::Click)
        .map_err(|e| format!("Failed to press V key: {}", e))?;
    enigo.key(Key::Meta, enigo::Direction::Release)
        .map_err(|e| format!("Failed to release Cmd key: {}", e))?;
    
    info!("Paste command simulated successfully");
    
    Ok(())
}

/// Start recording and transcription
fn start_recording(state: &AppState, app: &tauri::AppHandle) -> Result<(), String> {
    info!("Starting recording");
    
    // Check if already recording
    if state.is_recording.load(Ordering::SeqCst) {
        warn!("Already recording, ignoring start request");
        return Err("Already recording".to_string());
    }
    
    // Get settings
    let settings = state.settings.lock()
        .map_err(|e| format!("Failed to lock settings: {}", e))?
        .clone();
    
    // Create transcriber config
    let config = asr::TranscriberConfig {
        model_name: settings.model.clone(),
        language: settings.language.clone(),
        ..Default::default()
    };
    
    // Create transcriber
    let mut transcriber = RealtimeTranscriber::new(config, candle_core::Device::Cpu)
        .map_err(|e| format!("Failed to create transcriber: {}", e))?;
    
    // Start transcriber
    transcriber.start()
        .map_err(|e| format!("Failed to start transcriber: {}", e))?;
    
    // Store transcriber in state
    *state.transcriber.lock()
        .map_err(|e| format!("Failed to lock transcriber: {}", e))? = Some(transcriber);
    
    // Set recording flag
    state.is_recording.store(true, Ordering::SeqCst);
    
    // Emit show-window event to frontend
    app.emit("show-window", ())
        .map_err(|e| format!("Failed to emit show-window event: {}", e))?;
    
    // Spawn transcription polling thread
    spawn_transcription_polling_thread(state, app.clone());
    
    info!("Recording started successfully");
    Ok(())
}

/// Handle the "Start Recording" menu item
fn handle_start_recording(app: &tauri::AppHandle) {
    info!("Start Recording triggered from menu");
    
    // Get the app state
    let state = app.state::<AppState>();
    
    // Start recording
    if let Err(e) = start_recording(&state, app) {
        error!("Failed to start recording: {}", e);
    }
}

/// Handle the "Quit" menu item
fn handle_quit(app: &tauri::AppHandle) {
    info!("Quit triggered from menu");
    
    // Get the app state
    let state = app.state::<AppState>();
    
    // Stop recording if active
    if state.is_recording.load(Ordering::SeqCst) {
        log::info!("Stopping active recording before quit");
        if let Err(e) = stop_recording(&state, app) {
            error!("Failed to stop recording: {}", e);
        }
    }
    
    // Exit the application
    app.exit(0);
}

/// Handle global hotkey events (press and release)
fn handle_hotkey_event(app: &tauri::AppHandle, event: GlobalHotKeyEvent) {
    let state = app.state::<AppState>();
    
    match event.state {
        HotKeyState::Pressed => {
            info!("Hotkey pressed - starting recording");
            
            // Start recording
            if let Err(e) = start_recording(&state, app) {
                error!("Failed to start recording: {}", e);
            }
        }
        HotKeyState::Released => {
            info!("Hotkey released - stopping recording");
            
            // Stop recording
            match stop_recording(&state, app) {
                Ok(text) => {
                    info!("Recording stopped with text: {}", text);
                    
                    // Copy and paste the transcription
                    if let Err(e) = copy_and_paste(&text) {
                        error!("Failed to copy and paste: {}", e);
                    }
                    
                    // Wait 1 second before hiding window
                    thread::sleep(Duration::from_secs(1));
                    
                    // Emit hide-window event
                    if let Err(e) = app.emit("hide-window", ()) {
                        error!("Failed to emit hide-window event: {}", e);
                    }
                }
                Err(e) => {
                    error!("Failed to stop recording: {}", e);
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
            
            loop {
                // Poll for hotkey events with a small delay to avoid busy-waiting
                if let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
                    handle_hotkey_event(&app, event);
                }
                
                // Sleep briefly to avoid consuming too much CPU
                thread::sleep(Duration::from_millis(10));
            }
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
    
    let settings = state.settings.lock()
        .map_err(|e| format!("Failed to lock settings: {}", e))?
        .clone();
    
    Ok(settings)
}

/// Update settings in AppState and save to config file
#[tauri::command]
fn update_settings(state: tauri::State<AppState>, settings: Settings) -> Result<(), String> {
    info!("Updating settings: {:?}", settings);
    
    // Update settings in AppState
    {
        let mut state_settings = state.settings.lock()
            .map_err(|e| format!("Failed to lock settings: {}", e))?;
        *state_settings = settings.clone();
    }
    
    // Save settings to config file
    settings.save()
        .map_err(|e| format!("Failed to save settings: {}", e))?;
    
    info!("Settings updated and saved successfully");
    Ok(())
}

/// Manually start recording (for UI control)
#[tauri::command]
fn manual_start_recording(state: tauri::State<AppState>, app: tauri::AppHandle) -> Result<(), String> {
    info!("Manual start recording triggered");
    start_recording(&state, &app)
}

/// Manually stop recording (for UI control)
#[tauri::command]
fn manual_stop_recording(state: tauri::State<AppState>, app: tauri::AppHandle) -> Result<String, String> {
    info!("Manual stop recording triggered");
    stop_recording(&state, &app)
}

pub fn run() {
    info!("🦉🦉🦉🦉🦉🦉🦉🦉 UNTER WHISPER STARTING 🦉🦉🦉🦉🦉🦉🦉🦉");

    // Load settings from config file
    let settings = Settings::load().unwrap_or_else(|e| {
        log::warn!("Failed to load settings: {}. Using defaults.", e);
        Settings::default()
    });
    
    // Initialize application state
    let app_state = AppState::with_settings(settings);

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            greet,
            get_settings,
            update_settings,
            manual_start_recording,
            manual_stop_recording
        ])
        .setup(move |app| {
            // Set up logging
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(if cfg!(debug_assertions) {
                        log::LevelFilter::Debug
                    } else {
                        log::LevelFilter::Info
                    })
                    .build(),
            )?;

            info!("Setting up system tray...");
            
            // Create system tray menu
            let menu = tauri::menu::MenuBuilder::new(app)
                .item(
                    &tauri::menu::MenuItemBuilder::with_id("start_recording", "Start Recording")
                        .build(app)?,
                )
                .separator()
                .item(
                    &tauri::menu::MenuItemBuilder::with_id("quit", "Quit")
                        .build(app)?,
                )
                .build()?;

            // Create system tray
            let _tray = tauri::tray::TrayIconBuilder::new()
                .menu(&menu)
                .on_menu_event(|app, event| {
                    match event.id().as_ref() {
                        "start_recording" => {
                            handle_start_recording(app);
                        }
                        "quit" => {
                            handle_quit(app);
                        }
                        _ => {}
                    }
                })
                .build(app)?;

            info!("System tray created successfully");

            // Register global hotkey (Cmd+Shift+Space)
            info!("Registering global hotkey (Cmd+Shift+Space)...");
            
            let hotkey_manager = match GlobalHotKeyManager::new() {
                Ok(manager) => manager,
                Err(e) => {
                    error!("Failed to create hotkey manager: {}", e);
                    
                    // Log error for user to see
                    error!("Hotkey Registration Failed: Failed to initialize hotkey system: {}. The app will continue without hotkey support.", e);
                    
                    warn!("Continuing without hotkey support");
                    return Ok(());
                }
            };
            
            let hotkey = HotKey::new(Some(Modifiers::SUPER | Modifiers::SHIFT), Code::Space);
            
            if let Err(e) = hotkey_manager.register(hotkey) {
                error!("Failed to register hotkey: {}", e);
                
                // Log error for user to see
                error!("Hotkey Registration Failed: Failed to register Cmd+Shift+Space hotkey: {}. The hotkey may be in use by another application. You can still use the tray menu to start recording.", e);
                
                warn!("Continuing without hotkey support");
                return Ok(());
            }

            info!("Global hotkey registered successfully");
            
            // Spawn background thread to poll for hotkey events
            spawn_hotkey_polling_thread(app.handle().clone());
            
            info!("Tauri setup completed successfully");

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
