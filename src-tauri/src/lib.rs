use candle_core::Device;
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

pub use settings::Settings;
use asr::RealtimeTranscriber;

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
    /// The real-time transcriber instance (created once, reused)
    pub transcriber: Arc<Mutex<RealtimeTranscriber>>,
    
    /// Flag indicating whether recording is currently active
    pub is_recording: Arc<AtomicBool>,
    
    /// Application settings
    pub settings: Arc<Mutex<Settings>>,
    
    /// Global hotkey manager (kept alive for app lifetime)
    pub hotkey_manager: Arc<Mutex<Option<GlobalHotKeyManager>>>,
}

impl AppState {
    /// Create a new AppState with transcriber initialized
    pub fn with_settings(settings: Settings) -> anyhow::Result<Self> {
        info!("Initializing AppState with transcriber");
        
        // Create transcriber config from settings
        let config = asr::TranscriberConfig {
            model_name: settings.model.clone(),
            language: settings.language.clone(),
            ..Default::default()
        };
        
        // Create device (Metal for macOS)
        let device = Device::new_metal(0)?;
        
        // Create transcriber once
        let transcriber = RealtimeTranscriber::new(config, device)?;
        
        info!("Transcriber created successfully");
        
        Ok(Self {
            transcriber: Arc::new(Mutex::new(transcriber)),
            is_recording: Arc::new(AtomicBool::new(false)),
            settings: Arc::new(Mutex::new(settings)),
            hotkey_manager: Arc::new(Mutex::new(None)),
        })
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
                    
                    // Try to get next transcription (non-blocking)
                    transcriber_guard.try_next_transcription().map(|result| result.text)
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
    
    // Get transcriber and stop it
    let mut transcriber_guard = state.transcriber.lock()
        .map_err(|e| format!("Failed to lock transcriber: {}", e))?;
    
    // Stop the transcriber
    transcriber_guard.stop();
    
    // Try to get any remaining transcription results
    let mut final_text = String::new();
    while let Some(result) = transcriber_guard.try_next_transcription() {
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

/// Show a window by label
fn show_window(app: &tauri::AppHandle, label: &str) -> Result<(), String> {
    info!("Showing window: {}", label);
    debug!("Attempting to get window with label: '{}'", label);
    
    // Get window by label
    let window = app.get_webview_window(label)
        .ok_or_else(|| {
            error!("Window not found: {}", label);
            format!("Window not found: {}", label)
        })?;
    
    debug!("Window '{}' found, calling show()", label);
    
    // Show the window
    window.show()
        .map_err(|e| {
            error!("Failed to show window {}: {}", label, e);
            format!("Failed to show window {}: {}", label, e)
        })?;
    
    info!("Window {} shown successfully", label);
    Ok(())
}

/// Hide a window by label
fn hide_window(app: &tauri::AppHandle, label: &str) -> Result<(), String> {
    info!("Hiding window: {}", label);
    debug!("Attempting to get window with label: '{}'", label);
    
    // Get window by label
    let window = app.get_webview_window(label)
        .ok_or_else(|| {
            error!("Window not found: {}", label);
            format!("Window not found: {}", label)
        })?;
    
    debug!("Window '{}' found, calling hide()", label);
    
    // Hide the window
    window.hide()
        .map_err(|e| {
            error!("Failed to hide window {}: {}", label, e);
            format!("Failed to hide window {}: {}", label, e)
        })?;
    
    info!("Window {} hidden successfully", label);
    Ok(())
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

    // Check microphone permissions
    debug!("Checking microphone permissions...");
    if !check_microphone_permission() {
        error!("Microphone permission denied - please grant microphone access in System Settings > Privacy & Security > Microphone");
        anyhow::bail!("Microphone permission denied");
    }
    debug!("Microphone permissions OK");

    // Get transcriber and start it
    debug!("Acquiring transcriber lock...");
    let mut transcriber = state.transcriber.lock()
        .map_err(|e| {
            error!("Failed to lock transcriber: {}", e);
            anyhow::anyhow!("Failed to lock transcriber: {}", e)
        })?;
    debug!("Transcriber lock acquired");

    // Start transcriber
    debug!("Starting transcriber...");
    transcriber.start()?;
    debug!("Transcriber started successfully");

    drop(transcriber);

    // Set recording flag
    debug!("Setting recording flag to true...");
    state.is_recording.store(true, Ordering::SeqCst);
    debug!("Recording flag set to true");

    // Show the transcription window directly
    debug!("Showing transcription window...");
    show_window(app, "transcription")
        .map_err(|e| {
            error!("Failed to show transcription window: {}", e);
            anyhow::anyhow!("Failed to show transcription window: {}", e)
        })?;
    debug!("Transcription window shown");
    
    // Also emit event for frontend to update state
    debug!("Emitting show-window event...");
    let _ = app.emit("show-window", ());
    debug!("Event emitted");
    
    // Spawn transcription polling thread
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
            
            // Stop recording
            match stop_recording(&state, app) {
                Ok(text) => {
                    info!("Recording stopped with text: {}", text);
                    
                    // Copy and paste the transcription
                    if let Err(e) = copy_and_paste(&text, app) {
                        error!("Failed to copy and paste: {}", e);
                    }
                    
                    // Wait 1 second before hiding window
                    thread::sleep(Duration::from_secs(1));
                    
                    // Hide the transcription window directly
                    if let Err(e) = hide_window(app, "transcription") {
                        error!("Failed to hide transcription window: {}", e);
                    }
                    
                    // Also emit event for frontend
                    let _ = app.emit("hide-window", ());
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
    start_recording(&state, &app).map_err(|e| e.to_string())
}

/// Manually stop recording (for UI control)
#[tauri::command]
fn manual_stop_recording(state: tauri::State<AppState>, app: tauri::AppHandle) -> Result<String, String> {
    info!("Manual stop recording triggered");
    stop_recording(&state, &app)
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

    // Load settings from config file
    eprintln!("Loading settings...");
    let settings = Settings::load().unwrap_or_else(|e| {
        eprintln!("Failed to load settings: {}. Using defaults.", e);
        Settings::default()
    });
    eprintln!("Settings loaded: model={}, language={:?}", settings.model, settings.language);
    
    // Initialize application state with transcriber
    eprintln!("Initializing application state and transcriber...");
    let app_state = AppState::with_settings(settings).unwrap_or_else(|e| {
        eprintln!("Failed to initialize transcriber: {}. Exiting.", e);
        std::process::exit(1);
    });
    eprintln!("Application state and transcriber initialized successfully");

    eprintln!("Building Tauri application...");
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_notification::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            greet,
            get_settings,
            update_settings,
            manual_start_recording,
            manual_stop_recording
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
                    .target(tauri_plugin_log::Target::new(
                        tauri_plugin_log::TargetKind::LogDir { file_name: Some("app".to_string()) }
                    ))
                    .build(),
            )?;

            info!("=== UNTERWHISPER SETUP: Logging initialized ===");
            debug!("Log directory: {:?}", log_dir);
            debug!("Log file: {:?}", log_file);

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
                .separator()
                .item(
                    &tauri::menu::MenuItemBuilder::with_id("quit", "Quit")
                        .build(app)?,
                )
                .build()?;
            
            debug!("Menu built successfully");
            debug!("Creating tray icon...");

            // Create system tray with icon
            let _tray = tauri::tray::TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&menu)
                .on_menu_event(|app, event| {
                    debug!("Tray menu event received: {:?}", event.id());
                    match event.id().as_ref() {
                        "start_recording" => {
                            debug!("Tray menu: 'start_recording' selected");
                            handle_start_recording(app);
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

            // Register global hotkey (Cmd+Shift+Space)
            info!("Registering global hotkey (Cmd+Shift+Space)...");
            
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
            
            let hotkey = HotKey::new(Some(Modifiers::SUPER | Modifiers::SHIFT), Code::Space);
            
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
            
            info!("Unterwhisper started - press Cmd+Shift+Space to start recording, or use the tray menu");
            info!("Tauri setup completed successfully");

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
