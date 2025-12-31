use global_hotkey::{GlobalHotKeyManager, hotkey::{Code, HotKey, Modifiers}};
use log::info;
use tauri::Manager;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

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

/// Handle the "Start Recording" menu item
fn handle_start_recording(app: &tauri::AppHandle) {
    info!("Start Recording triggered from menu");
    
    // Get the app state
    let state = app.state::<AppState>();
    
    // Check if already recording
    if state.is_recording.load(Ordering::SeqCst) {
        log::warn!("Already recording, ignoring start request");
        return;
    }
    
    // TODO: Implement actual recording start in task 5.1
    info!("Recording start requested (implementation pending)");
}

/// Handle the "Quit" menu item
fn handle_quit(app: &tauri::AppHandle) {
    info!("Quit triggered from menu");
    
    // Get the app state
    let state = app.state::<AppState>();
    
    // Stop recording if active
    if state.is_recording.load(Ordering::SeqCst) {
        log::info!("Stopping active recording before quit");
        // TODO: Implement proper cleanup in task 5.2
        state.is_recording.store(false, Ordering::SeqCst);
    }
    
    // Exit the application
    app.exit(0);
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
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
        .invoke_handler(tauri::generate_handler![greet])
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
            let hotkey_manager = GlobalHotKeyManager::new().map_err(|e| {
                log::error!("Failed to create hotkey manager: {}", e);
                anyhow::anyhow!("Failed to create hotkey manager: {}", e)
            })?;
            
            let hotkey = HotKey::new(Some(Modifiers::SUPER | Modifiers::SHIFT), Code::Space);
            hotkey_manager.register(hotkey).map_err(|e| {
                log::error!("Failed to register hotkey: {}", e);
                anyhow::anyhow!("Failed to register hotkey: {}", e)
            })?;

            info!("Global hotkey registered successfully");
            info!("Tauri setup completed successfully");

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
