//! The window's own appearance, to match the theme the person chose.
//!
//! Pigeon's appearance.rs, for the same reason: the page draws itself dark or
//! light from `data-theme`, but the traffic lights, the file pickers and the
//! window behind the page are AppKit's, drawn in the WINDOW's appearance — the
//! Mac's, unless told otherwise. In Pigeon that left a black title on a dark
//! bar. So the window is given the same choice the page has (Settings →
//! Appearance), and Match Mac hands it back.

use tauri::{Manager, Runtime, Theme, WebviewWindow};

/// Light and dark paper, as tokens.css has them, for the frames before the
/// page paints (and under it while a window is resized).
const PAPER_DARK: tauri::window::Color = tauri::window::Color(0x0D, 0x17, 0x20, 0xFF);
const PAPER_LIGHT: tauri::window::Color = tauri::window::Color(0xE5, 0xF1, 0xF7, 0xFF);

/// `theme` is the saved setting: "dark", "light", or anything else for Match Mac.
pub fn apply<R: Runtime>(w: &WebviewWindow<R>, theme: Option<&str>) {
    let forced = match theme {
        Some("dark") => Some(Theme::Dark),
        Some("light") => Some(Theme::Light),
        _ => None,
    };
    let _ = w.set_theme(forced);
    let dark = match forced {
        Some(t) => t == Theme::Dark,
        None => matches!(w.theme(), Ok(Theme::Dark)),
    };
    let _ = w.set_background_color(Some(if dark { PAPER_DARK } else { PAPER_LIGHT }));
}

/// Settings → Appearance changed. Takes only the three words Settings offers
/// and acts on the main window alone; the page gets nothing back.
#[tauri::command]
pub fn window_theme<R: Runtime>(app: tauri::AppHandle<R>, theme: String) -> Result<(), String> {
    if !matches!(theme.as_str(), "dark" | "light" | "system") {
        return Err("the theme is dark, light or system".into());
    }
    let w = app.get_webview_window("main").ok_or("the main window is not open")?;
    apply(&w, Some(&theme));
    Ok(())
}
