mod appearance;
mod chunker;
mod fsutil;
mod material;
mod genserver;
mod paths;
mod pigeons;
mod pigeonstore;
mod pyproc;
mod quickbuild;
mod recorder;
mod python_env;
mod runs;
mod table;

use tauri::{Emitter, Manager};

use genserver::GenServer;
use pyproc::RunningJob;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        // Rust-side only — see genserver.rs. capabilities/default.json still
        // grants the webview core:default and nothing else.
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_window_state::Builder::new().build())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        /* Pictures: pic://localhost/<run>/<path> — see table::serve_picture. */
        .register_asynchronous_uri_scheme_protocol("pic", |ctx, request, responder| {
            let app = ctx.app_handle().clone();
            tauri::async_runtime::spawn_blocking(move || responder.respond(table::serve_picture(&app, &request)));
        })
        .manage(RunningJob::default())
        .manage(GenServer::default())
        .manage(recorder::Recording::default())
        .manage(table::Named::default())
        .manage(pigeons::OpenedFiles::default())
        .manage(pigeonstore::LastMove(std::sync::Mutex::new(None)))
        /* Models live in Pigeon: move anything still in Flight School's old
           folders into Pigeon's store before a window can ask for a model. */
        .setup(|app| {
            use tauri::Manager;
            use tauri::menu::{MenuBuilder, MenuItemBuilder, SubmenuBuilder};
            /* The menu bar: the standard Mac items, plus New run and the
               three views under their ⌘ keys. Each choice is an event the
               page routes (app.js). */
            let new_run = MenuItemBuilder::with_id("new-run", "New Run").accelerator("CmdOrCtrl+N").build(app)?;
            let runs = MenuItemBuilder::with_id("view-runs", "Runs").accelerator("CmdOrCtrl+1").build(app)?;
            let pigeons = MenuItemBuilder::with_id("view-pigeons", "Pigeons").accelerator("CmdOrCtrl+2").build(app)?;
            let bench = MenuItemBuilder::with_id("view-bench", "Bench").accelerator("CmdOrCtrl+3").build(app)?;
            /* Quit is the app's own item, not the predefined one: the
               predefined Quit exits at once, past the question ⌘W asks
               while a run is teaching. This one asks the same question. */
            let settings = MenuItemBuilder::with_id("settings", "Settings…").accelerator("CmdOrCtrl+,").build(app)?;
            let quit = MenuItemBuilder::with_id("quit", "Quit Flight School").accelerator("CmdOrCtrl+Q").build(app)?;
            let app_menu = SubmenuBuilder::new(app, "Flight School").about(None).separator().item(&settings).separator().services().separator().hide().hide_others().show_all().separator().item(&quit).build()?;
            let file = SubmenuBuilder::new(app, "File").item(&new_run).separator().close_window().build()?;
            let edit = SubmenuBuilder::new(app, "Edit").undo().redo().separator().cut().copy().paste().select_all().build()?;
            let view = SubmenuBuilder::new(app, "View").item(&runs).item(&pigeons).item(&bench).build()?;
            let window = SubmenuBuilder::new(app, "Window").minimize().build()?;
            app.set_menu(MenuBuilder::new(app).items(&[&app_menu, &file, &edit, &view, &window]).build()?)?;
            app.on_menu_event(|app, ev| {
                let id = ev.id().as_ref().to_string();
                if id == "quit" {
                    let busy = app.state::<RunningJob>().0.try_lock().map(|slot| slot.is_some()).unwrap_or(true);
                    if busy {
                        if let Some(w) = app.get_webview_window("main") { let _ = w.emit("close-requested", ()); }
                    } else {
                        app.exit(0);
                    }
                    return;
                }
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.emit("menu", id);
                }
            });
            tauri::async_runtime::spawn(quickbuild::log_registry(app.handle().clone()));
            if pigeonstore::pigeon_present() {
                if let (Ok(fs), Ok(store)) = (paths::data_dir(app.handle()), pigeonstore::store()) {
                    let report = pigeonstore::migrate(&fs, &store);
                    if !report.moved.is_empty() || !report.problems.is_empty() || !report.kept_in_place.is_empty() {
                        eprintln!("[models] moved into Pigeon: {:?}", report);
                        if let Ok(mut g) = app.state::<pigeonstore::LastMove>().0.lock() { *g = Some(report); }
                    }
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            pigeonstore::models_home_status,
            python_env::python_env_status,
            python_env::setup_environment,
            python_env::stop_job,
            python_env::force_quit,
            genserver::gen_start,
            genserver::gen_stop,
            genserver::gen_chat,
            quickbuild::qb_options,
            quickbuild::qb_vision_ready,
            quickbuild::qb_setup_vision,
            quickbuild::qb_pigeon_look,
            quickbuild::qb_reveal,
            pigeons::pigeon_list,
            pigeons::pigeon_add_pick,
            pigeons::pigeon_remove,
            pigeons::pigeon_reveal,
            pigeons::pigeon_export_nest,
            pigeons::pigeon_take_opens,
            recorder::rec_permission,
            recorder::rec_request_permission,
            recorder::rec_open_privacy,
            recorder::rec_windows,
            recorder::rec_still,
            recorder::rec_start,
            recorder::rec_stop,
            recorder::rec_table,
            table::table_list,
            table::table_trash,
            table::table_duplicate,
            table::table_save,
            table::table_get,
            table::table_signatures,
            table::table_import_folder,
            table::table_pair_files,
            table::table_teach,
            table::table_ride,
            table::table_parts,
            table::table_send,
            table::table_base_pack,
            table::table_base_picture,
            material::table_material_add,
            material::table_material_remove,
            material::table_material_get,
            appearance::window_theme,
            table::table_look_kit,
            table::table_pick_icon,
            table::table_import_play,
        ])
        // THREE exit paths for the drafting sidecar, mirroring the earlier Airoplane
        // Mode app's engine.rs. Drop alone is not enough: a process that exits
        // without unwinding never runs it, and an orphaned llama-server
        // silently holding several gigabytes is the worst failure mode either
        // app has. Killing an already-dead child is a no-op, so belt and
        // braces costs nothing.
        .on_window_event(|window, event| match event {
            // Exit path 0 — the one that is NOT an exit. A training run is
            // twenty minutes (an hour for a photo job) and cannot be resumed
            // from where it stopped, so a reflex ⌘W must not silently throw it
            // away. Hold the close, ask the webview to ask the person, and let
            // them answer with `force_quit`.
            //
            // A contended lock counts as busy on purpose: asking a question
            // nobody needed is cheap, and losing a run is not.
            tauri::WindowEvent::CloseRequested { api, .. } => {
                let busy = window
                    .app_handle()
                    .state::<RunningJob>()
                    .0
                    .try_lock()
                    .map(|slot| slot.is_some())
                    .unwrap_or(true);
                if busy {
                    api.prevent_close();
                    let _ = window.emit("close-requested", ());
                }
            }
            // Exit path 1: the window went away.
            tauri::WindowEvent::Destroyed => {
                window.app_handle().state::<GenServer>().kill();
            }
            _ => {}
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        // Exit path 2: the app is quitting (Cmd+Q, quit menu).
        // Exit path 3 is GenServer's Drop impl — see genserver.rs.
        .run(|app_handle, event| {
            match event {
                tauri::RunEvent::Exit => app_handle.state::<GenServer>().kill(),
                // A double-clicked .pigeon or .nest (the file associations in
                // tauri.conf.json). macOS delivers it here whether the app was
                // already running or was launched by the click — on a cold
                // launch it lands before the webview has any listener, so
                // Rust does the add and parks the result (pigeons.rs).
                tauri::RunEvent::Opened { urls } => {
                    let paths: Vec<String> = urls.iter()
                        .filter_map(|u| u.to_file_path().ok())
                        .map(|p| p.to_string_lossy().into_owned())
                        .collect();
                    pigeons::handle_opened(app_handle, paths);
                }
                _ => {}
            }
        });
}
