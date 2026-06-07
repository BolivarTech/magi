//! This module implements the Terminal User Interface using Ratatui.

use crate::agent::{Agent, ApprovalRequest};
use crate::system::secrets::SecretStore;
use crossterm::{
    event::{self, DisableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use magi_core::schema::Mode;
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame, Terminal,
};
use std::io;
use tokio::sync::mpsc;

/// Different interaction modes for the TUI.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum AppMode {
    Normal,
    Selection,
    Visual, // Mode for selecting text within a message
}

/// Parses a trimmed input line as a `/consult` command. `Some(query)` for
/// `/consult <query>` (empty string for bare `/consult`), `None` otherwise.
/// Requires a space boundary so `/consultation` is treated as normal input.
pub(crate) fn parse_consult_command(trimmed: &str) -> Option<&str> {
    let rest = trimmed.strip_prefix("/consult")?;
    if rest.is_empty() {
        return Some("");
    }
    Some(rest.strip_prefix(' ')?.trim())
}

/// Events that can happen in the UI.
pub enum UiEvent {
    Input(String),
    Clear,
    Login,
    Logout,
    /// Trigger a forced MAGI multi-perspective analysis with the given question.
    Consult(String),
    Quit,
}

/// Messages from the Agent to the UI.
pub enum AgentResponse {
    Text(String),
    Error(String),
    Info(String),
    /// An incremental text delta from the streaming provider.
    StreamDelta(String),
}

/// Represents the state of the TUI application.
pub struct App {
    /// The input string currently being typed.
    pub input: String,
    /// Current cursor position in the input string (byte index).
    pub cursor_position: usize,
    /// Selection start position (if any)
    pub selection_start: Option<usize>,
    /// History of messages to display.
    pub messages: Vec<String>,
    /// Channel to send events to the agent runner.
    pub event_tx: mpsc::Sender<UiEvent>,
    /// Channel to receive responses from the agent.
    pub response_rx: mpsc::Receiver<AgentResponse>,
    /// Channel to receive approval requests from the agent.
    pub approval_rx: mpsc::Receiver<ApprovalRequest>,
    /// Pending approval request
    pub pending_approval: Option<ApprovalRequest>,
    /// Current UI mode
    pub mode: AppMode,
    /// Index of the selected message in Selection mode
    pub selected_index: usize,
    /// Cursor position within the selected message (Visual mode)
    pub visual_cursor: usize,
    /// Selection start within the selected message (Visual mode)
    pub visual_selection_start: Option<usize>,
    /// Whether the agent is currently streaming a response.
    pub streaming: bool,
}

impl App {
    pub fn new(
        event_tx: mpsc::Sender<UiEvent>,
        response_rx: mpsc::Receiver<AgentResponse>,
        approval_rx: mpsc::Receiver<ApprovalRequest>,
    ) -> Self {
        Self {
            input: String::new(),
            cursor_position: 0,
            selection_start: None,
            messages: Vec::new(),
            event_tx,
            response_rx,
            approval_rx,
            pending_approval: None,
            mode: AppMode::Normal,
            selected_index: 0,
            visual_cursor: 0,
            visual_selection_start: None,
            streaming: false,
        }
    }

    /// Moves the cursor to the left, respecting Unicode character boundaries.
    pub fn move_cursor_left(&mut self, select: bool) {
        if select && self.selection_start.is_none() {
            self.selection_start = Some(self.cursor_position);
        } else if !select {
            self.selection_start = None;
        }

        if self.cursor_position > 0 {
            let indices = self.input.char_indices().rev();
            for (idx, _) in indices {
                if idx < self.cursor_position {
                    self.cursor_position = idx;
                    return;
                }
            }
            self.cursor_position = 0;
        }
    }

    /// Moves the cursor to the right, respecting Unicode character boundaries.
    pub fn move_cursor_right(&mut self, select: bool) {
        if select && self.selection_start.is_none() {
            self.selection_start = Some(self.cursor_position);
        } else if !select {
            self.selection_start = None;
        }

        if self.cursor_position < self.input.len() {
            let indices = self.input.char_indices();
            for (idx, _) in indices {
                if idx > self.cursor_position {
                    self.cursor_position = idx;
                    return;
                }
            }
            self.cursor_position = self.input.len();
        }
    }

    /// Inserts a character at the current cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.delete_selection();
        // Ensure cursor is at char boundary before insert
        if !self.input.is_char_boundary(self.cursor_position) {
            self.cursor_position = 0; // Emergency fallback
        }
        self.input.insert(self.cursor_position, c);
        self.cursor_position += c.len_utf8();
    }

    /// Deletes the character before the current cursor position.
    pub fn delete_char(&mut self) {
        if self.selection_start.is_some() {
            self.delete_selection();
            return;
        }

        if self.cursor_position > 0 {
            self.move_cursor_left(false);
            let prev_pos = self.cursor_position;
            if self.input.is_char_boundary(prev_pos) {
                self.input.remove(prev_pos);
            }
        }
    }

    /// Deletes the currently selected text.
    pub fn delete_selection(&mut self) {
        if let Some(start) = self.selection_start {
            let end = self.cursor_position;
            let (from, to) = if start < end {
                (start, end)
            } else {
                (end, start)
            };
            if self.input.is_char_boundary(from) && self.input.is_char_boundary(to) {
                self.input.drain(from..to);
                self.cursor_position = from;
            }
            self.selection_start = None;
        }
    }

    /// Returns the selected text if any.
    pub fn get_selected_text(&self) -> Option<String> {
        self.selection_start.and_then(|start| {
            let end = self.cursor_position;
            let (from, to) = if start < end {
                (start, end)
            } else {
                (end, start)
            };
            if self.input.is_char_boundary(from) && self.input.is_char_boundary(to) {
                Some(self.input[from..to].to_string())
            } else {
                None
            }
        })
    }

    /// Appends a message to the UI history.
    pub fn push_message(&mut self, message: String) {
        self.messages.push(message);
    }

    /// Appends a streaming delta to the in-progress assistant message,
    /// creating the line on the first delta. Append-only; never byte-indexes.
    pub fn append_stream_delta(&mut self, delta: String) {
        if self.streaming {
            if let Some(last) = self.messages.last_mut() {
                last.push_str(&delta);
                return;
            }
        }
        self.messages.push(format!("Magi Agent: {}", delta));
        self.streaming = true;
    }

    /// Marks the end of a streamed assistant turn.
    pub fn finalize_stream(&mut self) {
        self.streaming = false;
    }
}

pub async fn run_tui_ext(
    agent: Agent,
    startup_notices: Vec<String>,
    consult: Option<std::sync::Arc<magi_core::orchestrator::Magi>>,
) -> anyhow::Result<()> {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen, DisableMouseCapture);
        let _ =
            Terminal::new(CrosstermBackend::new(io::stdout())).and_then(|mut t| t.show_cursor());
        original_hook(panic_info);
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (event_tx, mut event_rx) = mpsc::channel(100);
    let (response_tx, response_rx) = mpsc::channel(100);
    let (approval_tx, approval_rx) = mpsc::channel(100);

    for notice in startup_notices {
        let _ = response_tx.send(AgentResponse::Info(notice)).await;
    }

    let mut runner_agent = agent;
    runner_agent.set_approval_channel(approval_tx);

    let mut consult_magi_runner = consult;

    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            match event {
                UiEvent::Input(text) => {
                    // Stream-bridge: `chunk_tx` is owned by `query_streaming`; when
                    // the method returns it is dropped, which closes the sender end of
                    // the channel. The forwarder task then drains any remaining deltas
                    // and exits its `recv()` loop naturally. `forwarder.await` joins
                    // the task before the end-of-turn marker is sent, guaranteeing
                    // all deltas arrive at the UI before `Text("")` (end-of-turn
                    // convention) or `Error(...)`.
                    let (chunk_tx, mut chunk_rx) = mpsc::channel::<String>(100);
                    let forward_tx = response_tx.clone();
                    let forwarder = tokio::spawn(async move {
                        while let Some(delta) = chunk_rx.recv().await {
                            if forward_tx
                                .send(AgentResponse::StreamDelta(delta))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                    });

                    let result = runner_agent.query_streaming(&text, chunk_tx).await;
                    // Join the forwarder: ensures all deltas are forwarded before the
                    // end-of-turn marker below is enqueued.
                    let _ = forwarder.await;

                    // `Text("")` signals end-of-turn to `run_app`; it calls
                    // `finalize_stream` instead of pushing an empty message line.
                    match result {
                        Ok(_) => {
                            let _ = response_tx.send(AgentResponse::Text(String::new())).await;
                        }
                        Err(e) => {
                            let _ = response_tx.send(AgentResponse::Error(e.to_string())).await;
                        }
                    }
                }
                UiEvent::Clear => {
                    runner_agent.clear_history();
                }
                UiEvent::Consult(query) => {
                    let magi = match consult_magi_runner.as_ref() {
                        Some(m) => m.clone(),
                        None => {
                            let _ = response_tx
                                .send(AgentResponse::Error(
                                    "consult requires a configured LLM provider — run /login or set a provider.".to_string(),
                                ))
                                .await;
                            continue;
                        }
                    };
                    // Cap forced /consult input too (the tool path caps in execute; this
                    // direct path bypasses it) — reject before any model call.
                    if query.len() > crate::tools::consult::MAX_QUERY_LEN {
                        let _ = response_tx
                            .send(AgentResponse::Error(format!(
                                "consult query too large ({} bytes; max {})",
                                query.len(),
                                crate::tools::consult::MAX_QUERY_LEN
                            )))
                            .await;
                        continue;
                    }
                    let _ = response_tx
                        .send(AgentResponse::Info(
                            "MAGI deliberating — 3 model calls…".to_string(),
                        ))
                        .await;
                    // MAGI FIX: joined spawn (awaited inline → serial, no finalize-order
                    // regression) isolates a panic in magi-core's analyze into a recoverable
                    // JoinError so the runner survives (see plan Task 6 iteration-3).
                    let join =
                        tokio::spawn(async move { magi.analyze(&Mode::Analysis, &query).await })
                            .await;
                    match join {
                        Ok(Ok(report)) => {
                            let body = if report.degraded {
                                format!(
                                    "[DEGRADED: fewer than 3 agents responded — consensus may be unreliable]\n\n{}",
                                    report.report
                                )
                            } else {
                                report.report
                            };
                            // Sanitize the verbatim report (LLM-generated) before rendering —
                            // strips ANSI escapes / control chars, matching the TextDelta path.
                            let body = crate::agent::Agent::sanitize_text(&body);
                            let _ = response_tx.send(AgentResponse::Text(body)).await;
                        }
                        Ok(Err(e)) => {
                            eprintln!("[consult] analyze failed: {e}");
                            let _ = response_tx
                                .send(AgentResponse::Error(
                                    "MAGI consult failed — check your provider/credentials and try again."
                                        .to_string(),
                                ))
                                .await;
                        }
                        Err(join_err) => {
                            eprintln!("[consult] analyze panicked: {join_err}");
                            let _ = response_tx
                                .send(AgentResponse::Error(
                                    "MAGI consult crashed unexpectedly; the session is still alive."
                                        .to_string(),
                                ))
                                .await;
                        }
                    }
                }
                UiEvent::Login => {
                    let oauth = crate::services::oauth::OAuthService::new();
                    let url = oauth.get_authorize_url();
                    let _ = response_tx.send(AgentResponse::Info(url)).await;

                    match oauth.start_callback_server().await {
                        Ok(code) => {
                            let _ = response_tx
                                .send(AgentResponse::Info("Authenticating...".to_string()))
                                .await;
                            match oauth.exchange_code_for_token(&code).await {
                                Ok(token) => match oauth.create_raw_api_key(&token).await {
                                    Ok(api_key) => {
                                        let store =
                                            crate::system::secrets::KeyringStore::new("magi-rs");
                                        if let Err(e) =
                                            store.set_secret("ANTHROPIC_API_KEY", &api_key).await
                                        {
                                            let _ = response_tx
                                                .send(AgentResponse::Error(format!(
                                                    "Failed to store key: {}",
                                                    e
                                                )))
                                                .await;
                                        } else {
                                            // #9: rebuild the running agent's provider in-session
                                            // so replies use the new key without a restart.
                                            let model = std::env::var("ANTHROPIC_MODEL")
                                                .unwrap_or_else(|_| {
                                                    crate::DEFAULT_MODEL.to_string()
                                                });
                                            // #16: only the canned StaticProvider history is safe to
                                            // clear; a re-login over a live provider must keep the
                                            // real conversation. Read before the swap, build banner
                                            // before `model` moves.
                                            let was_static = runner_agent.provider_is_static();
                                            let banner = if was_static {
                                                format!("Successfully logged in! Now using Magi API (model: {model}) — no restart needed; prior canned replies cleared.")
                                            } else {
                                                format!("Re-authenticated. Now using Magi API (model: {model}) — conversation kept.")
                                            };
                                            let provider_arc: std::sync::Arc<
                                                dyn crate::agent::provider::Provider,
                                            > = std::sync::Arc::new(
                                                crate::agent::provider::AnthropicProvider::new(
                                                    api_key,
                                                    model.clone(),
                                                ),
                                            );
                                            runner_agent.set_provider(provider_arc.clone());
                                            // I-5 + MAGI: rebuild the consult orchestrator over the new
                                            // provider so BOTH the forced /consult handle and the
                                            // registered auto-path tool use the new credentials
                                            // (register_or_replace adds it if it was absent, e.g. after
                                            // a static -> login transition).
                                            let new_magi = std::sync::Arc::new(magi_core::orchestrator::Magi::new(
                                                std::sync::Arc::new(crate::agent::magi_adapter::MagiCoreProviderAdapter::new(
                                                    provider_arc, "anthropic", model,
                                                )),
                                            ));
                                            runner_agent.register_or_replace_tool(Box::new(
                                                crate::tools::consult::ConsultTool::new(
                                                    new_magi.clone(),
                                                ),
                                            ));
                                            consult_magi_runner = Some(new_magi);
                                            if was_static {
                                                runner_agent.clear_history();
                                            }
                                            let _ =
                                                response_tx.send(AgentResponse::Info(banner)).await;
                                        }
                                    }
                                    Err(e) => {
                                        let _ = response_tx
                                            .send(AgentResponse::Error(format!(
                                                "Failed to create API key: {}",
                                                e
                                            )))
                                            .await;
                                    }
                                },
                                Err(e) => {
                                    let _ = response_tx
                                        .send(AgentResponse::Error(format!(
                                            "OAuth exchange failed: {}",
                                            e
                                        )))
                                        .await;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = response_tx
                                .send(AgentResponse::Error(format!(
                                    "Callback server error: {}",
                                    e
                                )))
                                .await;
                        }
                    }
                }
                UiEvent::Logout => {
                    // Clear from both canonical ("magi-rs") and legacy ("magi-rust") services
                    // so a key stored by either the new or the pre-migration login flow is removed.
                    // Mirrors the CLI --logout path in main.rs. delete_secret treats NoEntry as Ok.
                    let canonical = crate::system::secrets::KeyringStore::new("magi-rs");
                    let legacy = crate::system::secrets::KeyringStore::new("magi-rust");
                    let res_canonical = canonical.delete_secret("ANTHROPIC_API_KEY").await;
                    let res_legacy = legacy.delete_secret("ANTHROPIC_API_KEY").await;
                    match (res_canonical, res_legacy) {
                        (Err(e), _) | (_, Err(e)) => {
                            let _ = response_tx
                                .send(AgentResponse::Error(format!("Logout failed: {}", e)))
                                .await;
                        }
                        (Ok(()), Ok(())) => {
                            let _ = response_tx
                                .send(AgentResponse::Info("Logged out successfully.".to_string()))
                                .await;
                        }
                    }
                }
                UiEvent::Quit => break,
            }
        }
    });

    let app = App::new(event_tx, response_rx, approval_rx);
    let res = run_app(&mut terminal, app).await;

    let _ = disable_raw_mode();
    let _ = execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    );
    let _ = terminal.show_cursor();

    if let Err(err) = res {
        eprintln!("TUI Error: {:?}", err)
    }
    Ok(())
}

async fn run_app<B: Backend>(terminal: &mut Terminal<B>, mut app: App) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, &app))?;

        while let Ok(response) = app.response_rx.try_recv() {
            match response {
                AgentResponse::StreamDelta(delta) => app.append_stream_delta(delta),
                AgentResponse::Text(t) => {
                    if t.is_empty() {
                        app.finalize_stream();
                    } else {
                        app.push_message(format!("Magi Agent: {}", t));
                    }
                }
                AgentResponse::Error(e) => {
                    app.finalize_stream();
                    app.push_message(format!("Error: {}", e));
                }
                AgentResponse::Info(i) => {
                    app.finalize_stream();
                    app.push_message(format!("System: {}", i));
                }
            }
        }

        while let Ok(req) = app.approval_rx.try_recv() {
            app.push_message(format!("APPROVAL REQUIRED: Execute {}?", req.tool_name));
            app.push_message("Press 'y' to approve, 'c' or 'Esc' to deny.".to_string());
            app.pending_approval = Some(req);
        }

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match app.mode {
                    AppMode::Selection => {
                        match key.code {
                            KeyCode::Up if app.selected_index > 0 => {
                                app.selected_index -= 1;
                            }
                            KeyCode::Down
                                if app.selected_index < app.messages.len().saturating_sub(1) =>
                            {
                                app.selected_index += 1;
                            }
                            KeyCode::Enter => {
                                app.mode = AppMode::Visual;
                                app.visual_cursor = 0;
                                app.visual_selection_start = None;
                            }
                            KeyCode::Char('y') => {
                                if let Some(msg) = app.messages.get(app.selected_index) {
                                    if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                        let _ = clipboard.set_text(msg.clone());
                                        app.push_message("System: Message copied".to_string());
                                    }
                                }
                                app.mode = AppMode::Normal;
                            }
                            KeyCode::Esc | KeyCode::Char('q') => {
                                app.mode = AppMode::Normal;
                            }
                            _ => {}
                        }
                        continue;
                    }
                    AppMode::Visual => {
                        let msg = app
                            .messages
                            .get(app.selected_index)
                            .cloned()
                            .unwrap_or_default();
                        match key.code {
                            KeyCode::Left => {
                                if key.modifiers.contains(KeyModifiers::SHIFT)
                                    && app.visual_selection_start.is_none()
                                {
                                    app.visual_selection_start = Some(app.visual_cursor);
                                } else if !key.modifiers.contains(KeyModifiers::SHIFT) {
                                    app.visual_selection_start = None;
                                }
                                if app.visual_cursor > 0 {
                                    let indices = msg.char_indices().rev();
                                    for (idx, _) in indices {
                                        if idx < app.visual_cursor {
                                            app.visual_cursor = idx;
                                            break;
                                        }
                                    }
                                }
                            }
                            KeyCode::Right => {
                                if key.modifiers.contains(KeyModifiers::SHIFT)
                                    && app.visual_selection_start.is_none()
                                {
                                    app.visual_selection_start = Some(app.visual_cursor);
                                } else if !key.modifiers.contains(KeyModifiers::SHIFT) {
                                    app.visual_selection_start = None;
                                }
                                if app.visual_cursor < msg.len() {
                                    let indices = msg.char_indices();
                                    for (idx, _) in indices {
                                        if idx > app.visual_cursor {
                                            app.visual_cursor = idx;
                                            break;
                                        }
                                    }
                                }
                            }
                            KeyCode::Enter => {
                                if let (Some(msg_ref), Some(start)) = (
                                    app.messages.get(app.selected_index),
                                    app.visual_selection_start,
                                ) {
                                    let (from, to) = if start < app.visual_cursor {
                                        (start, app.visual_cursor)
                                    } else {
                                        (app.visual_cursor, start)
                                    };
                                    if msg_ref.is_char_boundary(from)
                                        && msg_ref.is_char_boundary(to)
                                    {
                                        if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                            let _ =
                                                clipboard.set_text(msg_ref[from..to].to_string());
                                            app.push_message("System: Fragment copied".to_string());
                                        }
                                    }
                                }
                                app.mode = AppMode::Normal;
                            }
                            KeyCode::Esc | KeyCode::Char('q') => {
                                app.mode = AppMode::Selection;
                            }
                            _ => {}
                        }
                        continue;
                    }
                    AppMode::Normal => {
                        match (key.code, key.modifiers) {
                            (KeyCode::Char('v'), KeyModifiers::CONTROL) => {
                                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                    if let Ok(text) = clipboard.get_text() {
                                        for c in text.chars() {
                                            app.insert_char(c);
                                        }
                                    }
                                }
                                continue;
                            }
                            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                if let Some(selected) = app.get_selected_text() {
                                    if let Ok(mut clipboard) = arboard::Clipboard::new() {
                                        let _ = clipboard.set_text(selected);
                                        app.push_message("System: Selection copied".to_string());
                                    }
                                    continue;
                                } else {
                                    let _ = app.event_tx.send(UiEvent::Quit).await;
                                    return Ok(());
                                }
                            }
                            (KeyCode::Char('s'), KeyModifiers::CONTROL) => {
                                if !app.messages.is_empty() {
                                    app.mode = AppMode::Selection;
                                    app.selected_index = app.messages.len().saturating_sub(1);
                                }
                                continue;
                            }
                            (KeyCode::Left, m) => {
                                app.move_cursor_left(m.contains(KeyModifiers::SHIFT));
                                continue;
                            }
                            (KeyCode::Right, m) => {
                                app.move_cursor_right(m.contains(KeyModifiers::SHIFT));
                                continue;
                            }
                            _ => {}
                        }

                        if let Some(req) = app.pending_approval.take() {
                            match key.code {
                                KeyCode::Char('y') | KeyCode::Char('Y') => {
                                    let _ = req.tx.send(true);
                                    app.push_message("User: Approved".to_string());
                                }
                                KeyCode::Char('c') | KeyCode::Char('C') | KeyCode::Esc => {
                                    let _ = req.tx.send(false);
                                    app.push_message("User: Denied".to_string());
                                }
                                _ => {
                                    app.pending_approval = Some(req);
                                }
                            }
                            continue;
                        }

                        match key.code {
                            KeyCode::Enter => {
                                let input = app.input.drain(..).collect::<String>();
                                app.cursor_position = 0;
                                let trimmed = input.trim();
                                if !trimmed.is_empty() {
                                    if let Some(query) = parse_consult_command(trimmed) {
                                        if query.is_empty() {
                                            app.push_message(
                                                "Usage: /consult <question> — forces MAGI multi-perspective analysis (3 model calls)"
                                                    .to_string(),
                                            );
                                        } else {
                                            app.push_message(format!("User: /consult {query}"));
                                            let _ = app
                                                .event_tx
                                                .send(UiEvent::Consult(query.to_string()))
                                                .await;
                                        }
                                        continue;
                                    }
                                    match trimmed {
                                        "/exit" | "/quit" => {
                                            let _ = app.event_tx.send(UiEvent::Quit).await;
                                            return Ok(());
                                        }
                                        "/clear" => {
                                            app.messages.clear();
                                            let _ = app.event_tx.send(UiEvent::Clear).await;
                                            continue;
                                        }
                                        "/login" => {
                                            let _ = app.event_tx.send(UiEvent::Login).await;
                                            continue;
                                        }
                                        "/logout" => {
                                            let _ = app.event_tx.send(UiEvent::Logout).await;
                                            continue;
                                        }
                                        "/help" => {
                                            app.push_message("Available commands:".to_string());
                                            app.push_message(
                                                "  /login, /logout - Identity management"
                                                    .to_string(),
                                            );
                                            app.push_message(
                                                "  /exit, /quit    - Exit the application"
                                                    .to_string(),
                                            );
                                            app.push_message(
                                                "  /clear          - Clear session history"
                                                    .to_string(),
                                            );
                                            app.push_message(
                                                "  /consult <q>    - Force MAGI multi-perspective analysis (3 model calls)"
                                                    .to_string(),
                                            );
                                            app.push_message(
                                                "  /help           - Show this help message"
                                                    .to_string(),
                                            );
                                            continue;
                                        }
                                        _ => {}
                                    }
                                    app.push_message(format!("User: {}", trimmed));
                                    let _ = app
                                        .event_tx
                                        .send(UiEvent::Input(trimmed.to_string()))
                                        .await;
                                }
                            }
                            KeyCode::Char(c) => {
                                app.insert_char(c);
                            }
                            KeyCode::Backspace => {
                                app.delete_char();
                            }
                            KeyCode::Esc => {
                                let _ = app.event_tx.send(UiEvent::Quit).await;
                                return Ok(());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}

/// Word-wraps `text` so each returned line fits in `width` CHARS (not bytes).
/// Existing `\n` are preserved as hard breaks. Words longer than `width`
/// are hard-split into chunks. `width == 0` is treated as no-op.
fn wrap_message(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut out: Vec<String> = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut line_w: usize = 0;
        for word in paragraph.split_whitespace() {
            let w = word.chars().count();
            if w > width {
                if !line.is_empty() {
                    out.push(std::mem::take(&mut line));
                    line_w = 0;
                }
                let chars: Vec<char> = word.chars().collect();
                for chunk in chars.chunks(width) {
                    out.push(chunk.iter().collect::<String>());
                }
                continue;
            }
            let need = if line.is_empty() { w } else { w + 1 };
            if line_w + need > width {
                out.push(std::mem::take(&mut line));
                line_w = 0;
            }
            if !line.is_empty() {
                line.push(' ');
                line_w += 1;
            }
            line.push_str(word);
            line_w += w;
        }
        if !line.is_empty() {
            out.push(line);
        }
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Selection index for the conversation `List` given the current UI mode.
///
/// In Selection / Visual mode the user-chosen index is used.  In Normal mode
/// the LAST message is selected so ratatui auto-scrolls the pane to keep the
/// newest message visible (follow-tail behavior). Empty history → `None` so
/// the list renders without an out-of-bounds selection.
fn effective_selection(mode: AppMode, selected_index: usize, messages_len: usize) -> Option<usize> {
    if messages_len == 0 {
        return None;
    }
    match mode {
        AppMode::Selection | AppMode::Visual => Some(selected_index),
        AppMode::Normal => Some(messages_len - 1),
    }
}

/// Highlight-symbol prefix for the conversation `List`.
///
/// Returns `">> "` only in Selection / Visual modes (where the user is
/// actively picking a message); Normal mode returns `""` so the auto-scroll
/// pin from `effective_selection` is invisible.
fn effective_highlight_symbol(mode: AppMode) -> &'static str {
    if matches!(mode, AppMode::Selection | AppMode::Visual) {
        ">> "
    } else {
        ""
    }
}

/// Returns the LAST `max` entries of `lines` if `lines.len() > max`, else returns `lines`
/// unchanged.  `max == 0` is treated as no-op (defensive: a viewport collapsed to height 0
/// during a resize must never silently drop data — the next non-zero frame restores everything).
fn tail_lines(lines: Vec<String>, max: usize) -> Vec<String> {
    if max == 0 || lines.len() <= max {
        return lines;
    }
    let skip = lines.len() - max;
    lines.into_iter().skip(skip).collect()
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Percentage(80), Constraint::Length(3)].as_ref())
        .split(f.size());

    let inner_width = chunks[0].width.saturating_sub(2) as usize; // subtract left + right borders
    let inner_height = chunks[0].height.saturating_sub(2) as usize; // subtract top + bottom borders

    let last_idx = app.messages.len().saturating_sub(1);
    let messages: Vec<ListItem> = app
        .messages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mut style = Style::default();
            if (app.mode == AppMode::Selection || app.mode == AppMode::Visual)
                && i == app.selected_index
            {
                style = style
                    .bg(Color::Blue)
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD);
            }
            let wrapped = wrap_message(m, inner_width);
            // Only the LAST message in Normal mode gets tail-truncated — this is the streaming
            // target that can grow taller than the viewport.  Selection / Visual modes show
            // every wrapped line so the user can review the full message (Ctrl+S → ↑ navigation).
            let displayed = if i == last_idx && app.mode == AppMode::Normal {
                tail_lines(wrapped, inner_height)
            } else {
                wrapped
            };
            let lines: Vec<Line> = displayed.into_iter().map(Line::from).collect();
            ListItem::new(Text::from(lines)).style(style)
        })
        .collect();

    let mut state = ListState::default();
    if let Some(idx) = effective_selection(app.mode, app.selected_index, app.messages.len()) {
        state.select(Some(idx));
    }

    let messages_list = List::new(messages)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Conversation History"),
        )
        .highlight_symbol(effective_highlight_symbol(app.mode));
    f.render_stateful_widget(messages_list, chunks[0], &mut state);

    let mut input_text = Text::raw(app.input.as_str());
    if let Some(start) = app.selection_start {
        let (from, to) = if start < app.cursor_position {
            (start, app.cursor_position)
        } else {
            (app.cursor_position, start)
        };
        if app.input.is_char_boundary(from) && app.input.is_char_boundary(to) {
            let spans = vec![
                Span::raw(&app.input[..from]),
                Span::styled(
                    &app.input[from..to],
                    Style::default().bg(Color::White).fg(Color::Black),
                ),
                Span::raw(&app.input[to..]),
            ];
            input_text = Text::from(Line::from(spans));
        }
    }

    let input_title = match app.mode {
        AppMode::Selection => {
            "SELECT MESSAGE (Enter to select text, 'y' to copy whole, Esc to exit)"
        }
        AppMode::Visual => "VISUAL SELECTION MODE",
        _ if app.pending_approval.is_some() => "WAITING FOR APPROVAL (y/c)",
        _ => "Input (Ctrl+S Copy Mode, Shift+Arrows Select)",
    };

    let input =
        Paragraph::new(input_text).block(Block::default().borders(Borders::ALL).title(input_title));
    f.render_widget(input, chunks[1]);

    if app.mode == AppMode::Normal {
        // Find visible width of input to position cursor correctly
        let prefix_len = app.input[..app.cursor_position].chars().count() as u16;
        f.set_cursor(chunks[1].x + prefix_len + 1, chunks[1].y + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_app_cursor_logic() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        app.insert_char('a');
        app.insert_char('c');
        assert_eq!(app.input, "ac");
        assert_eq!(app.cursor_position, 2);

        app.move_cursor_left(false);
        app.insert_char('b');
        assert_eq!(app.input, "abc");
        assert_eq!(app.cursor_position, 2);

        app.delete_char();
        assert_eq!(app.input, "ac");
        assert_eq!(app.cursor_position, 1);
    }

    #[tokio::test]
    async fn test_unicode_character_boundary_panic() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        app.insert_char('á');
        assert_eq!(app.cursor_position, 2);

        app.move_cursor_left(false);
        assert_eq!(app.cursor_position, 0);

        app.insert_char('x');
        assert_eq!(app.input, "xá");
    }

    #[test]
    fn test_wrap_message_normal_word_wrap() {
        let out = wrap_message("the quick brown fox jumps over the lazy dog", 12);
        // No line should exceed the width.
        for line in &out {
            assert!(line.chars().count() <= 12, "line {line:?} > 12");
        }
        // Joining with single spaces reconstructs the original text.
        assert_eq!(out.join(" "), "the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn test_wrap_message_preserves_embedded_newlines() {
        // Existing \n in the text becomes a hard line break — wrap each paragraph independently.
        let out = wrap_message("hello world\n\nsecond paragraph here", 20);
        // The blank line between paragraphs is preserved as an empty entry.
        assert!(
            out.iter().any(|l| l.is_empty()),
            "expected an empty line for the blank paragraph: {out:?}"
        );
        assert!(out.iter().any(|l| l == "hello world"));
        assert!(out.iter().any(|l| l.contains("second paragraph")));
    }

    #[test]
    fn test_wrap_message_breaks_oversized_word() {
        // A single word longer than width must be split into chunks of <= width chars each,
        // not infinite-loop and not exceed width.
        let out = wrap_message("supercalifragilisticexpialidocious", 5);
        for line in &out {
            assert!(line.chars().count() <= 5, "line {line:?} > 5");
        }
        assert!(!out.is_empty());
        // The chunks, concatenated, must equal the original word.
        assert_eq!(out.join(""), "supercalifragilisticexpialidocious");
    }

    #[test]
    fn test_wrap_message_handles_multibyte_utf8() {
        // Spanish accents: chars().count() not byte length. Width is measured in CHARS, not bytes.
        let out = wrap_message("La capital de Venezuela es Caracas — está al norte", 18);
        for line in &out {
            assert!(line.chars().count() <= 18, "line {line:?} > 18 chars");
        }
    }

    #[test]
    fn test_wrap_message_width_zero_yields_at_least_one_line() {
        // Defensive: width 0 must not panic / loop. A single line containing the original text is acceptable.
        let out = wrap_message("anything", 0);
        assert_eq!(out, vec!["anything".to_string()]);
    }

    #[test]
    fn test_wrap_message_empty_input() {
        let out = wrap_message("", 80);
        // An empty input should produce one empty line so the message still renders as a row.
        assert_eq!(out, vec!["".to_string()]);
    }

    #[test]
    fn test_effective_selection_normal_mode_follows_tail() {
        // Normal mode auto-pins the last message so the list auto-scrolls to bottom.
        assert_eq!(effective_selection(AppMode::Normal, 0, 5), Some(4));
        assert_eq!(effective_selection(AppMode::Normal, 99, 5), Some(4)); // ignores stale idx
    }

    #[test]
    fn test_effective_selection_selection_and_visual_use_index() {
        // Selection / Visual modes use the user's chosen index verbatim.
        assert_eq!(effective_selection(AppMode::Selection, 2, 5), Some(2));
        assert_eq!(effective_selection(AppMode::Visual, 0, 5), Some(0));
        assert_eq!(effective_selection(AppMode::Visual, 4, 5), Some(4));
    }

    #[test]
    fn test_effective_selection_empty_messages_yields_none() {
        // No messages → no selection (avoid out-of-bounds; the List renders nothing).
        assert_eq!(effective_selection(AppMode::Normal, 0, 0), None);
        assert_eq!(effective_selection(AppMode::Selection, 0, 0), None);
        assert_eq!(effective_selection(AppMode::Visual, 0, 0), None);
    }

    #[test]
    fn test_effective_highlight_symbol_by_mode() {
        // Selection / Visual show the ">> " marker; Normal hides it.
        assert_eq!(effective_highlight_symbol(AppMode::Selection), ">> ");
        assert_eq!(effective_highlight_symbol(AppMode::Visual), ">> ");
        assert_eq!(effective_highlight_symbol(AppMode::Normal), "");
    }

    #[test]
    fn test_tail_lines_keeps_last_n_when_input_exceeds_max() {
        let input: Vec<String> = (0..10).map(|i| format!("line {i}")).collect();
        let out = tail_lines(input, 3);
        assert_eq!(
            out,
            vec![
                "line 7".to_string(),
                "line 8".to_string(),
                "line 9".to_string()
            ]
        );
    }

    #[test]
    fn test_tail_lines_returns_input_unchanged_when_max_ge_len() {
        let input: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        assert_eq!(tail_lines(input.clone(), 3), input); // equal
        assert_eq!(tail_lines(input.clone(), 100), input); // greater
    }

    #[test]
    fn test_tail_lines_max_zero_returns_input_unchanged() {
        // Defensive: max=0 (degenerate viewport) is a no-op — never lose data on resize-to-zero.
        let input: Vec<String> = vec!["a".into(), "b".into()];
        assert_eq!(tail_lines(input.clone(), 0), input);
    }

    #[test]
    fn test_tail_lines_empty_input() {
        let out = tail_lines(Vec::<String>::new(), 5);
        assert!(out.is_empty());
    }

    #[test]
    fn test_tail_lines_shifts_by_one_when_input_grows_by_one() {
        // Simulates a streaming tick: the tail visually scrolls up by one line.
        let before: Vec<String> = (0..20).map(|i| format!("L{i}")).collect();
        let mut after = before.clone();
        after.push("L20".into());
        let max = 10;
        let tail_before = tail_lines(before, max);
        let tail_after = tail_lines(after, max);
        // Tail moves forward by 1: L10..=L19  →  L11..=L20.
        assert_eq!(tail_before.first().unwrap(), "L10");
        assert_eq!(tail_before.last().unwrap(), "L19");
        assert_eq!(tail_after.first().unwrap(), "L11");
        assert_eq!(tail_after.last().unwrap(), "L20");
    }

    #[test]
    fn test_parse_consult_command() {
        assert_eq!(
            super::parse_consult_command("/consult should we X?"),
            Some("should we X?")
        );
        assert_eq!(super::parse_consult_command("/consult"), Some(""));
        assert_eq!(super::parse_consult_command("hello"), None);
        assert_eq!(super::parse_consult_command("/consultation"), None);
    }

    #[test]
    fn test_full_report_renders_without_panic() {
        let (event_tx, _) = mpsc::channel(1);
        let (_, response_rx) = mpsc::channel(1);
        let (_, approval_rx) = mpsc::channel(1);
        let mut app = App::new(event_tx, response_rx, approval_rx);

        let report = format!(
            "+{}+\n|  MAGI VERDICT  |\n+{}+\nMelchior: APPROVE — café ☕ {}\n",
            "=".repeat(50),
            "=".repeat(50),
            "x".repeat(500)
        );
        // Push each line of the report as a separate message (simulates how
        // AgentResponse::Text is rendered line-by-line in run_app).
        for line in report.lines() {
            app.push_message(line.to_string());
        }
        // Must not panic; assert that the VERDICT line is present.
        assert!(
            app.messages.iter().any(|m| m.contains("MAGI VERDICT")),
            "expected a message containing 'MAGI VERDICT', got: {:?}",
            app.messages
        );
    }
}
