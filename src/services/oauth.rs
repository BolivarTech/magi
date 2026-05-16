//! This module implements the OAuth 2.0 flow with PKCE for Anthropic Console.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use rand::{RngCore, thread_rng};
use tokio::sync::{oneshot, Mutex};
use axum::{
    extract::{Query, State},
    response::Redirect,
    routing::get,
    Router,
};
use std::sync::Arc;
use urlencoding::encode;

pub const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
pub const REDIRECT_PORT: u16 = 54545;
pub const AUTHORIZE_URL: &str = "https://console.anthropic.com/oauth/authorize";
pub const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
pub const API_KEY_URL: &str = "https://api.anthropic.com/api/oauth/claude_cli/create_api_key";

/// PKCE helper to generate code verifier and challenge.
pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
}

impl Pkce {
    pub fn generate() -> Self {
        let mut verifier_bytes = [0u8; 32];
        thread_rng().fill_bytes(&mut verifier_bytes);
        let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);
        
        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        let challenge = URL_SAFE_NO_PAD.encode(hasher.finalize());
        
        Self { verifier, challenge }
    }
}

#[derive(Deserialize)]
struct CallbackParams {
    code: String,
    state: String,
}

struct AppState {
    expected_state: String,
    tx: Mutex<Option<oneshot::Sender<String>>>,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
}

pub struct OAuthService {
    pkce: Pkce,
    state: String,
    base_url: String, // Allow overriding for tests
    token_url: String,
    api_key_url: String,
}

impl OAuthService {
    pub fn new() -> Self {
        let mut state_bytes = [0u8; 32];
        thread_rng().fill_bytes(&mut state_bytes);
        let state = URL_SAFE_NO_PAD.encode(state_bytes);
        
        Self {
            pkce: Pkce::generate(),
            state,
            base_url: AUTHORIZE_URL.to_string(),
            token_url: TOKEN_URL.to_string(),
            api_key_url: API_KEY_URL.to_string(),
        }
    }

    #[cfg(test)]
    pub fn with_urls(token_url: String, api_key_url: String) -> Self {
        let mut s = Self::new();
        s.token_url = token_url;
        s.api_key_url = api_key_url;
        s
    }

    pub fn get_authorize_url(&self) -> String {
        let redirect_uri = format!("http://localhost:{}/callback", REDIRECT_PORT);
        let scope = "org:create_api_key user:profile";
        format!(
            "{}?client_id={}&response_type=code&redirect_uri={}&scope={}&code_challenge={}&code_challenge_method=S256&state={}",
            self.base_url,
            CLIENT_ID,
            encode(&redirect_uri),
            encode(scope),
            self.pkce.challenge,
            self.state
        )
    }

    pub async fn exchange_code_for_token(&self, code: &str) -> Result<String> {
        let client = reqwest::Client::new();
        let redirect_uri = format!("http://localhost:{}/callback", REDIRECT_PORT);
        
        let response = client.post(&self.token_url)
            .json(&serde_json::json!({
                "grant_type": "authorization_code",
                "code": code,
                "client_id": CLIENT_ID,
                "code_verifier": self.pkce.verifier,
                "redirect_uri": redirect_uri,
                "state": self.state,
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Token exchange failed: {}", response.text().await?));
        }

        let token_res: TokenResponse = response.json().await?;
        Ok(token_res.access_token)
    }

    pub async fn create_raw_api_key(&self, access_token: &str) -> Result<String> {
        let client = reqwest::Client::new();
        let response = client.post(&self.api_key_url)
            .header("Authorization", format!("Bearer {}", access_token))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to create API key: {}", response.text().await?));
        }

        #[derive(Deserialize)]
        struct ApiKeyResponse {
            raw_key: String,
        }

        let res: ApiKeyResponse = response.json().await?;
        Ok(res.raw_key)
    }

    pub async fn start_callback_server(&self) -> Result<String> {
        let (tx, rx) = oneshot::channel();
        let app_state = Arc::new(AppState {
            expected_state: self.state.clone(),
            tx: Mutex::new(Some(tx)),
        });

        let app = Router::new()
            .route("/callback", get(callback_handler))
            .with_state(app_state);

        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", REDIRECT_PORT)).await?;
        let server = axum::serve(listener, app);

        tokio::select! {
            _ = server => Err(anyhow::anyhow!("Server closed prematurely")),
            code = rx => Ok(code?),
        }
    }
}

async fn callback_handler(
    Query(params): Query<CallbackParams>,
    State(state): State<Arc<AppState>>,
) -> Redirect {
    if params.state == state.expected_state {
        let mut lock = state.tx.lock().await;
        if let Some(tx) = lock.take() {
            let _ = tx.send(params.code);
        }
    }
    Redirect::to("https://console.anthropic.com/buy_credits?returnUrl=/oauth/code/success")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;

    #[test]
    fn test_pkce_generation() {
        let pkce = Pkce::generate();
        assert!(!pkce.verifier.is_empty());
        assert!(!pkce.challenge.is_empty());
        assert_ne!(pkce.verifier, pkce.challenge);
    }

    #[test]
    fn test_authorize_url_generation() {
        let service = OAuthService::new();
        let url = service.get_authorize_url();
        
        assert!(url.contains(AUTHORIZE_URL));
        assert!(url.contains(CLIENT_ID));
        assert!(url.contains(&service.state));
        assert!(url.contains(&service.pkce.challenge));
        
        // Rigorous check for URL encoding of redirect_uri
        assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A54545%2Fcallback"), 
            "redirect_uri must be properly URL-encoded");
            
        // Check for encoded scope (space replaced by %20 or +)
        assert!(url.contains("scope=org%3Acreate_api_key%20user%3Aprofile"),
            "scope must be properly URL-encoded");
    }

    #[tokio::test]
    async fn test_token_exchange() {
        let mut server = mockito::Server::new_async().await;
        let url = server.url();

        let service = OAuthService::with_urls(format!("{}/token", url), "".to_string());
        let expected_state = service.state.clone();

        let _m = server.mock("POST", "/token")
            .match_body(mockito::Matcher::Json(serde_json::json!({
                "grant_type": "authorization_code",
                "code": "mock_code",
                "client_id": CLIENT_ID,
                "code_verifier": service.pkce.verifier,
                "redirect_uri": format!("http://localhost:{}/callback", REDIRECT_PORT),
                "state": expected_state, // NEW REQUIREMENT
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({
                "access_token": "mock_access_token"
            }).to_string())
            .create_async().await;

        let token = service.exchange_code_for_token("mock_code").await.unwrap();

        assert_eq!(token, "mock_access_token");
    }

    #[tokio::test]
    async fn test_api_key_creation() {
        let mut server = mockito::Server::new_async().await;
        let url = server.url();

        let _m = server.mock("POST", "/create_key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({
                "raw_key": "sk-ant-real-key"
            }).to_string())
            .create_async().await;

        let service = OAuthService::with_urls("".to_string(), format!("{}/create_key", url));
        let key = service.create_raw_api_key("mock_token").await.unwrap();

        assert_eq!(key, "sk-ant-real-key");
    }

    #[tokio::test]
    async fn test_callback_handler_csrf_protection() {
        let (tx, rx) = oneshot::channel();
        let app_state = Arc::new(AppState {
            expected_state: "secure_state_123".to_string(),
            tx: Mutex::new(Some(tx)),
        });

        // Simulate a request with INCORRECT state
        let params = CallbackParams {
            code: "malicious_code".to_string(),
            state: "wrong_state".to_string(),
        };

        let _res = callback_handler(Query(params), State(app_state.clone())).await;

        // Verify that the code was NOT sent through the channel
        let result = rx.now_or_never();
        assert!(result.is_none(), "Code should NOT be sent if state is incorrect");
    }

    #[tokio::test]
    async fn test_callback_handler_success() {
        let (tx, rx) = oneshot::channel();
        let app_state = Arc::new(AppState {
            expected_state: "secure_state_123".to_string(),
            tx: Mutex::new(Some(tx)),
        });

        // Simulate a CORRECT request
        let params = CallbackParams {
            code: "valid_code".to_string(),
            state: "secure_state_123".to_string(),
        };

        let _res = callback_handler(Query(params), State(app_state)).await;

        // Verify that the code WAS sent
        let received_code = rx.await.unwrap();
        assert_eq!(received_code, "valid_code");
    }
}
