use std::env;

use config::{Config, ConfigError, Environment, File}; 
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};
use std::fs;

lazy_static! {
    pub static ref FILSETTINGS: Settings = {
        let m = Settings::new().unwrap();
        m
    }; 
}

const SETTINGS_PATH: &str = "./fil-zk.config.toml";
#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub cpu_utilization: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            cpu_utilization: 0.2_f64,
        }
    }
}

impl Settings{
    pub fn new() -> Result<Settings, ConfigError> {
        let data = fs::read(SETTINGS_PATH);
        let result: Settings;
        match data {
            Ok(d) => {
                result = serde_json::de::from_slice(&d).unwrap();
            }
            Err(e) => {
                result = Self::default();
            }
        }
        Ok(result)
    }
}
