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
    pub max_window_size: f64,
    pub work_size_multiplier: f64,
    pub chunk_size_multiplier: f64,
    pub size: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            cpu_utilization: 0.2_f64,
            max_window_size: 10_f64,
            work_size_multiplier: 2_f64,
            chunk_size_multiplier: 2_f64,
            size: 4,
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
    pub fn set_value(&mut self, i: usize, a: f64) {
        match i {
            0 => self.cpu_utilization = a,
            1 => self.max_window_size = a,
            2 => self.work_size_multiplier = a,
            3 => self.chunk_size_multiplier = a,
            _ => panic!("Settings have not this field")
        }
    }

}
