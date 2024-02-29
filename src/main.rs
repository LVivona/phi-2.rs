#![allow(dead_code)]
use clap::Parser;
use std::path::PathBuf;
// simple transformer config struct
struct Config {
    name: String,
}

impl Config {
    fn build() -> Config {
        Config {
            name: "temp".to_string(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long, value_name = "FILE")]
    tokenizer: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE")]
    model: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    println!("{:?}", args);
}
