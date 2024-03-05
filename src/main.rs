#![allow(dead_code)]
use clap::Parser;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
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

#[derive(Debug)]
struct Tokenizer {
    vocab: HashMap<String, i32>,
}

impl Tokenizer {
    fn new(path: PathBuf) -> Tokenizer {
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);

        // Read the length (number of words)
        let mut length_bytes = [0u8; 4];
        reader.read_exact(&mut length_bytes).unwrap();
        let length = u32::from_be_bytes(length_bytes) as usize;

        let mut tokeizer = std::collections::HashMap::new();

        for pos in 0..length {
            let mut line = Vec::new();
            let bytes_read = reader.read_until(b'\n', &mut line).unwrap();
            if bytes_read == 0 {
                break; // End of file
            }
            tokeizer.insert(
                String::from_utf8_lossy(&line[..bytes_read - 1]).to_string(),
                pos as i32,
            );
        }

        Tokenizer { vocab: tokeizer }
    }

    fn encoder(self, txt: String) -> Vec<i32> {
        vec![1, 2, 3]
    }

    fn decoder(self, encoder: Vec<i32>) -> String {
        "hello world".to_string()
    }
}

#[derive(Parser, Debug)]
#[command(version="0.0.1", about=include_str!("./assets/about.txt"), long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short = 't', long = "tokenizer", value_name = "FILE")]
    tokenizer: PathBuf,

    #[arg(short = 'c', long = "config", value_name = "FILE")]
    config: Option<PathBuf>,

    #[arg(short = 'm', long = "model", value_name = "FILE")]
    model: Option<PathBuf>,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let tok = Tokenizer::new(args.tokenizer);

    println!("{:?}", tok);
    Ok(())
}
