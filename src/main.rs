#![allow(dead_code)]

use crate::nn::{embedding::Embedding, Linear};

mod config;
mod tokeizer;
mod nn;
mod tensor;






// phi-2 =============================
    // - config                     [x] 
    // - tokenizer                  [x]
    // - Linear Layers              []
        // - class                  [x]
        // - matrix multiplication  []
    // - Embedding
    // - Attention matrix (Q,K,V)   []
        // - MultiHeadAttention     []
    // - activation                 [x]
    // Extra                        []
        // - LoRA                   []
        // - Chat                   []
        // - inference training     []
        // - 16-bit floating points []
        // - new tensor datatype    []
// ====================================


// Stage - 1 do it with standard library


// Stage - 2 abstract so we could implement other architecture



fn main() ->  Result<(), ()>{
    let mut x: Embedding<f32> = nn::Embedding::new(1, 1);
    let mut l: Linear<f32> = nn::Linear::new(10, 10, false);

    l.forward(x)
    println!("{}", x.to_toml());
    println!("Hello world welcome to Phi-2...");
    Ok(())
}
