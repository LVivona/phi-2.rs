#![allow(dead_code)]

use crate::nn::{Embedding, Linear};

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
    // TODO: fix this to have asserts to check we don't go over space provided
    let mut x: Embedding<f32> = Embedding::<f32>::new(3, 1);
    let mut _l: Linear<f32> = Linear::new(10, 10, false);

    x.weight.push(1.0);
    x.weight.push(2.0);
    x.weight.push(3.0);
    // println!("{:?}", &x[1]);
    println!("{}", x.to_toml());
    println!("Hello world welcome to Phi-2...");
    Ok(())
}
