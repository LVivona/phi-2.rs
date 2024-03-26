
use crate::nn;
mod linear;

// NOTE is it better to implement this as one long vector or 
// small vector 
struct MultiHeadAttention<T>{
    k : nn::linear::Linear,
    q : nn::linear::Linear,
    v : nn::linear::Linear,
}

impl<T> MultiHeadAttention<T>{

    fn new(embed_dim: i32, num_heads: i32, dropout: Option<T>) -> Self 
        where T: Clone
    {
        todo!("NotImplemented: soon will be implemented later")
    }
}






