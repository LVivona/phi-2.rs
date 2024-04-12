

mod linear;
use linear::Linear;

// NOTE is it better to implement this as one long vector or 
// small vector 
struct MultiHeadAttention<T>{
    k : Linear,
    q : Linear,
    v : Linear,
}

impl<T> MultiHeadAttention<T>{

    fn new(embed_dim: i32, num_heads: i32, dropout: Option<f32>) -> Self 
        where T: Clone
    {
        todo!("NotImplemented: soon will be implemented later")
    }
}






