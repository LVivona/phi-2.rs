use std::{ops::{Index}, slice::SliceIndex};

use serde::{Deserialize, Serialize};

#[path="../tensor.rs"]
mod tensor;


#[derive(Debug, Serialize, Deserialize)]
pub struct Embedding<T> {
    num_embeddings : i32, 
    embedding_dim : i32,
    pub weight : tensor::Tensor<T>
}

impl<T> Embedding<T> {

    pub (crate)fn new(num_embeddings : i32, embedding_dim : i32) -> Self
        where T : Clone
    {
        assert!(num_embeddings > 0, "ValueError: num_embeddings must be larger then 0, try a i32 larger then 0.");
        assert!(embedding_dim > 0, "ValueError: embedding_dim must be larger then 0, try a i32 larger then 0.");
        

        Embedding {
            num_embeddings,
            embedding_dim,
            weight : tensor::Tensor::<T>::with_capacity((num_embeddings * embedding_dim) as usize)
        }
    }

    pub (crate)fn forward(&self, index : &[i32]) -> &[T] {
        todo!("NotImplemented: soon will be implemented later")
    }

    pub (crate)fn to_toml(&self) -> String
        where T : Serialize 
    {
        toml::to_string(&self).unwrap()
    }


    
}

impl<T, I : SliceIndex<[T]>> Index<I> for Embedding<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        assert!(self.weight.len() > 0, "IndexError: Nothing has been allocated");
        &self.weight[index]
    }
}


#[cfg(test)]
mod test {
    use super::Embedding;

    #[test]
    fn test_embedding_new(){
        let embedding: Embedding<f32> = Embedding::new(10, 10);
        assert_eq!(embedding.embedding_dim == 10, embedding.num_embeddings == 10);
    }

    #[test]
    fn test_forward(){
        println!("{:?}", Embedding::<f32>::new(10, 10).to_toml());
    }

    #[test]
    fn test_index(){
         let emb = Embedding::<f32>::new(10, 10);
         println!("{}", emb[0]);
    }
}
