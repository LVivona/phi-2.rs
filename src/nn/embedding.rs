use rand::prelude::*;
use rand::distributions::{Uniform, Distribution};
use std::{ops::Index, usize};
use serde::{Deserialize, Serialize};


#[path="../tensor.rs"]
mod tensor;
use crate::tensor::Tensor;

#[derive(Debug, Serialize, Deserialize)]
pub struct Embedding<T> {
    num_embeddings : usize, 
    embedding_dim : usize,
    pub weight : Tensor<T>
}


impl<T> Embedding<T> 
    where
    T: Copy
{

    pub (crate)fn new(num_embeddings : usize, embedding_dim : usize) -> Self
    {
        assert!(
            num_embeddings > 0, 
            "ValueError: num_embeddings must be larger then 0, try a i32 larger then 0."
        );
        assert!(
                embedding_dim > 0,
                "ValueError: embedding_dim must be larger then 0, try a i32 larger then 0."
            );

        Embedding {
            num_embeddings,
            embedding_dim,
            weight : Vec::<T>::with_capacity(embedding_dim * num_embeddings)
        }
    }

    pub (crate)fn forward(&self, x : &[usize]) -> Vec<&T>
    {
        x.iter().flat_map(|&idx| {
            let start = idx * self.embedding_dim;
            let end = start + self.embedding_dim;
            &self.weight[start..end]
        }).collect::<Vec<&T>>()
    }


    pub (crate)fn to_toml(&self) -> String
        where T : Serialize 
    {
        toml::to_string(&self).unwrap()
    }


}

impl<T> Index<usize> for Embedding<T> 
{
    type Output = [T];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.embedding_dim;
        let end = start + self.embedding_dim;
        &self.weight[start..end]
    }
}

#[cfg(test)]
mod test {
    use super::Embedding;
    use rand::prelude::*;
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
         let mut emb = Embedding::<f32>::new(1, 1);
         emb.weight.push(1.0);
         assert_eq!([1.0], emb[0]);
    }

    #[test]
    fn test_forward_pass(){
         let mut emb = Embedding::<f32>::new(1, 1);
         emb.weight.push(1.0);

         assert_eq!(emb.forward(&[0, 0]).iter().map(|x| **x == 1.0).collect::<Vec<bool>>(), vec![true, true]);
    }   

    #[test]
    fn random_test(){

        let mut rng = thread_rng();
        if rng.gen() { // random bool
            let x: f64 = rng.gen(); // random number in range [0, 1)
            let y = rng.gen_range(-10.0..10.0);
            println!("x is: {}", x);
            println!("y is: {}", y);
        }
    }
}
