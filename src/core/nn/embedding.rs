use serde::{Deserialize, Serialize};
use std::{ops::Index, usize};

#[path = "../tensor/mod.rs"]
mod tensor;
use crate::tensor::Tensor;

#[derive(Debug, Serialize, Deserialize)]
pub struct Embedding<Dtype> {
    num_embeddings: usize,
    embedding_dim: usize,
    weight: Tensor<Dtype>,
}

///
impl<Dtype> Embedding<Dtype>
where
    Dtype: Copy + Default,
{
    ///
    pub(crate) fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        assert!(
            num_embeddings > 0,
            "ValueError: num_embeddings must be larger then 0, try a i32 larger then 0."
        );
        assert!(
            embedding_dim > 0,
            "ValueError: embedding_dim must be larger then 0, try a i32 larger then 0."
        );
        let weight = vec![Dtype::default(); embedding_dim * num_embeddings];

        Embedding {
            num_embeddings,
            embedding_dim,
            weight: weight,
        }
    }

    ///
    pub(crate) fn forward(&self, x: &[usize]) -> Vec<&Dtype> {
        x.iter()
            .flat_map(|&idx| {
                let start = idx * self.embedding_dim;
                let end = start + self.embedding_dim;
                &self.weight[start..end]
            })
            .collect::<Vec<&Dtype>>()
    }

    ///
    ///
    pub(crate) fn to_toml(&self) -> String
    where
        Dtype: Serialize,
    {
        toml::to_string(&self).unwrap()
    }
}

impl<Dtype> Index<usize> for Embedding<Dtype> {
    type Output = [Dtype];

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

    #[test]
    fn test_embedding_new() {
        let embedding: Embedding<f32> = Embedding::new(10, 10);
        assert!(embedding.weight.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_forward() {
        println!("{:?}", Embedding::<f32>::new(10, 10).to_toml());
    }

    #[test]
    fn test_index() {
        let mut emb = Embedding::<f32>::new(1, 1);
        emb.weight[0] = 1.0;
        let y = vec![1.0; 1];
        assert_eq!(y, emb[0]);
    }

    #[test]
    fn test_forward_pass() {
        let emb = Embedding::<f32>::new(1, 1);
        assert_eq!(
            emb.forward(&[0, 0])
                .iter()
                .map(|x| **x == 0.0)
                .collect::<Vec<bool>>(),
            vec![true, true]
        );
    }
}
