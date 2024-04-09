#![allow(unreachable_code, dead_code, unused_imports)]
// Simplest case, using 1D vec reproduce a matrix
// later can expand on our own datatype that utilize
// low level hardware
// pub type Tensor<T> = Vec<T>;

// Expand On Tensor Operation
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<Dtype> {
    shape: (usize, usize),
    storage: Vec<Dtype>,
    _storage: PhantomData<Dtype>,
}

trait TensorTrait<Dtype> {
    fn to_tensor(self, row: usize, col: usize) -> Tensor<Dtype>;
}

impl<Dtype> TensorTrait<Dtype> for Vec<Dtype>
where
    Dtype: Debug,
{
    fn to_tensor(self, row: usize, col: usize) -> Tensor<Dtype> {
        assert_eq!(self.len(), row * col, "Incorrect dimensions for tensor");
        Tensor {
            shape: (row, col),
            storage: self,
            _storage: PhantomData,
        }
    }
}

impl<Dtype> Tensor<Dtype>
where
    Dtype: Clone + Debug + From<u8>,
{
    pub fn new(row: usize, col: usize, vec: Vec<Dtype>) -> Self {
        assert_eq!(row * col, vec.len(), "Incorrect dimensions for tensor");
        Tensor {
            shape: (row, col),
            storage: vec,
            _storage: PhantomData,
        }
    }

    pub fn ones(row: usize, col: usize) -> Self {
        let storage = vec![Dtype::from(1); row * col];
        Tensor {
            shape: (row, col),
            storage,
            _storage: PhantomData,
        }
    }

    pub fn zeros(row: usize, col: usize) -> Self {
        let storage = vec![Dtype::from(0); row * col];
        Tensor {
            shape: (row, col),
            storage,
            _storage: PhantomData,
        }
    }

    pub fn rand_f32(row: usize, col: usize) -> Tensor<f32> {
        let mut rng = rand::thread_rng();
        let storage = (0..row * col).map(|_| rng.gen::<f32>()).collect();

        Tensor {
            shape: (row, col),
            storage,
            _storage: PhantomData,
        }
    }
    pub fn rand_f64(row: usize, col: usize) -> Tensor<f64> {
        let mut rng = rand::thread_rng();
        let storage = (0..row * col).map(|_| rng.gen::<f64>()).collect();

        Tensor {
            shape: (row, col),
            storage,
            _storage: PhantomData,
        }
    }
}

// Seen these implemented but don't exactly know what there used for
// ==================================================================
// #[derive(Debug, Clone, Copy)]
// struct TensorSlice<'a, T: 'a>{
//     ptr : *const T,
//     shape: Shape<'a>,
//     marker : PhantomData<&'a T>
// }

// #[derive(Debug, Clone, Copy)]
// struct TensorSliceMut<'a, T: 'a>{
//     ptr : *mut T,
//     shape: _Shape<'a>,
//     marker : PhantomData<&'a mut T>
// }
// ==================================================================

#[cfg(test)]
mod test {

    use super::{Tensor, TensorTrait};

    #[test]
    fn vec_to_tensor() {
        let vec1 = vec![0; 10];
        let vec2 = vec![0; 10];
        let tensor1 = vec1.to_tensor(2, 5);
        let tensor2 = Tensor::new(2, 5, vec2);
        assert_eq!(tensor1, tensor2);
    }

    fn rng_test() {
        let _tensor: Tensor<f32> = Tensor::<f32>::rand_f32(10usize, 10usize);
        println!("{:?}", _tensor);
    }
}
