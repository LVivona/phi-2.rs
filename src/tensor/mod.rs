#![allow(unreachable_code, dead_code, unused_imports)]
// Simplest case, using 1D vec reproduce a matrix
// later can expand on our own datatype that utilize
// low level hardware
// pub type Tensor<T> = Vec<T>;

pub(crate) mod error;

// rand == "0.8.5"
use rand::distributions::{Distribution, Standard};
use rand::Rng;

// std library
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

// tensor sub module(s)
use error::TensorError;

// simple tensor form with no batch
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<Dtype> {
    pub shape: (usize, usize),
    pub storage: Vec<Dtype>,
    _storage: PhantomData<Dtype>,
    #[cfg(feature = "retain_gradients")]
    pub gradients: Option<Vec<Dtype>>,
}

/// Implementation of the `IntoTensor` trait for vectors.
///
/// This implementation allows a vector of data to be converted into a tensor,
/// given the desired shape (number of rows and columns). If the size of the vector
/// does not match the specified shape, an error is returned.
pub trait IntoTensor<Dtype> {
    /// Converts a collection into a tensor with the specified shape.
    ///
    /// This method takes ownership of the collection and reorganizes its elements into a tensor
    /// with the given number of rows and columns. The size of the collection must exactly match
    /// the product of `row` and `col`, otherwise an error is returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use your_crate::{Tensor, IntoTensor, TensorError};
    /// let vec = vec![1, 2, 3, 4, 5, 6];
    /// let tensor = vec.to_tensor(2, 3).unwrap();
    /// assert_eq!(tensor.shape(), (2, 3));
    /// assert_eq!(tensor.storage(), &[1, 2, 3, 4, 5, 6]);
    /// ```
    ///
    /// Error case:
    ///
    /// ```
    /// # use your_crate::{Tensor, IntoTensor, TensorError};
    /// let vec = vec![1, 2, 3, 4, 5];
    /// let result = vec.to_tensor(2, 3);
    /// assert!(result.is_err());
    /// assert_eq!(result.unwrap_err(), TensorError::InvalidShape { expected: (2, 3), found: (5, 1) });
    /// ```
    ///
    /// # Parameters
    ///
    /// - `self`: The collection to be converted into a tensor.
    /// - `row`: The number of rows for the resulting tensor.
    /// - `col`: The number of columns for the resulting tensor.
    ///
    /// # Returns
    ///
    /// - `Ok(Tensor<Dtype>)`: A tensor with the specified shape, if the size of the collection matches the shape.
    /// - `Err(TensorError)`: An error if the size of the collection does not match the specified shape.
    fn to_tensor(self, row: usize, col: usize) -> Result<Tensor<Dtype>, TensorError>;
}

impl<Dtype> IntoTensor<Dtype> for Vec<Dtype>
where
    Dtype: Debug,
{
    fn to_tensor(self, row: usize, col: usize) -> Result<Tensor<Dtype>, TensorError> {
        if self.len() != row * col {
            return Err(TensorError::InvalidShape {
                expected: (row, col),
                found: (self.len(), 1),
            });
        }

        Ok(Tensor {
            shape: (row, col),
            storage: self,
            _storage: PhantomData,
            #[cfg(feature = "retain_gradients")]
            gradients: None,
        })
    }
}

impl<Dtype> Tensor<Dtype>
where
    Dtype: Clone + Debug + From<u8>,
{
    pub fn new(shape: (usize, usize), storage: Vec<Dtype>) -> Result<Self, TensorError> {
        if storage.len() != shape.0 * shape.1 {
            return Err(TensorError::InvalidShape {
                expected: shape,
                found: (storage.len(), 1),
            });
        }

        Ok(Tensor {
            shape,
            storage,
            _storage: PhantomData,
            #[cfg(feature = "retain_gradients")]
            gradients: None,
        })
    }

    pub fn ones(shape: (usize, usize)) -> Self {
        let storage = vec![Dtype::from(1); shape.0 * shape.1];
        Tensor {
            shape,
            storage,
            _storage: PhantomData,
            #[cfg(feature = "retain_gradients")]
            gradients: None,
        }
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        let storage = vec![Dtype::from(0); shape.0 * shape.1];
        Tensor {
            shape,
            storage,
            _storage: PhantomData,
            #[cfg(feature = "retain_gradients")]
            gradients: None,
        }
    }

    pub fn rand_f32(shape: (usize, usize)) -> Tensor<f32> {
        let mut rng = rand::thread_rng();
        let storage: Vec<f32> = (0..shape.0 * shape.1).map(|_| rng.gen::<f32>()).collect();

        Tensor {
            shape,
            storage,
            _storage: PhantomData,
            #[cfg(feature = "retain_gradients")]
            gradients: None,
        }
    }

    pub fn retain_grad(&self) -> bool {
        cfg!(feature = "retain_gradients")
    }

    ///
    /// Example:
    /// ```
    /// let shape : (usize, usize) = (1, 5);
    /// let t = Tensor::<f64>::rand_f64(shape);
    /// ```
    pub fn rand_f64(shape: (usize, usize)) -> Tensor<f64> {
        let mut rng = rand::thread_rng();
        let storage: Vec<f64> = (0..shape.0 * shape.1).map(|_| rng.gen::<f64>()).collect();

        Tensor {
            shape,
            storage,
            _storage: PhantomData,
            #[cfg(feature = "retain_gradients")]
            gradients: None,
        }
    }

    // Implement this
    // pub fn mul_(&self, y: Tensor<Dtype>) {}
    // pub fn add_(&self, y: Tensor<Dtype>) {}
    // pub fn sub_(&self, y: Tensor<Dtype>) {}
    // pub fn div_(&self, y: Tensor<Dtype>) {}
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
    use super::{IntoTensor, Tensor, TensorError};

    #[test]
    fn vec_to_tensor_success() {
        let vec = vec![1, 2, 3, 4, 5, 6];
        let tensor: Tensor<i32> = vec.to_tensor(2, 3).unwrap();
        assert_eq!(tensor.shape, (2, 3));
        assert_eq!(tensor.storage, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn vec_to_tensor_invalid_shape() {
        let vec = vec![1, 2, 3, 4, 5];
        let result = vec.to_tensor(2, 3);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            TensorError::InvalidShape {
                expected: (2, 3),
                found: (5, 1)
            }
        );
    }

    #[test]
    fn tensor_rand_shape() {
        let shape: (usize, usize) = (10, 10);
        let tensor = Tensor::<f32>::rand_f32(shape);
        assert_eq!(tensor.shape, shape);
    }

    #[test]
    fn tensor_equality() {
        let tensor1 = Tensor::new((2, 2), vec![1, 2, 3, 4]).unwrap();
        let tensor2 = Tensor::new((2, 2), vec![1, 2, 3, 4]).unwrap();
        let tensor3 = Tensor::new((2, 2), vec![4, 3, 2, 1]).unwrap();
        assert_eq!(tensor1, tensor2);
        assert_ne!(tensor1, tensor3);
    }

    #[test]
    fn test_retain_grad() {
        let tensor = Tensor::new((2, 2), vec![1, 2, 3, 4]).unwrap();
        assert_eq!(cfg!(feature = "retain_gradients"), tensor.retain_grad())
    }
}
