
//NOTE: convert these into macros 

/// Applies the Gaussian Error Linear Unit (GELU) activation function to a single-precision floating-point number.
///
/// The GELU activation function is defined as:
/// `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`
///
/// # Examples
///
/// ```
/// let x = 0.5;
/// let y = new_gelu_f32(x);
/// println!("GELU({}) = {}", x, y);
/// ```
///
/// # Parameters
/// - `x`: A single-precision floating-point number.
///
/// # Returns
/// - The result of applying the GELU activation function to `x`.
pub fn new_gelu_f32(x : f32) -> f32
{
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    
}


/// Applies the Gaussian Error Linear Unit (GELU) activation function to a double-precision floating-point number.
///
/// The GELU activation function is defined as:
/// `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`
///
/// # Examples
///
/// ```
/// let x = 0.5;
/// let y = new_gelu_f64(x);
/// println!("GELU({}) = {}", x, y);
/// ```
///
/// # Parameters
/// - `x`: A double-precision floating-point number.
///
/// # Returns
/// - The result of applying the GELU activation function to `x`.
pub fn new_gelu_f64(x : f64) -> f64
{
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

pub fn sigmoid_f64(x : f64) -> f64{
    1.0 / (1.0 + (-x).exp())
}
pub fn sigmoid_f32(x : f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// TODO: add 16-bit and lower...
// ADD HERE

#[cfg(test)]
mod test {
    use super::{new_gelu_f32, new_gelu_f64, sigmoid_f32, sigmoid_f64};


    #[test]
    fn gelu_test_f32(){

        assert_eq!(new_gelu_f32(-6.0), 0.0);
    }

    #[test]
    fn gelu_test_f64(){
        assert_eq!(new_gelu_f64(2.0), 1.954597694087775);
    }

    #[test]
    fn test_sigmoid_f64(){
        assert_eq!(sigmoid_f64(std::f64::MAX), 1.0);
        assert_eq!(sigmoid_f64(std::f64::MIN), 0.0);
        assert_eq!(sigmoid_f64(0.0), 0.5);
    }

    #[test]
    fn test_sigmoid_f32(){
        assert_eq!(sigmoid_f32(std::f32::MAX), 1.0);
        assert_eq!(sigmoid_f32(std::f32::MIN), 0.0);
        assert_eq!(sigmoid_f32(0.0), 0.5);
    }
}