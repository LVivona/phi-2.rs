#[derive(Debug, PartialEq)]
pub enum TensorError {
    InvalidShape {
        expected: (usize, usize),
        found: (usize, usize),
    },
    OutOfBounds {
        index: usize,
    },
    ArithmeticMismatch {
        operation: String,
        shape1: (usize, usize),
        shape2: (usize, usize),
    },
    NotImplemented {
        feature: String,
    },
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::InvalidShape { expected, found } => {
                write!(
                    f,
                    "Invalid tensor shape: expected {:?}, found {:?}",
                    expected, found
                )
            }
            TensorError::OutOfBounds { index } => {
                write!(f, "Index out of bounds: {}", index)
            }
            TensorError::ArithmeticMismatch {
                operation,
                shape1,
                shape2,
            } => {
                write!(
                    f,
                    "Arithmetic operation '{}' cannot be performed on shapes {:?} and {:?}",
                    operation, shape1, shape2
                )
            }
            TensorError::NotImplemented { feature } => {
                write!(f, "Feature not implemented: {}", feature)
            }
        }
    }
}

impl std::error::Error for TensorError {}
