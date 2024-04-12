use std::fmt::{Display, Formatter};

#[derive(Debug, PartialEq)]
pub enum Device {
    Cpu,
    Cuda(u8),
}

impl Display for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => {
                write!(f, "cpu")
            }
            Device::Cuda(id) => {
                write!(f, "cuda:{0}", id)
            }
        }
    }
}
