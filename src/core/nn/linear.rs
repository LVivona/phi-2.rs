use crate::tensor::Tensor;

pub struct Linear<Dtype> {
    in_features: usize,
    out_features: usize,
    pub weight: Tensor<Dtype>,
    bias: Option<Vec<Dtype>>,
}

///
///
impl<Dtype> Linear<Dtype>
where
    Dtype: Clone + Default,
{
    ///
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        assert!(
            in_features > 0,
            "ValueError: in_feature={}, must be greater then 0",
            in_features
        );
        assert!(
            out_features > 0,
            "ValueError: out_feature={}, must be greater then 0",
            out_features
        );

        // Initialize the weight vector with a default value and the specified size
        // NOTE: T::default could be used for tenosr::default
        let weight = vec![Dtype::default(); in_features * out_features];

        // Initialize the bias vector if needed
        let bias = if bias {
            Some(vec![Dtype::default(); out_features])
        } else {
            None
        };

        Linear {
            in_features,
            out_features,
            weight,
            bias,
        }
    }

    ///
    pub fn forward(&self, x: Vec<Dtype>) -> Vec<Dtype>
    where
        Dtype: Clone,
    {
        assert!(x.len() % self.in_features == 0, "");

        // apply W (out_features, in_features)
        // apply x (*, in_features)
        // bias    (1, out_features)
        // output  (*, out_features)
        // xW^T + b

        
    }
}


