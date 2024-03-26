
pub struct Linear<T>{
    weight : Vec<T>,
    bias : Option<Vec<T>>
}


impl<T> Linear<T> {
   pub fn new(in_feature : usize, out_feature : usize, bias : bool) -> Self 
        where T : Clone
    {
        assert!(in_feature > 0, "ValueError: in_feature={}, must be greater then 0", in_feature);
        assert!(out_feature > 0, "ValueError: out_feature={}, must be greater then 0", out_feature);

        // Simple solution but need to abstract for better memory allocation
        Linear {
            weight : Vec::with_capacity(in_feature * out_feature),
            bias : match bias { 
                true => { Some(Vec::<T>::with_capacity(out_feature)) } 
                false => { None } 
            }
        }
    }

    pub fn forward(&self, x : Vec<T>) -> Vec<T> 
        where T : Clone
    {
        // apply W (out_features, in_features)
        // apply x (*, in_features)
        // bias    (1, out_features)
        // output  (*, out_features)
        // xW^T + b
        todo!("NotImplemented: soon will be implemented later")

    }
}
