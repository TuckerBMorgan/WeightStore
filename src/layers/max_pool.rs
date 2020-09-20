use ndarray::prelude::*;
use ndarray::Zip;

pub use crate::stores::*;
pub use crate::layers::*;


pub struct DenseLayer {
}

impl DenseLayer {
    pub fn new() -> DenseLayer {
        DenseLayer {

        }
    }
}

impl Node for DenseLayer {
    
    fn forward(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> {
        
    }
    
    fn backprop(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> {
    
    }
}