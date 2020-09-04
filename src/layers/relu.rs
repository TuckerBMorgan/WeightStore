pub use crate::stores::*;
pub use crate::layers::*;

pub struct Relu {
    
}

impl Relu {
    pub fn new() -> Relu {
        Relu {

        }
    }
}

impl Node for Relu {
    fn forward(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> { 
        //Relu is a strict mapping of M -> M only changing the values
        //We only need a single weight token, the inputs
        let inputs = weight_tokens[0];
        let input_array = weight_store.get_weights(inputs);

        //Preform the mapping
        let relued_values = input_array.iter().map(|x|{
            if *x > 0.0f32 {
                return *x;
            }
            return 0f32;
        }).collect();
        //add them to the weight store
        return vec![weight_store.add_weights(relued_values, inputs.dimensions.clone())];
    }

    fn backprop(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> { 
        let output = weight_tokens[0];
        let output_array = weight_store.get_weights(output);
        let derivative_values = output_array.iter().map(|x| {
            if *x > 0.0f32 {
                return 1.0f32;
            }
            else if *x == 0.0f32 {
                return 0.5f32;
            }
            return 0.0f32;
        }).collect();
        return vec![weight_store.add_weights(derivative_values, output.dimensions.clone())];
    }
}