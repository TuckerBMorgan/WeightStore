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
        
        //A dense layer needs the weights of the layer, and the input
        let weights = weight_tokens[0];
        let inputs = weight_tokens[1];

        //Get the values for the weights and the inputs that are in the weight store
        let layer_weights = weight_store.get_weights(weights);
        let inputs_values = weight_store.get_weights(inputs);

        //Get the layer weights 
        let layer_weights_array = Array::from_shape_vec((weights.dimensions[0], weights.dimensions[1]), layer_weights).unwrap();
        let input_values_array = Array::from_shape_vec((1, inputs.dimensions[0]), inputs_values).unwrap();

        //Preform the mul + sum that is a linear layer
        let layer_result = input_values_array.dot(&layer_weights_array);
        
        //Convert it into a flat array for storing
        let stored_resulted : Vec<f32> = layer_result.iter().map(|x|*x).collect();
    
        //We need to get the dimensions for the weight token
        let mut dimensions = vec![];
        for s in input_values_array.shape() {
            dimensions.push(*s);
        }

        //Add the weights into the weight store, returning the token that points to them
        return vec![weight_store.add_weights(stored_resulted, dimensions)];
    }

    fn backprop(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> {

        //TODO: while this is very explicit, it also is very boiler plate, can be done better
        let weights = weight_tokens[0];
        let derivative_of_output = weight_tokens[1];
        let error = weight_tokens[2];

        let weights_array = weight_store.get_weights(weights);
        let derivative_of_output_array = weight_store.get_weights(derivative_of_output);
        let error_array = weight_store.get_weights(error);

        let weight_matrix = Array::from_shape_vec((weights.dimensions[0], weights.dimensions[1]), weights_array).unwrap();
        let derivative_of_output_matrix = Array::from_shape_vec((derivative_of_output.dimensions[0], derivative_of_output.dimensions[1]), derivative_of_output_array).unwrap();
        let error_matrix = Array::from_shape_vec((error.dimensions[0], error.dimensions[1]), error_array).unwrap();
        
        

        //This is actually the entire work of the backprop, the rest of the code is boilerplate
        let deltas = error_matrix * derivative_of_output_matrix;
        let next_error = weight_matrix.dot(&deltas.t()).reversed_axes();

        //Convert it into a flat array for storing
        let delta_store : Vec<f32> = deltas.iter().map(|x|*x).collect();
    
        //We need to get the dimensions for the weight token
        let mut delta_dimensions = vec![];
        for s in deltas.shape() {
            delta_dimensions.push(*s);
        }

        let delta_weight_token = weight_store.add_weights(delta_store, delta_dimensions);

        //TODO: check the actual data type of shape, and see if we can do it better, or if I can just precalucalte this
        let next_error_store : Vec<f32> = next_error.iter().map(|x|*x).collect();
        let mut next_error_dimenions = vec![];
        for s in next_error.shape() {
            next_error_dimenions.push(*s);
        }

        let next_error_token = weight_store.add_weights(next_error_store, next_error_dimenions);
        return vec![delta_weight_token, next_error_token];
    }
}