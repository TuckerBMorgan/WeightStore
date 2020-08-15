use ndarray::prelude::*;

pub struct WeightToken {
    pub start_index: usize,
    pub dimensions: Vec<usize>
}

impl WeightToken {
    pub fn new(start_index: usize, dimensions: Vec<usize>) -> WeightToken {
        WeightToken {
            start_index,
            dimensions
        }
    }
}

pub struct WeightStore {
    weights: Vec<f32>
}

impl WeightStore {
    pub fn new() -> WeightStore {
        WeightStore {
            weights: vec![]
        }
    }

    pub fn get_weights(&self, weight_token: &WeightToken) -> Vec<f32> {
        let start_index = weight_token.start_index;
        let number_of = weight_token.dimensions.iter().fold(1, |x, sum| x * sum);
        return self.weights[start_index..(start_index + number_of)].to_vec();
    }

    pub fn set_weights(&mut self, weight_token: &WeightToken, new_weights: Vec<f32>) {
        let number_of = weight_token.dimensions.iter().fold(1, |x, sum| x * sum);
        self.weights[weight_token.start_index..(weight_token.start_index + number_of)].clone_from_slice(new_weights.as_slice());
    }

    pub fn add_weights(&mut self, new_weights: Vec<f32>, dimensions: Vec<usize>) -> WeightToken {
        let start_index = self.weights.len();
        let weight_token = WeightToken::new(start_index, dimensions);
        self.weights.extend(new_weights.iter());
        return weight_token;
    }
}

pub trait Node {
    fn forward(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken>;
    fn backprop(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken>;
}

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
        

        let deltas = (error_matrix * derivative_of_output_matrix);
        let next_error = weight_matrix.dot(&deltas.t()).reversed_axes();

        //Convert it into a flat array for storing
        let delta_store : Vec<f32> = deltas.iter().map(|x|*x).collect();
    
        //We need to get the dimensions for the weight token
        let mut delta_dimensions = vec![];
        for s in deltas.shape() {
            delta_dimensions.push(*s);
        }

        let delta_weight_token = weight_store.add_weights(delta_store, delta_dimensions);

        //TODO: check the actual data type of shape, and see if we can do it better
        let next_error_store : Vec<f32> = next_error.iter().map(|x|*x).collect();
        let mut next_error_dimenions = vec![];
        for s in next_error.shape() {
            next_error_dimenions.push(*s);
        }

        let next_error_token = weight_store.add_weights(next_error_store, next_error_dimenions);
        return vec![delta_weight_token, next_error_token];
    }
}

pub struct Conv2d {

}

impl Conv2d {
    pub fn new() -> Conv2d {
        Conv2d {

        }
    }
}
fernal size 

impl Node for Conv2d {
    fn forward(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> {
        //the weights for the filters, they are form, number of filters, channels, height, weight
        let filter_weights_token = weight_tokens[0];
        let input_weight_token = weight_token[1];

        //Get the weights from the store
        let filter_weight_array = weight_store.get_weights(filter_weights_token);
        let input_weight_array = weight_store.get_weights(input_weight_token);

        //turn them into the Ndarray Matrix for easy matrix math operations
        let filter_weight_shape_tuple = (filter_weight_array.dimensions[0], filter_weight_array.dimensions[1], filter_weight_array.dimensions[2], filter_weight_array.dimensions[3]);
        let filter_weight_matrix = Array::<f32>::from_shape_vec(filter_weight_shape_tuple, filter_weight_array);

        let input_weights_tuple = (input_weight_token.dimensions[0], input_weight_token.dimensions[1], input_weight_token.dimensions[2]);
        let input_weight_matrix = Array::<f32>::from_shape_vec(input_weights_tuple, input_weight_array);

        //The size of the filter is the height and width, which are stored as part of the dimensions of the entire
        //filter weight block, number of filters, channels(EG, RGB for an image), height, weight
        let filter_and_window_size = (filter_weights_token.dimensions[2], filter_weights_token.dimensions[3]);
        let number_of_filters = filter_weights_token.dimensions[0];
        let number_of_channels = filter_weights_token.dimensions[1];

        //We will do this one filter at a time
        for filter_index in 0..number_of_filters {
            //Ndarray does not let us to dot product operations with 3D matricies, 
            //so we will break it down once more by the number of channels

            let filter_with_channel = filter_weight_matrix[filter_index];
            for channel_index in 0..number_of_channels {
                let filter = filter_with_channel[channel_index];
                let windows = input_weight_matrix.windows((self.filter_size.0, self.filter_size.1, 1, 1)); 
            }
        }
    }

    fn single_filter_on_image(filter: WeightToken, ) {

    }

    fn single_image_single_channel_single_filter(image: WeightToken, filter: WeightToken, weight_store: &mut WeightStore) {
        let image_values = weight_store.get_weights(image);
        let filter_values = weight_store.get_weights(filter);

        let image_values_dimensions = (image.dimensions[0], image.dimensions[1]);
        let input_weight_matrix = Array::<f32>::from_shape_vec(image_values_dimensions, image_values);

        let filter_values_dimensions = (filter.dimensions[0], filter.dimensions[1]);
        let filter_weight_matrix = Array::<f32>::from_shape_vec(filter_values_dimensions, filter_values);

        let mut results = vec![];
        for f in filter_weight_matrix.windows(image_values_dimensions) {
            let result = f.dot(&input_weight_matrix);
            results.push(result);
        }
    }

    fn backprop(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> {

    }
}
 

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

fn main() {
    let mut weight_store = WeightStore::new();
    let test_weight_token = weight_store.add_weights(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
}
