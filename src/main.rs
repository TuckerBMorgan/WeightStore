use ndarray::prelude::*;
use ndarray::Zip;
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

    fn break_up_image_by_channels_and_store(image_with_channels: &WeightToken, weight_store: &mut WeightStore) -> Vec<WeightToken> {
        let full_image = weight_store.get_weights(&image_with_channels);

        //The dimensions of the stored weights is, (channels, width, height)
        let image_size = image_with_channels.dimensions[1] * image_with_channels.dimensions[2];
        let chunks = full_image.chunks(image_size as usize);
        let mut weight_tokens = vec![];
        for c in chunks {
            let as_vec = c.to_vec();
            weight_tokens.push(weight_store.add_weights(as_vec, vec![image_with_channels.dimensions[1], image_with_channels.dimensions[2]]));
        }
        return weight_tokens;
    }

    fn single_filter_on_image(filter: &WeightToken, image_with_channel: &WeightToken, weight_store: &mut WeightStore) {
        //This is a channels first library, got a problem with it, go write python
        let channels = image_with_channel.dimensions[0];
        let split_up_image = Conv2d::break_up_image_by_channels_and_store(&image_with_channel, weight_store);
        let split_up_filter = Conv2d::break_up_image_by_channels_and_store(&filter, weight_store);

        let mut convauled_channels = vec![];
        for i in 0..channels {
            let convaluted_image = Conv2d::single_image_single_channel_single_filter(&split_up_image[i], &split_up_filter[i], weight_store);
            convauled_channels.push(convaluted_image);
        }

        let mut combined_convalted_image = vec![];
        for i in 0..convauled_channels[0].len() {
            let mut sum = 0.0;
            for j in 0..channels {
                sum += convauled_channels[j][i];
            }
            combined_convalted_image.push(sum);
        }
        let weight_token = weight_store.add_weights(combined_convalted_image, vec![1, image_with_channel.dimensions[1], image_with_channel.dimensions[2]]);
    }

    fn single_image_single_channel_single_filter(image: &WeightToken, filter: &WeightToken, weight_store: &mut WeightStore) -> Vec<f32> {
        let image_values = weight_store.get_weights(image);
        let filter_values = weight_store.get_weights(filter);

        let image_values_dimensions = (image.dimensions[0], image.dimensions[1]);
        let input_weight_matrix = Array::from_shape_vec(image_values_dimensions, image_values).unwrap();

        let filter_values_dimensions = (filter.dimensions[0], filter.dimensions[1]);
        let filter_weight_matrix = Array::from_shape_vec(filter_values_dimensions, filter_values).unwrap();

        let mut results = vec![];
        for f in filter_weight_matrix.windows(image_values_dimensions) {
            let result = f.dot(&input_weight_matrix);
            results.push(result);
        }
        
        let mut final_result = vec![];
        for matrix in results {
            final_result.extend(matrix.iter().map(|x|*x));
        }
        return final_result;
    }

    //this will remap the image into a array such that it can be dot producted by a similar 
    //remapped matrix of the filters to preform the conv operation
    fn remap_image_to_dense(image: &WeightToken, filter_dimensions: &Vec<usize>, weight_store: &mut WeightStore) -> WeightToken {
        
        let kernal_size = filter_dimensions[2];
        let image_array = weight_store.get_weights(&image);
        let (height, width) = (image.dimensions[1], image.dimensions[2]);
        
        //This will create sub vectors of image channel
        let channels = image_array.chunks(height * width);
        let mut full_image_with_channels = vec![];
        let number_of_kernals_per_single_channel_of_image = (height - kernal_size) + 1;
        for sub_image in channels {
            let mut single_channel = vec![];
            //I hate doing this, but I also don't want to figure out the fucking array index math
            //dont look at the next function where I do actually work on the array index math
            let image_as_matrix = Array::from_shape_vec((height, width), sub_image.to_vec()).unwrap();
            for w in image_as_matrix.windows((kernal_size, kernal_size)) {
                let s : Vec<f32> = w.iter().map(|x|*x).collect();
                single_channel.extend(s);
            }
            full_image_with_channels.extend(single_channel);
        }
        
        return weight_store.add_weights(full_image_with_channels, vec![image.dimensions[0], number_of_kernals_per_single_channel_of_image, kernal_size * kernal_size]);
    }

    fn remap_filters_into_dot_product_matrix(filters: &WeightToken, image_dimensions: &Vec<usize>, weight_store: &mut WeightStore) -> WeightToken {
        //Filter dimensions should be(number of filters, channels, height, weight)
        //Image dimensions hsould be (channels, height, weight)
        let number_of_channels = image_dimensions[0];
        let filter_size = filters.dimensions[2];
        let number_of_filters = filters.dimensions[0];
        let (image_height, _) = (image_dimensions[1], image_dimensions[2]);
        let number_of_kernals_per_single_channel_of_image = (image_height - filter_size) + 1;
        //need to copy a single channel of a filter number_filters_per_single_of_image times, flatten it, then copy that for each channel, for each filter
        let combined_height_width = filters.dimensions[2] * filters.dimensions[3];
        let filter_weights : Vec<f32> = weight_store.get_weights(&filters);
        //this could be paralized without MUCH problem
        let mut stacked_filter_weights = vec![];
        for i in 0..number_of_filters {
            for j in 0..number_of_channels { 
                let start_index = (j * filter_size * filter_size) + (i * number_of_channels * filter_size * filter_size);
                let filter_weights = &filter_weights[start_index..(filter_size * filter_size)];
                for k in 0..number_of_kernals_per_single_channel_of_image {
                    stacked_filter_weights.extend(filter_weights.clone());
                }
            }
        }
        let len = stacked_filter_weights.len();
        return weight_store.add_weights(stacked_filter_weights, vec![number_of_filters, number_of_channels, number_of_kernals_per_single_channel_of_image ,combined_height_width]);
    }
    
}

impl Node for Conv2d {


    fn forward(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> {

        //the weights for the filters, they are form, number of filters, channels, height, weight
        let filter_weights_token = weight_tokens[0];
        let input_weight_token = weight_tokens[1];
        let filter_weight_array = weight_store.get_weights(&filter_weights_token);
        let remapped_filters = Conv2d::remap_filters_into_dot_product_matrix(filter_weights_token, &input_weight_token.dimensions, weight_store);
        let remapped_image = Conv2d::remap_image_to_dense(&input_weight_token, &filter_weights_token.dimensions, weight_store);
        
        let remapped_filters_weights = weight_store.get_weights(&remapped_filters);
        let remapped_image_weights = weight_store.get_weights(&remapped_image);

        //This is not quite correct, not taking into account Channel;s
        let remapped_filters_matrix = Array::from_shape_vec((remapped_filters.dimensions[0], remapped_filters.dimensions[1], remapped_filters.dimensions[2], remapped_filters.dimensions[3]), remapped_filters_weights).unwrap();
        let remapped_image_matrix = Array::from_shape_vec((remapped_image.dimensions[0], remapped_image.dimensions[1], remapped_image.dimensions[2]), remapped_image_weights).unwrap();
        //For each filter
        for f in remapped_filters_matrix.outer_iter() {
            //For each channel
            let mut channels = vec![];
            let mut kernal_channel = f.outer_iter();
            let mut image_channel = remapped_image_matrix.outer_iter();
            let combined = kernal_channel.zip(image_channel);
            for (kc, im) in combined {
                let result = im.dot(&kc);
                channels.push(combined);
            }
            //flatten them
            //sum them
            //Store them
            //drink yourself to death
            
        }


        return vec![];
    }



    fn backprop(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> {
        return vec![];
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
