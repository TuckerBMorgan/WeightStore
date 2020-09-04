use ndarray::prelude::*;
use ndarray::Zip;

pub use crate::stores::*;
pub use crate::layers::*;

pub struct Conv2d {

}

impl Conv2d {
    pub fn new() -> Conv2d {
        Conv2d {

        }   
    }

    //this will remap the image into a array such that it can be dot producted by a similar 
    //remapped matrix of the filters to preform the conv operation
    fn remap_image_to_dot_product_matrix(image: &WeightToken, filter_dimensions: &Vec<usize>, weight_store: &mut WeightStore) -> WeightToken {
        
        let kernal_size = filter_dimensions[2];
        let image_array = weight_store.get_weights(&image);
        let (height, width) = (image.dimensions[1], image.dimensions[2]);
        let number_of_channels = image.dimensions[0];
        
        let number_of_kernals_per_single_channel_of_image = (height - kernal_size) + 1;
        let image_as_matrix = Array::from_shape_vec((number_of_channels, height, width), image_array.to_vec()).unwrap();
        let windows = image_as_matrix.windows((number_of_channels,kernal_size, kernal_size));
        let mut remapped_image = vec![];
        for w in windows {
            remapped_image.extend(w.iter());
        }
        return weight_store.add_weights(remapped_image, vec![number_of_kernals_per_single_channel_of_image, kernal_size * kernal_size * image.dimensions[0],]);
    }

    fn remap_filters_into_dot_product_matrix(filters: &WeightToken, weight_store: &mut WeightStore) -> WeightToken {
        //Filter dimensions should be(number of filters, channels, height, weight)
        let number_of_channels = filters.dimensions[1];
        let filter_size = filters.dimensions[2];
        let number_of_filters = filters.dimensions[0];
        //need to copy a single channel of a filter number_filters_per_single_of_image times, flatten it, then copy that for each channel, for each filter
        let combined_height_width = filters.dimensions[2] * filters.dimensions[3];
        let filter_weights : Vec<f32> = weight_store.get_weights(&filters);
        //this could be paralized without MUCH problem
        let mut stacked_filter_weights = vec![];
        for i in 0..number_of_filters {
            let mut single_filter_all_channels : Vec<f32> = vec![];
            for j in 0..number_of_channels { 
                let start_index = (j * filter_size * filter_size) + (i * number_of_channels * filter_size * filter_size);
                let filter_weights = &filter_weights[start_index..(filter_size * filter_size)];
                single_filter_all_channels.extend(filter_weights.clone());
            }
            stacked_filter_weights.extend(single_filter_all_channels);
        }

        return weight_store.add_weights(stacked_filter_weights, vec![number_of_filters, combined_height_width * number_of_channels]);
    }

    fn remap_filter_output_dervitaive_for_full_convulution_with_filter(fiter: &WeightToken, output_derivative: &WeightToken, weight_store: &mut WeightStore) -> WeightToken {
        //rotate the filter
        //zero pad the outuput as part it its remapping
        let image_size = output_derivative.dimensions[1];
        let kernel_size = filter.dimensions[2];
        let image_weights = weight_store.get_weights(filter);
        //height of the filter + height of the output - 1(they will always overlap by at least one pixel)
        let extra_size = kernel_size - 1 + image_size;
        let mut for_windowing = Array::zeros((extra_size, extra_size));
        for y in 0..image_size {
            for x in 0..image_size {
                let one_d_index = x + (y * image_size);
                let two_d_index = (x + , y + );
                for_windowing[two_d_index.0][two_d_index.1] = image_weights[one_d_index];
            }
        }
        let windows = for_windowing.windows((kernel_size, kernel_size));
        let mut flattened_windows = vec![];
        for window in windows {
            flattened_windows.extend(window.iter());
        }
        return weight_store.add_weights(flattened_windows, vec![extra_size, extra_size]);
    }
    
}

impl Node for Conv2d {


    fn forward(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> {

        //the weights for the filters, they are form, number of filters, channels, height, weight
        let filter_weights_token = weight_tokens[0];
        let input_weight_token = weight_tokens[1];
        let remapped_filters = Conv2d::remap_filters_into_dot_product_matrix(filter_weights_token, weight_store);
        let remapped_image = Conv2d::remap_image_to_dot_product_matrix(&input_weight_token, &filter_weights_token.dimensions, weight_store);
        
        let remapped_filters_weights = weight_store.get_weights(&remapped_filters);
        let remapped_image_weights = weight_store.get_weights(&remapped_image);

        //This is not quite correct, not taking into account Channel;s
        let remapped_filters_matrix = Array::from_shape_vec((remapped_filters.dimensions[0], remapped_filters.dimensions[1]), remapped_filters_weights).unwrap();
        let remapped_image_matrix = Array::from_shape_vec((remapped_image.dimensions[0], remapped_image.dimensions[1]), remapped_image_weights).unwrap();
        //For each filter
        let result = remapped_image_matrix.dot(&remapped_filters_matrix);
        let stored_resulted : Vec<f32> = result.iter().map(|x|*x).collect();
        let number_of_kernals_per_single_channel_of_image = (input_weight_token.dimensions[1] - filter_weights_token.dimensions[2]) + 1;
        //TODO: I may need to shuffle stored_resulted so that each of the outputs of an image are stacked ontop of each other
        return vec![weight_store.add_weights(stored_resulted, vec![filter_weights_token.dimensions[0], number_of_kernals_per_single_channel_of_image, number_of_kernals_per_single_channel_of_image])];
    }

    fn backprop(weight_tokens: Vec<&WeightToken>, weight_store: &mut WeightStore) -> Vec<WeightToken> {

        //TODO: while this is very explicit, it also is very boiler plate, can be done better
        let weights = weight_tokens[0];
        let derivative_of_output = weight_tokens[1];
        let error = weight_tokens[2];

        let weights_array = weight_store.get_weights(weights);
        let derivative_of_output_array = weight_store.get_weights(derivative_of_output);
        let error_array = weight_store.get_weights(error);

        let weight_matrix = Array::from_shape_vec((weights.dimensions[0], weights.dimensions[1], weights.dimensions[2]), weights_array).unwrap();
        let derivative_of_output_matrix = Array::from_shape_vec((derivative_of_output.dimensions[0], derivative_of_output.dimensions[1], derivative_of_output.dimensions[2]), derivative_of_output_array).unwrap();
        let error_matrix = Array::from_shape_vec((error.dimensions[0], error.dimensions[1], error.dimensions[2]), error_array).unwrap();
        

        //This is actually the entire work of the backprop, the rest of the code is boilerplate
        let deltas = error_matrix * derivative_of_output_matrix;
        //create an image for each filter
        //sum them up then
        // I think I need to sum each index as 
        //let next_error = weight_matrix.dot(&deltas.t()).reversed_axes();

        //Convert it into a flat array for storing
        let delta_store : Vec<f32> = deltas.iter().map(|x|*x).collect();
    
        //We need to get the dimensions for the weight token
        let mut delta_dimensions = vec![];
        for s in deltas.shape() {
            delta_dimensions.push(*s);
        }

        let delta_weight_token = weight_store.add_weights(delta_store, delta_dimensions);

        //TODO: check the actual data type of shape, and see if we can do it better, or if I can just precalucalte this
        //let next_error_store : Vec<f32> = next_error.iter().map(|x|*x).collect();
        //let mut next_error_dimenions = vec![];
       // for s in next_error.shape() {
       //     next_error_dimenions.push(*s);
       // }

        //let next_error_token = weight_store.add_weights(next_error_store, next_error_dimenions);
        return vec![delta_weight_token];//, next_error_token];
    }
}