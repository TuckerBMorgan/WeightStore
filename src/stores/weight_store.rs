pub struct WeightToken {
    pub start_index: usize,
    pub dimensions: Vec<usize>
}

impl WeightToken {
    fn new(start_index: usize, dimensions: Vec<usize>, id: usize) -> WeightToken {
        WeightToken {
            start_index,
            dimensions
        }
    }
}

pub struct WeightStore {
    weights: Vec<f32>,
    number_of_allocated_tokens: usize
}

impl WeightStore {
    pub fn new() -> WeightStore {
        WeightStore {
            weights: vec![],
            number_of_allocated_tokens: 0
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
        self.number_of_allocated_tokens += 1;
        let weight_token = WeightToken::new(start_index, dimensions, self.number_of_allocated_tokens);
        self.weights.extend(new_weights.iter());
        return weight_token;
    }
    /*
    pub fn get_windows_over_weights(&mut self, weight_token: &WeightToken, windows: Vec<usize>) -> Result<WeightToken> {
        if windows.len() != 2 {
            return Err("Windows must be of MxN size");
        }
        
        if windows[0] != windows[1] {
            return Err("Window size must be equal for the moment");
        }

        //how the fuck do I gen index for this
        //the first index is always... (y * size) + x
        for y in 0..windows[0] {
            let first_index = (y * windows[0]);
            for x in 0..windows[1] {
                let copy_index = first_index + x;
            }
        }
    }
    */
}