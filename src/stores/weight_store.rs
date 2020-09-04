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