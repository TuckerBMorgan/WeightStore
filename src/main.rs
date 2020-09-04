use ndarray::prelude::*;
use ndarray::Zip;

pub mod stores;
use stores::*;

pub mod layers;
use layers::*;

fn main() {
    let mut weight_store = WeightStore::new();
    let test_weight_token = weight_store.add_weights(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
//    let re = Relu::new();
}
