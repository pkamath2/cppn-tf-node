"use strict"

const tf = require('@tensorflow/tfjs-node');
const Gene = require('./gene');

class ConnectionGene extends Gene{
    constructor(innovation_number, weight_seed, from_node_id, to_node_id, is_active, is_output_connection, disabled){
        super();
        this.innovation_number = innovation_number;
        this.weight_seed = weight_seed;
        this.from_node_id = from_node_id;
        this.to_node_id = to_node_id;
        this.is_active = is_active;
        this.is_output_connection = is_output_connection;
        this.disabled = disabled;
    }

    evaluate(input, output_size, with_bias){
        if(output_size==null) output_size= 64;
        if(with_bias ==null) with_bias = false;
        if(this.is_active){
            var weights = tf.variable(tf.randomNormal([input.shape[1], output_size],0.0,2.0,'float32', this.weight_seed));
            var result = tf.matMul(input, weights);
            if(with_bias){
            var bias = tf.variable(tf.randomNormal([1, output_size],0.0,2.0,'float32', this.weight_seed));
            result = tf.add(result, tf.mul(bias, tf.ones([input.shape[0],1])));
            }
            return result;
        }
        else{
            return input;
        }
        
    }
}
module.exports = ConnectionGene;