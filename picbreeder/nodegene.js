"use strict"

const tf = require('@tensorflow/tfjs-node');
const Gene = require('./gene');

class NodeGene extends Gene{
    constructor(innovation_number, name, activation, hidden, input_conn_arr, output_conn_arr){
        super();
        this.innovation_number = innovation_number;
        this.name = name;
        this.activation = activation;
        this.hidden = hidden;
        this.input_conn_arr = (input_conn_arr==null?[]:input_conn_arr);
        this.output_conn_arr = (output_conn_arr==null?[]:output_conn_arr);
    }

    addInputConnection(input_conn_arr){
        this.input_conn_arr = input_conn_arr;
    }

    addOutputConnection(output_conn_arr){
        this.output_conn_arr = output_conn_arr;
    }

    evaluate(input){console.log(this.activation)
        return this.activation(input);
    }
}
module.exports = NodeGene;