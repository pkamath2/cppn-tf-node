"use strict"

const tf = require('@tensorflow/tfjs-node');
const NodeGene = require('./nodegene');
const ConnectionGene = require('./connectiongene');
const util = require('./activationsutil');

class Genome{
    constructor(id, final_output_size, init_num_hidden_neurons){
        this.id = id;
        this.final_output_size = final_output_size;
        this.init_num_hidden_neurons = init_num_hidden_neurons;

        this.global_node_innovation_number = 0; //Initialize global innovation number. In this version of CPPN, we will be creating Genome exactly once only. Subsequent only mutations occur. 
        this.global_conn_innovation_number = 0; //Initialize global innovation number. In this version of CPPN, we will be creating Genome exactly once only. Subsequent only mutations occur. 
        this.node_gene_arr = [];
        this.connection_gene_arr = [];

        //Maintaining Map as well. Check which one is better. 
        this.node_gene_map = new Map();
        this.connection_gene_map = new Map();
    }

    incr_node_innovation_number(){
        return this.global_node_innovation_number++;
    }

    incr_conn_innovation_number(){
        return this.global_conn_innovation_number++;
    }

    addNode(input_conn_arr, output_conn_arr, is_hidden, name){
        var activation = (name=='output'?util.random_final_activation():util.random_activation());
        var nodeGene = new NodeGene(this.incr_node_innovation_number(), name, activation, is_hidden, input_conn_arr, output_conn_arr)
        this.node_gene_arr.push(nodeGene);
        this.node_gene_map.set(nodeGene.innovation_number, nodeGene);
        return nodeGene;
    }

    addConnection(from_node_id, to_node_id){
        var connectionGene = new ConnectionGene(this.incr_conn_innovation_number(), Math.random(), from_node_id, to_node_id, true);
        this.connection_gene_arr.push(connectionGene);
        this.connection_gene_map.set(connectionGene.innovation_number, connectionGene);
        return connectionGene;
    }

    initialNetwork(){
        //Create input nodes
        for(var i=0;i<4;i++){ // 4 inputs - [t_x, t_y, t_r, t_z]
            this.addNode(null, null, false, 'input_'+i);
        }

        //Create few hidden nodes
        this.addNode(null, null, true, 'hidden_1');
        this.addNode(null, null, true, 'hidden_2');
        this.addNode(null, null, true, 'hidden_3');
        //Create output node
        this.addNode(null, null, false, 'output');

        //Randomly connect nodes
        for(var i=0;i<this.node_gene_arr.length;i++){
            var node_i = this.node_gene_arr[i];
            for(var j=0;j<this.node_gene_arr.length;j++){
                var node_j = this.node_gene_arr[j];
                if(Math.random() > 0.4){ //Add output nodes only if there are some inputs.
                    if(node_i.name.indexOf('input') > -1 && node_j.name.indexOf('input') > -1){
                        continue;//Do not connect input to input
                    }
                    if(node_j.innovation_number > node_i.innovation_number){
                        var connection = this.addConnection(node_i.innovation_number, node_j.innovation_number);
                        if(node_j.name == 'output') connection.is_output_connection = true;
                        node_i.output_conn_arr.push(connection.innovation_number);
                        node_j.input_conn_arr.push(connection.innovation_number);
                    }
                }
            }

            //0.0. If there are no inputs to a node, disable the node. Also disable all outgoing connections (if any). 
            if(node_i.name.indexOf('hidden')>-1 && node_i.input_conn_arr.length==0) {
                node_i.disabled = true;
                var output_connections = node_i.output_conn_arr;
                if(output_connections && output_connections.length>0){
                    output_connections.forEach(conn_id => {
                        var conn = this.connection_gene_map.get(conn_id);
                        conn.disabled = true;
                    });
                }
            }

            //0.1. If the output node has no incoming connections, atleast feed it t_r.
            if(node_i.name=='output' && node_i.input_conn_arr.length==0){
                //TODO 
                // var t_r =  this.node_gene_map.get()
                // var connection = this.addConnection(node_i.innovation_number, node_j.innovation_number);
            }
        }
        // console.log(this.node_gene_arr)
        // console.log(this.connection_gene_arr)
    }

    evaluate(inputs){
        var node_outputs_map = new Map();
        var connection_outputs_map = new Map();
        var result_node_id = 0;

        //1. Iterate through all the nodes
        for(var i=0;i<this.node_gene_arr.length;i++){
            var curr_node = this.node_gene_arr[i];

            if(curr_node.disabled){
                continue;
            }

            var input = null;
            if(curr_node.name.indexOf('input') > -1){
                input = inputs[curr_node.name.substring('input_'.length)];
                node_outputs_map.set(curr_node.innovation_number, curr_node.evaluate(input));
            }else{
                try {
                    //2. Sum all the inputs
                    var input_conn_arr = curr_node.input_conn_arr;
                    var connInput = null;//tricky
                    for(var j=0;j<input_conn_arr.length;j++){
                        var connection = this.connection_gene_map.get(input_conn_arr[j]);
                        if(connection.disabled){
                            continue;
                        }
                        if(connInput == null) connInput = tf.zeros(connection_outputs_map.get(input_conn_arr[j]).shape);
                        connInput = tf.add(connInput, connection_outputs_map.get(input_conn_arr[j]));
                    }
                    //3. Evaluate the node
                    node_outputs_map.set(curr_node.innovation_number, curr_node.evaluate(connInput));
                } catch (error) {
                    console.log(curr_node)
                    throw error;
                }
                
            }
            
            //4. Calculate all the output connections.
            var output_conn_arr = curr_node.output_conn_arr;
            for(var j=0;j<output_conn_arr.length;j++){
                try {
                    var connection = this.connection_gene_map.get(output_conn_arr[j]);
                    if(connection.disabled){
                        continue;
                    }
                    var conn_output = connection.evaluate(node_outputs_map.get(curr_node.innovation_number), (connection.is_output_connection?this.final_output_size:this.init_num_hidden_neurons));
                    connection_outputs_map.set(connection.innovation_number, conn_output);
                } catch (error) {
                    console.log(curr_node)
                    throw error;
                }
            }

            if(curr_node.name.indexOf('output') > -1){
                result_node_id = curr_node.innovation_number;
            }
        }
        return {output:node_outputs_map.get(result_node_id), node_genes: this.node_gene_arr, connection_genes: this.connection_gene_arr};
    }
}
module.exports = Genome;