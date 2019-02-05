"use strict"

const tf = require('@tensorflow/tfjs-node')
const express = require('express');
const {createCanvas, createImageData} = require('canvas');
const Genome = require('./genome');

/**Express props */
const port = 9995;
/*Thumbnail props*/
const height = 120;
const width = 136;
/*Genome Props*/
let num_hidden_neurons = 8;
let num_output = 1;
let z_dim = 8; //Latent vec
/*Montage props*/
const montage_size = 25; //Always use a proper square

function main() {
    const app = express();
    app.listen(process.env.PORT || port, () => console.log(`Example app listening on port ${port}!`));
    const canvas = createCanvas(width*Math.sqrt(montage_size), height*Math.sqrt(montage_size));
    canvas.width = width*Math.sqrt(montage_size);
    canvas.height = height*Math.sqrt(montage_size);
    var ctx = canvas.getContext('2d');
    
    app.get('/', function (req, res, next) {
        res.sendFile(__dirname+'/views/index.html');
    });

    app.get('/montage', function (req, res, next) {
        console.log(req.query.color)
        if(req.query.color == 'true') {
            num_output = 3;
        }else{
            num_output = 1;
        }
        if(req.query.dense =='true') { 
            num_hidden_neurons = 64; 
            z_dim=32; 
        }else{
            num_hidden_neurons = 32; 
            z_dim=16; 
        }
        res.on('finish', () => { console.log('Response sent.') });
        let dataArr = [];
        for(var m_ind=0;m_ind<montage_size;m_ind++){
            let t_x = tf.div(tf.div(tf.mul(tf.sub(tf.range(0, width), (width - 1) / 2.0), 1.0), (width - 1)), 0.5);
            t_x = t_x.reshape([1, width]);
            t_x = tf.matMul(tf.ones([height, 1], 'float32'), t_x);
            t_x = t_x.flatten().reshape([height * width, 1])
    
            let t_y = tf.div(tf.div(tf.mul(tf.sub(tf.range(0, height), (height - 1) / 2.0), 1.0), (height - 1)), 0.5);
            t_y = t_y.reshape([height, 1]);
            t_y = tf.matMul(t_y, tf.ones([1, width], 'float32'));
            t_y = t_y.flatten().reshape([height * width, 1])
    
            /*Initialize r*/
            let t_r = tf.sqrt(tf.add(tf.pow(t_x, 2), tf.pow(t_y, 2)));
            t_r = tf.transpose(t_r).flatten().reshape([height * width, 1])
    
            /*Initialize z*/
            let z = tf.randomUniform([1, z_dim], -1.0, 1.0, 'float32');
            let t_z = z.reshape([1, z_dim]);
            t_z = tf.mul(t_z, tf.mul(tf.ones([height * width, 1], 'float32'), 1));
            t_z = (t_z).flatten().reshape([height * width, z_dim])
    
            
            var genome = new Genome(1, num_output, num_hidden_neurons);
            genome.initialNetwork(); 
            var outputs = genome.evaluate([t_x, t_y, t_r, t_z]);
            var node_genes = outputs.node_genes;
            var connection_genes = outputs.connection_genes;
            var data = outputs.output;
            if(num_output==3) data = data.reshape([data.shape[0]*3])//RGB Channels

            
            //Duplicated from Tensorflow array_ops.toPixels(). That method does not work with await.
            data = data.dataSync();
            var bytes = new Array(width * height * 4); //Convert to Uint8 array on the UI
            var r,g,b,a,j,i;
            for (i = 0; i < height * width; ++i) {
                r = void 0, g = void 0, b = void 0, a = void 0;
                if(num_output == 3){
                    r = data[i * 3] * 255;
                    g = data[i * 3 + 1] * 255;
                    b = data[i * 3 + 2] * 255;
                }else{
                    r = data[i] * 255;
                    g = data[i] * 255;
                    b = data[i] * 255;
                }
                a = 255;
                j = i * 4;
                bytes[j + 0] = Math.round(r);
                bytes[j + 1] = Math.round(g);
                bytes[j + 2] = Math.round(b);
                bytes[j + 3] = Math.round(a);
            }
            dataArr.push({data:bytes, node_genes:node_genes, connection_genes:connection_genes});
        }
        res.send(dataArr);
    });
}
main();
