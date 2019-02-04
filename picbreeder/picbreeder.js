"use strict"

const tf = require('@tensorflow/tfjs-node')
const express = require('express');
const {createCanvas, createImageData} = require('canvas');
const Genome = require('./genome');

/**Express props */
const port = 9995;
/*Canvas props*/
const height = 500;
const width = 520;

const num_hidden_neurons = 64;
const num_output = 3;
const z_dim = 32; //Latent vec

function main() {
    const app = express();
    app.listen(port, () => console.log(`Example app listening on port ${port}!`));
    const canvas = createCanvas(width, height);

    app.get('/', function (req, res, next) {

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
        genome.initialNetwork(); //[t_x, t_y, t_r, t_z]
        var data = genome.evaluate([t_x, t_y, t_r, t_z]);
        if(num_output==3) data = data.reshape([data.shape[0],1,3])//RGB Channels

        tf.toPixels(data).then((data_uint8) => {
            canvas.width = width;
            canvas.height = height;
            var ctx = canvas.getContext('2d');
            var imageData = createImageData(data_uint8, width, height);
            ctx.putImageData(imageData, 0, 0);
            const stream = canvas.createPNGStream();
            stream.pipe(res)
            res.on('finish', () => { console.log('Response sent.') });
        }).catch(function (error) {
            console.log(error);
        });
        
    });
}
main();
