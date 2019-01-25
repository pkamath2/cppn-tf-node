const tf = require('@tensorflow/tfjs-node');
const { createCanvas, createImageData } = require('canvas')
const express = require('express')

/**Express props */
const port = 9997;
/*Canvas props*/
const height = 980;
const width = 1000;

/*ML props */
const num_hidden_neurons = 32;
const num_output = 3;
const z_dim = 32; //Latent vec
// const config = {scale:0.3, activations:[tf.tanh, tf.tanh, tf.tanh, tf.tanh, tf.sigmoid]};
const config = {scale:0.3, activations:[tf.tanh,tf.tanh, tf.tanh, tf.tanh, gaussian]};//Colors look good.
// const config = {scale:1, activations:[tf.tanh, tf.softplus, tf.tanh, tf.softplus, (d)=>tf.add(tf.mul(tf.sin(d),0.5),0.5)]};
// const config = {scale:0.01, activations:[tf.tanh, tf.tanh, tf.tanh, (d)=>tf.add(tf.mul(tf.sin(d),0.5),0.5)]};//Psychedelic colors.

//For neurogram
// const config = {scale:0.0001, activations:[(d)=>tf.add(tf.mul(tf.sin(d),0.5),0.5)]};
// const config = {scale:0.0001, activations:[(d)=>tf.abs(tf.mul(tf.tanh(d),0.5))]};
// const config = {scale:0.0001, activations:[gaussian]};
// const config = {scale:0.0001, activations:[tf.sigmoid]};

function model(input_shape, n_output, with_bias) {
    const input = tf.input({ shape: [input_shape[0], input_shape[1]] })
    var dense = input;
    dense = tf.layers.dense({
        units: n_output,
        useBias: with_bias,
        kernelInitializer: new tf.initializers.randomNormal({ mean: 0.0, stddev: 2.0 }),
        biasInitializer: with_bias?new tf.initializers.randomNormal({ mean: 0.0, stddev: 2.0 }):null
    }).apply(dense);
    var m = tf.model({ inputs: input, outputs: dense });
    return m;
}

function getImageArray() {
    let t_x = tf.div(tf.div(tf.mul(tf.sub(tf.range(0, width), (width - 1) / 2.0), 1.0), (width - 1)), 0.5);
    t_x = t_x.reshape([1, width]);
    t_x = tf.matMul(tf.ones([height, 1], 'float32'), t_x);
    t_x = t_x.flatten().reshape([height * width, 1])

    let t_y = tf.div(tf.div(tf.mul(tf.sub(tf.range(0, height), (height - 1) / 2.0), 1.0), (height - 1)), 0.5);
    t_y = t_y.reshape([height, 1]);
    t_y = tf.matMul(t_y, tf.ones([1, width], 'float32'));
    t_y = t_y.flatten().reshape([height * width, 1])

    /*Initialize r*/
    t_r = tf.sqrt(tf.add(tf.pow(t_x, 2), tf.pow(t_y, 2)));
    t_r = tf.transpose(t_r).flatten().reshape([height * width, 1])

    /*Initialize z*/
    let z = tf.randomUniform([1, z_dim], -10.0, 10.0, 'float32');
    let t_z = z.reshape([1, z_dim]);
    t_z = tf.mul(t_z, tf.mul(tf.ones([height * width, 1], 'float32'), config.scale));
    t_z = (t_z).flatten().reshape([height * width, z_dim])

    // t_x = tf.sin(tf.mul(t_x, 10)) //Repitition/Periodic
    // t_y = tf.sin(tf.mul(t_y, 10))

    t_x = model([height * width, 1], num_hidden_neurons, false).predict(t_x.reshape([1,t_x.shape[0], t_x.shape[1]]));
    t_y = model([height * width, 1], num_hidden_neurons, false).predict(t_y.reshape([1,t_y.shape[0], t_y.shape[1]]));
    t_r = model([height * width, 1], num_hidden_neurons, false).predict(t_r.reshape([1,t_r.shape[0], t_r.shape[1]]));

    t_z = model([height * width, z_dim], num_hidden_neurons, true).predict(t_z.reshape([1,t_z.shape[0], t_z.shape[1]]));

    
    let t_final = tf.add(tf.add(tf.add(t_z, t_x), t_y), t_r);
    if(config.activations.length>1)t_final = config.activations[0](t_final);
    for(var a=1;a<config.activations.length-1;a++){
        t_final = config.activations[a](model([t_final.shape[1], t_final.shape[2]], num_hidden_neurons, false).predict(t_final));
    }
    t_final = config.activations[config.activations.length-1](model([t_final.shape[1], t_final.shape[2]], num_output, false).predict(t_final));

    console.log(t_final.shape)
    if(num_output==3) t_final = t_final.reshape([t_final.shape[1],1,3])//RGB Channels
    return t_final;
}

function main() {
    const app = express();
    app.listen(port, () => console.log(`Example app listening on port ${port}!`));
    const canvas = createCanvas(width, height);

    app.get('/', function (req, res, next) {
        var data = tf.tidy(getImageArray);
        tf.toPixels(data).then((data_uint8) => {
            canvas.width = width;
            canvas.height = height;
            ctx = canvas.getContext('2d');
            imageData = createImageData(data_uint8, width, height);
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

function gaussian(d){ //Assuming mean=0.0 and stddev=1.0
    return tf.div(tf.pow(Math.E, tf.div(tf.pow(d,2),-2.0)),Math.sqrt(2*Math.PI));
}