"use strict"

const tf = require('@tensorflow/tfjs-node');

class activationutil{

    static gaussian(d){ //Assuming mean=0.0 and stddev=1.0
        var g_std_dev = 2.0;
        var g_mean = 0.0;
        return tf.div(tf.pow(Math.E, tf.div(tf.pow(tf.div(tf.sub(d, g_mean), g_std_dev),2),-2.0)),Math.sqrt(g_std_dev*2*Math.PI));
    }

    static sine(d){
        return tf.add(tf.mul(tf.mul(tf.sin(d),0.5), -0.25),0.2);
    }

    static psychdelic(d){
        return tf.add(tf.mul(tf.sin(d),0.5),0.5);
    }

    static tanh(d){
        return tf.abs(tf.mul(tf.tanh(d),0.5));
    }

    static enum_activations(ind){
        return [tf.tanh, tf.softplus, tf.sin, activationutil.gaussian, tf.sigmoid][ind];
        // return tf.tanh;
    }

    static enum_final_activations(ind){
        return [activationutil.tanh, activationutil.sine, activationutil.gaussian, tf.sigmoid, activationutil.psychdelic][ind];
        // return tf.sigmoid;
    }

    static random_activation(){
        return activationutil.enum_activations(Math.floor(Math.random()*(5)));
    }

    static random_final_activation(){
        return activationutil.enum_final_activations(Math.floor(Math.random()*(5)));
    }
}
module.exports = activationutil;