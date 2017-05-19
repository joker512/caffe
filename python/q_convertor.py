#!/usr/bin/python2

import os
os.environ['GLOG_minloglevel'] = '1'
import caffe
import caffe.proto.caffe_pb2 as pb2
import google.protobuf.text_format as tf
import extract_weights as ew
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Convert conv and fc layers to quantized format')
parser.add_argument('-C', '--newmodel', help='file to save new prototxt')
parser.add_argument('-W', '--newweights', help='file to save new weights')
parser.add_argument('-P', '--srcmodel', help='optional config to load model with source weights for layer')
parser.add_argument('-S', '--srcweights', help='optional file to load source weights for layer')
parser.add_argument('-F', '--force', help='recompress all layers, including already quantized', action='store_true')
parser.add_argument('model', help='model.prototxt')
parser.add_argument('weights', help='weights.caffemodel')
parser.add_argument('config', help ='quantize configuration file in format: "layer k m"')

args = parser.parse_args()

print('Start model converting')
net = caffe.Net(network_file=args.model, phase=caffe.TEST, weights=args.weights)
if args.srcweights:
    src_net = caffe.Net(network_file=args.srcmodel, phase=caffe.TEST, weights=args.srcweights)
net_params = pb2.NetParameter()
with open(args.model) as f:
    tf.Merge(f.read(), net_params)
print('Model has been loaded')

q_layer_k_m = []
if args.config:
    with open(args.config) as f:
        for line in f.readlines():
            if not line.startswith('#'):
                name, k, m = line.split()
                k, m = int(k), int(m)
                q_layer_k_m.append((name, k, m))

for layer_name, K, M in q_layer_k_m:
    weights = net.params[layer_name][0].data if not src_net else src_net.params[layer_name][0].data
    # TODO: D calculation doesn't need to depend from fc or conv type
    is_fc = False
    i = next(pos for pos, e in enumerate(net._layer_names) if e == layer_name)
    index, layer = next((j, e) for j, e in enumerate(net_params.layer) if e.name == layer_name)
    if net.layers[i].type == 'Convolution':
        param = layer.convolution_param
        if param.engine == pb2.ConvolutionParameter.Engine.Value('QUANT') and param.k == K and param.m == M and not args.force:
            print('Skipping layer %s which is already quantized.' % layer_name)
            continue
        else:
            print('Start processing layer %s (%d, %d)' % (layer_name, K, M))
        param.engine = pb2.ConvolutionParameter.Engine.Value('QUANT')
        param.ClearField('bias_filler')
        param.ClearField('weight_filler')

        in_channels = weights.shape[1]
        weights = weights.transpose((0, 2, 3, 1)).reshape((-1, in_channels))
    elif net.layers[i].type == 'InnerProduct':
        is_fc = True
        layer.type = pb2.V1LayerParameter.LayerType.Value('INNER_PRODUCT_Q')
        # TODO: type change is impossible
        # need to make engine like for conv layer or save-load-save .caffemodel
        # net.layers[i].type = 'InnerProductQ'
        param = layer.inner_product_q_param
        param.num_output = layer.inner_product_param.num_output
        layer.ClearField('inner_product_param')
    else:
        raise ValueError('%s layer %s type is unknown' % (layer_name, net.layers[i].type))

    if not layer.top[0].endswith("_"):
        layer.top[0] += "_"
        net_params.layer[index + 1].bottom[0] += "_"
    if hasattr(layer, 'blobs_lr'):
        layer.ClearField('blobs_lr')
    if hasattr(layer, 'weight_decay'):
        layer.ClearField('weight_decay')

    param.k = K
    param.m = M
    print('\tWeights have been extracted')

    D, B = ew.calcBD(weights, K, M, is_fc)
    print('\tD and B have been calced')
    #D = ew.codeD(D)
    B = ew.codeB(B, K)
    print('\tD and B has been coded')

    net.params[layer_name][0].reshape(*D.shape)
    net.params[layer_name][0].data[...] = D
    if len(net.params[layer_name]) == 2:
        net.params[layer_name].add_blob()
    net.params[layer_name][2].reshape(*B.shape)
    net.params[layer_name][2].data[...] = B
    print('\tD and B have been saved to layer %s' % layer_name)

new_weights = args.newweights if args.newweights else args.weights
new_model = args.newmodel if args.newmodel else args.model
with open(new_model, 'w') as f:
    f.write(tf.MessageToString(net_params))
print('Model has been saved to %s' % new_model)

net.save(new_weights)
print('Weights have been saved to %s' % new_weights)
