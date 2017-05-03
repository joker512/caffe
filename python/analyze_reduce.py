#!/usr/bin/python2

import os
os.environ['GLOG_minloglevel'] = '1'
import caffe.proto.caffe_pb2 as pb2
import google.protobuf.text_format as tf
import argparse
import operator
import math
import subprocess

subprocess.call(["tabs", "20"])
parser = argparse.ArgumentParser(description='Reduce level calculator')
parser.add_argument('model', help='model.prototxt')
parser.add_argument('config', help ='quantize configuration file in format: "layer k m"', nargs='?')
args = parser.parse_args()

net_params = pb2.NetParameter()
with open(args.model) as f:
    tf.Merge(f.read(), net_params)

q_layer_k_m = {}
if args.config:
    with open(args.config) as f:
        for line in f.readlines():
            if not line.startswith('#'):
                name, k, m = line.split()
                k, m = int(k), int(m)
                q_layer_k_m[name] = (k, m)

total_size_src = 0
total_size_q = 0
channels_from = {}
channels_from['data'] = 3
convolutions_src = []
convolutions_q = {}
for layer in net_params.layer:
    if layer.type == 'Convolution':
        param = layer.convolution_param
        in_channels = channels_from[layer.bottom[0]]
        out_channels = int(param.num_output)
        kernel_size = int(param.kernel_size[0])

        layer_size_src = 4 * in_channels * out_channels * kernel_size**2 + out_channels
        if layer.name in q_layer_k_m:
            k, m = q_layer_k_m[layer.name]
            layer_size_q_d = 4 * in_channels * k
            layer_size_q_b = kernel_size**2 * in_channels * out_channels * int(math.log(k, 2)) / (m * 8)
            layer_size_q = layer_size_q_d + layer_size_q_b + out_channels
            convolutions_q[layer.name] = (layer_size_q_d, layer_size_q_b, layer_size_q)
            total_size_q += layer_size_q
        else:
            total_size_q += layer_size_src

        convolutions_src.append((layer.name, in_channels, out_channels, kernel_size, layer_size_src))
        total_size_src += layer_size_src
        channels_from[layer.top[0]] = out_channels
    elif layer.type == 'Concat':
        channels_from[layer.top[0]] = sum([channels_from[channel] for channel in layer.bottom])
    elif len(layer.bottom) > 0:
        channels_from[layer.top[0]] = channels_from[layer.bottom[0]]

cum_percent_src = 0.
cum_percent_q = 0.
num = 0
for (name, in_channels, out_channels, kernel_size, layer_size_src) in \
        sorted(convolutions_src, key=operator.itemgetter(4), reverse=True):
    percent_src = 100 * float(layer_size_src) / total_size_src
    cum_percent_src += percent_src
    num += 1
    print('Size for layer %s (%d)\tformula\t\tsize (bytes)\tportion\ttotal portion\tcompression level' % (name, num))
    print('\twithout quantize:\t4 * %d * %d * %d^2 + %d\t= %d\t%.2f %%\t%.2f %%' % \
        (in_channels, out_channels, kernel_size, out_channels, layer_size_src, percent_src, cum_percent_src))
    if name in convolutions_q:
        k, m = q_layer_k_m[name]
        layer_size_q_d, layer_size_q_b, layer_size_q = convolutions_q[name]
        percent_q_d = 100 * float(layer_size_q_d) / total_size_src
        percent_q_b = 100 * float(layer_size_q_b) / total_size_src
        percent_q = percent_q_d + percent_q_b
        print('\twith quantize (d):\t4 * %d * %d\t\t= %d\t%.2f %%\t\t%.2f %%' % \
            (in_channels, k, layer_size_q_d, percent_q_d, 
                100 * float(layer_size_q_d) / layer_size_src))
        print('\twith quantize (b):\t1/8 * %d^2 * %d * %d * log2(%d) / %d\t= %d\t%.2f %%\t\t%.2f %%' % \
            (kernel_size, in_channels, out_channels, k, m, layer_size_q_b, percent_q_b, \
                100 * float(layer_size_q_b) / layer_size_src))
        cum_percent_q += percent_q
        compression_level = float(layer_size_src) / layer_size_q
        print('\twith quantize:\t%d + %d + %d\t\t= %d\t%.2f %%\t%.2f %%\t%.2f %% (%.2f)' % \
            (layer_size_q_d, layer_size_q_b, out_channels, layer_size_q, percent_q, cum_percent_q, \
                100 / compression_level, compression_level))
    else:
        cum_percent_q += percent_src
        print('\tnot quantized:\t4 * %d * %d * %d^2 + %d\t= %d\t%.2f %%\t%.2f %%' % \
            (in_channels, out_channels, kernel_size, out_channels, layer_size_src, percent_src, cum_percent_q))
    print

total_compression_level = float(total_size_src) / total_size_q
print('Total conv weights size without quantize:\t\t= %d\t\t%.2f %%' % (total_size_src, 100))
print('Total conv weights size with quantize:\t\t\t= %d\t\t%.2f %%\t%.2f' % \
        (total_size_q, 100 / total_compression_level, total_compression_level))
