#!/usr/bin/python2

import os
os.environ['GLOG_minloglevel'] = '1'
import caffe.proto.caffe_pb2 as pb2
import google.protobuf.text_format as tf
import argparse
import operator
import math
import subprocess

subprocess.call(["tabs", "22"])
parser = argparse.ArgumentParser(description='Reduce level calculator')
parser.add_argument('-C', '--channels', type=int, help='number of channels of source image')
parser.add_argument('-W', '--width', type=int, help='width of source image')
parser.add_argument('-H', '--height', type=int, help='height of source image')
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

total_operations_src = 0
total_operations_q = 0
channels_from = {}
channels_from['data'] = args.channels if args.channels else 3
width_from = {}
width_from['data'] = args.width if args.width else 191
height_from = {}
height_from['data'] = args.height if args.height else 271
convolutions_src = []
convolutions_q = {}
inner_products_src = []
inner_products_q = {}
for layer in net_params.layer:
    if layer.type == 'Convolution':
        param = layer.convolution_param
        in_height = height_from[layer.bottom[0]]
        in_width = width_from[layer.bottom[0]]
        in_channels = channels_from[layer.bottom[0]]
        out_channels = int(param.num_output)
        kernel_size = int(param.kernel_size[0])
        pad = int(param.pad[0]) if len(param.pad) > 0 else 0
        stride = int(param.stride[0]) if len(param.stride) > 0 else 1
        out_height = (in_height - kernel_size + 2 * pad) / stride + 1
        out_width = (in_width - kernel_size + 2 * pad) / stride + 1

        if layer.name in q_layer_k_m:
            k, m = q_layer_k_m[layer.name]
            layer_operations_q_d = in_width * in_height * in_channels * k
            layer_operations_q_b = out_width * out_height * out_channels * kernel_size * kernel_size * in_channels / m
            layer_operations_q = layer_operations_q_d + layer_operations_q_b
            convolutions_q[layer.name] = (layer_operations_q_d, layer_operations_q_b, layer_operations_q)
            total_operations_q += layer_operations_q

            layer_operations_src = in_channels * out_channels * kernel_size * kernel_size * out_width * out_height
            convolutions_src.append((layer.name, in_channels, out_channels, kernel_size, in_height, in_width, out_height, out_width, layer_operations_src))
            total_operations_src += layer_operations_src

        channels_from[layer.top[0]] = out_channels
        height_from[layer.top[0]] = out_height
        width_from[layer.top[0]] = out_width
    elif layer.type == 'InnerProduct':
        param = layer.inner_product_param
        height = height_from[layer.bottom[0]]
        width = width_from[layer.bottom[0]]
        in_channels = channels_from[layer.bottom[0]]
        num_input = in_channels * height * width
        num_output = int(param.num_output)

        layer_operations_src = num_input * num_output
        if layer.name in q_layer_k_m:
            k, m = q_layer_k_m[layer.name]
            layer_operations_q_d = num_input * k
            layer_operations_q_b = num_output * num_input / m
            layer_operations_q = layer_operations_q_d + layer_operations_q_b
            inner_products_q[layer.name] = (layer_operations_q_d, layer_operations_q_b, layer_operations_q)
            total_operations_q += layer_operations_q
        else:
            total_operations_q += layer_operations_src

        inner_products_src.append((layer.name, num_input, num_output, layer_operations_src))
        total_operations_src += layer_operations_src
        channels_from[layer.top[0]] = num_output
        height_from[layer.top[0]] = 1
        width_from[layer.top[0]] = 1
    elif layer.type == 'Pooling':
        param = layer.pooling_param
        height = height_from[layer.bottom[0]]
        width = width_from[layer.bottom[0]]
        kernel_size = param.kernel_size
        pad = param.pad
        stride = param.stride

        channels_from[layer.top[0]] = channels_from[layer.bottom[0]]
        height_from[layer.top[0]] = (height - kernel_size + 2 * pad + stride - 1) / stride + 1
        width_from[layer.top[0]] = (width - kernel_size + 2 * pad + stride - 1) / stride + 1
    elif layer.type == 'Concat':
        channels_from[layer.top[0]] = sum([channels_from[channel] for channel in layer.bottom])
        height_from[layer.top[0]] = height_from[layer.bottom[0]]
        width_from[layer.top[0]] = width_from[layer.bottom[0]]
    elif len(layer.bottom) > 0:
        channels_from[layer.top[0]] = channels_from[layer.bottom[0]]
        height_from[layer.top[0]] = height_from[layer.bottom[0]]
        width_from[layer.top[0]] = width_from[layer.bottom[0]]

cum_percent_src = 0.
cum_percent_q = 0.
num = 0
for (name, in_channels, out_channels, kernel_size, in_height, in_width, out_height, out_width, layer_operations_src) in \
        sorted(convolutions_src, key=operator.itemgetter(8), reverse=True):
    percent_src = 100 * float(layer_operations_src) / total_operations_src
    cum_percent_src += percent_src
    num += 1
    print('Operations for layer %s (%d)\tformula\t\toperations\tportion\ttotal portion\tacceleration level' % (name, num))
    print('\twithout quantize:\t%d * %d * %d * %d^2 * %d           \t= %d\t%.2f %%\t%.2f %%' % \
        (out_height, out_width, out_channels, kernel_size, in_channels, layer_operations_src, percent_src, cum_percent_src))
    if name in convolutions_q:
        k, m = q_layer_k_m[name]
        layer_operations_q_d, layer_operations_q_b, layer_operations_q = convolutions_q[name]
        percent_q_d = 100 * float(layer_operations_q_d) / total_operations_src
        percent_q_b = 100 * float(layer_operations_q_b) / total_operations_src
        percent_q = percent_q_d + percent_q_b
        print('\twith quantize (d):\t%d * %d * %d * %d               \t= %d\t%.2f %%\t\t%.2f %%' % \
            (in_height, in_width, in_channels, k, layer_operations_q_d, percent_q_d,
                100 * float(layer_operations_q_d) / layer_operations_src))
        print('\twith quantize (b):\t%d * %d * %d * %d^2 * %d / %d\t= %d\t%.2f %%\t\t%.2f %%' % \
            (out_height, out_width, out_channels, kernel_size, in_channels, m, layer_operations_q_b, percent_q_b, \
                100 * float(layer_operations_q_b) / layer_operations_src))
        cum_percent_q += percent_q
        acceleration_level = float(layer_operations_src) / layer_operations_q
        print('\twith quantize:\t%d + %d\t\t= %d\t%.2f %%\t%.2f %%\t%.2f %% (%.2f)' % \
            (layer_operations_q_d, layer_operations_q_b, layer_operations_q, percent_q, cum_percent_q, \
                100 / acceleration_level, acceleration_level))
    else:
        cum_percent_q += percent_src
        print('\tnot quantized:\t%d * %d * %d * %d^2 * %d           \t= %d\t%.2f %%\t%.2f %%' % \
            (out_height, out_width, out_channels, kernel_size, in_channels, layer_operations_src, percent_src, cum_percent_src))
    print

for (name, num_input, num_output, layer_operations_src) in \
        sorted(inner_products_src, key=operator.itemgetter(3), reverse=True):
    percent_src = 100 * float(layer_operations_src) / total_operations_src
    cum_percent_src += percent_src
    num += 1
    print('Operations for layer %s (%d)\tformula\t\toperations\tportion\ttotal portion\tacceleration level' % (name, num))
    print('\twithout quantize:\t%d * %d        \t= %d\t%.2f %%\t%.2f %%' % \
            (num_input, num_output, layer_operations_src, percent_src, cum_percent_src))
    if name in inner_products_q:
        k, m = q_layer_k_m[name]
        layer_operations_q_d, layer_operations_q_b, layer_operations_q = inner_products_q[name]
        percent_q_d = 100 * float(layer_operations_q_d) / total_operations_src
        percent_q_b = 100 * float(layer_operations_q_b) / total_operations_src
        percent_q = percent_q_d + percent_q_b
        print('\twith quantize (d):\t%d * %d\t\t= %d\t%.2f %%\t\t%.2f %%' % \
            (num_input, k, layer_operations_q_d, percent_q_d,
                100 * float(layer_operations_q_d) / layer_operations_src))
        print('\twith quantize (b):\t%d * %d / %d\t= %d\t%.2f %%\t\t%.2f %%' % \
            (num_output, num_input, m, layer_operations_q_b, percent_q_b, \
                100 * float(layer_operations_q_b) / layer_operations_src))
        cum_percent_q += percent_q
        acceleration_level = float(layer_operations_src) / layer_operations_q
        print('\twith quantize:\t%d + %d     \t= %d\t%.2f %%\t%.2f %%\t%.2f %% (%.2f)' % \
            (layer_operations_q_d, layer_operations_q_b, layer_operations_q, percent_q, cum_percent_q, \
                100 / acceleration_level, acceleration_level))
    else:
        cum_percent_q += percent_src
        print('\twithout quantize:\t%d * %d        \t= %d\t%.2f %%\t%.2f %%' % \
                (num_input, num_output, layer_operations_src, percent_src, cum_percent_src))
    print

total_acceleration_level = float(total_operations_src) / total_operations_q
print('Total operations without quantize:\t\t\t= %d\t\t%.2f %%' % (total_operations_src, 100))
print('Total operations with quantize:\t\t\t= %d\t\t%.2f %%\t%.2f' % \
        (total_operations_q, 100 / total_acceleration_level, total_acceleration_level))
