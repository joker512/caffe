#ifndef CAFFE_CONV_Q_UNPACK_LAYER_HPP_
#define CAFFE_CONV_Q_UNPACK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class ConvolutionQUnpackLayer : public ConvolutionLayer<Dtype> {
public:
	explicit ConvolutionQUnpackLayer(const LayerParameter& param) : ConvolutionLayer<Dtype>(param) {}
	virtual inline const char* type() const { return "Convolution"; }

protected:
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual inline bool reverse_dimensions() { return false; }

	Blob<Dtype> weights;
};

}

#endif  // CAFFE_CONV_Q_UNPACK_LAYER_HPP_
