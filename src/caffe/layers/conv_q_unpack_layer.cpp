#include <vector>

#include "caffe/layers/conv_q_unpack_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void ConvolutionQUnpackLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
		Blob<Dtype> D_;
		Blob<int> B_;

		const int K = this->layer_param_.convolution_param().k();
		const int M = this->layer_param_.convolution_param().m();
		const int BITS = (int)log2(K);
		const int TOTAL_BITS = 32;
		const int REST_BITS = TOTAL_BITS - BITS;
		const int D_DIVIDER = 10000;

		const int kernel_h = this->kernel_shape_.cpu_data()[0];
		const int kernel_w = this->kernel_shape_.cpu_data()[1];
		const int kernel_dim_ = kernel_h * kernel_w;
		const int b_shape_size = this->channels_ / M * kernel_dim_ * this->num_output_;
		if (this->blobs_.size() < 3) {
			this->blobs_.resize(3);
			vector<int> d_shape(2);
			d_shape[0] = K * this->channels_ / M / 2;
			d_shape[1] = M;
			this->blobs_[0].reset(new Blob<Dtype>(d_shape));
			vector<int> b_binary_shape(1, BITS * b_shape_size / (8 * sizeof(Dtype)) + (BITS * b_shape_size % (8 * sizeof(Dtype)) ? 1 : 0));
			this->blobs_[2].reset(new Blob<Dtype>(b_binary_shape));
		}
		else {
			if (B_.count() == 0) {
				vector<int> b_shape(1, b_shape_size);
				B_.Reshape(b_shape);
			}
			unsigned int* B_hash = (unsigned int*)(this->blobs_[2]->cpu_data());
			int* B = B_.mutable_cpu_data();
			for (int i = 0, total_bit_shift = 0; i < b_shape_size; ++i, total_bit_shift += BITS) {
				int byte_shift = total_bit_shift / TOTAL_BITS;
				int bit_shift = total_bit_shift % TOTAL_BITS;
				int shift = REST_BITS - bit_shift;
				B[i] = (shift < 0 ? B_hash[byte_shift] << -shift | B_hash[byte_shift + 1] >> (TOTAL_BITS + shift) :
										  B_hash[byte_shift] >> shift) & (K - 1);
			}

			if (D_.count() == 0) {
				vector<int> d_binary_shape(1, this->blobs_[0]->count() * 2);
				D_.Reshape(d_binary_shape);
			}
			short* D_hash = (short*)(this->blobs_[0]->cpu_data());
			Dtype* D = D_.mutable_cpu_data();
			for (int i = 0; i < D_.count(); ++i) {
				D[i] = 1.f * D_hash[i] / D_DIVIDER;
			}

			vector<int> weight_shape(4);
			weight_shape[0] = this->num_output_;
			weight_shape[1] = this->channels_;
			weight_shape[2] = kernel_h;
			weight_shape[3] = kernel_w;
			weights.Reshape(weight_shape);
			for (int ct = 0; ct < this->num_output_; ++ct) {
				Dtype* weights_ct = &weights.mutable_cpu_data()[ct * this->channels_ * kernel_dim_];
				const int* B_ct = B + ct * kernel_dim_;
				for (int cs = 0; cs < this->channels_; ++cs) {
					Dtype* weights_ct_cs = weights_ct + cs * kernel_dim_;
					const int* B_ct_cs = B_ct + cs / M * kernel_dim_ * this->num_output_;
					const Dtype* D_slice = D + cs / M * K * M;
					for (int ck= 0; ck < kernel_dim_; ++ck) {
						weights_ct_cs[ck] = D_slice[B_ct_cs[ck] * M + cs % M];
					}
				}
			}

		}
	}

	template <typename Dtype>
	void ConvolutionQUnpackLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* weight = weights.cpu_data();
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->cpu_data();
					this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}

	template <typename Dtype>
	void ConvolutionQUnpackLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionQUnpackLayer);
#endif

INSTANTIATE_CLASS(ConvolutionQUnpackLayer);

}  // namespace caffe
