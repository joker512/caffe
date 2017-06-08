#include <vector>
#include <time.h>

#include "caffe/layers/conv_q_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void ConvolutionQLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		BaseConvolutionLayer<Dtype>::Reshape(bottom, top);

		conv_out_spatial_dim_ =  top[0]->count(this->channel_axis_ + 1);
		conv_in_spatial_dim_ = bottom[0]->count(this->channel_axis_ + 1);
		K = this->layer_param_.convolution_param().k();
		int conv_mode = this->layer_param_.convolution_param().conv_mode();
		K_only = conv_mode == ConvolutionParameter_ConvMode_SINGLE_K ? this->layer_param_.convolution_param().for_k() : -1;
		pos_only = conv_mode == ConvolutionParameter_ConvMode_SINGLE_POS ? this->layer_param_.convolution_param().for_pos() : -1;
		M = this->layer_param_.convolution_param().m();
		const int BITS = (int)log2(K);
		const int TOTAL_BITS = 32;
		const int REST_BITS = TOTAL_BITS - BITS;
		const int D_DIVIDER = 10000;
		vector<int> cache_shape_(1, K * this->channels_ / M * conv_in_spatial_dim_);
		cache_.Reshape(cache_shape_);

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
			if (B_ == 0) {
				B_ = new unsigned char[b_shape_size];
			}
			unsigned int* B_hash = (unsigned int*)(this->blobs_[2]->cpu_data());
			vector<int> b_shape(1, b_shape_size);
			Blob<int> B_cache(b_shape);
			int* B = B_cache.mutable_cpu_data();
			for (int i = 0, total_bit_shift = 0; i < b_shape_size; ++i, total_bit_shift += BITS) {
				int byte_shift = total_bit_shift / TOTAL_BITS;
				int bit_shift = total_bit_shift % TOTAL_BITS;
				int shift = REST_BITS - bit_shift;
				B[i] = (shift < 0 ? B_hash[byte_shift] << -shift | B_hash[byte_shift + 1] >> (TOTAL_BITS + shift) :
										  B_hash[byte_shift] >> shift) & (K - 1);
			}

			for (int slice = 0; slice < this->channels_ / M; ++slice) {
				const int* B_cache_slice = B + slice * kernel_dim_ * this->num_output_;
				unsigned char* B_slice = B_ + slice * this->num_output_;
				for (int out_channel = 0; out_channel < this->num_output_; ++out_channel) {
					const int* B_cache_slice_out = B_cache_slice + out_channel * kernel_dim_;
					unsigned char* B_slice_out = B_slice + out_channel;
					for (int khw = 0; khw < kernel_dim_; ++khw) {
						B_slice_out[khw * this->channels_ / M * this->num_output_] = B_cache_slice_out[khw];
					}
				}
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
		}
	}

	template <typename Dtype>
	void ConvolutionQLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* D = D_.cpu_data();
		// hash with indexes of D columns, each number uses log2(K) bits
		const unsigned char* B = B_;
		Blob<Dtype> temp_cache;
		temp_cache.Reshape(cache_.shape());

		const int slice_num = this->channels_ / M;
		const int height = this->conv_input_shape_.cpu_data()[1];
		const int width = this->conv_input_shape_.cpu_data()[2];
		const int kernel_h = this->kernel_shape_.cpu_data()[0];
		const int kernel_w = this->kernel_shape_.cpu_data()[1];
		const int pad_h = this->pad_.cpu_data()[0];
		const int pad_w = this->pad_.cpu_data()[1];
		const int stride_h = this->stride_.cpu_data()[0];
		const int stride_w = this->stride_.cpu_data()[1];
		const int dilation_h = this->dilation_.cpu_data()[0];
		const int dilation_w = this->dilation_.cpu_data()[1];
		const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int output_w = (width + 2 * pad_w -(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

		// TODO: support for multiple bottoms and tops
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();
			Blob<Dtype> out_cache;
			out_cache.Reshape(1, this->num_output_, output_h, output_w);

			for (int n = 0; n < this->num_; ++n) {
				Dtype* top_data_image = &top_data[n * this->top_dim_];
				caffe_set(this->top_dim_, (Dtype)0., top_data_image);
				const Dtype* bottom_data_image = &bottom_data[n * this->bottom_dim_];

				for (int slice = 0; slice < slice_num; ++slice) {
					caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, K, conv_in_spatial_dim_, M, (Dtype)1.,
							D + slice * M * K, bottom_data_image + slice * M * conv_in_spatial_dim_, (Dtype)0.,
							temp_cache.mutable_cpu_data() + slice * K * conv_in_spatial_dim_);
				}

				for (int hw = 0; hw < conv_in_spatial_dim_; ++hw) {
					const Dtype* temp_cache_slice = temp_cache.cpu_data() + hw;
					Dtype* cache_slice = cache_.mutable_cpu_data() + hw * K * slice_num;
					for (int slice = 0; slice < slice_num; ++slice) {
						const Dtype* temp_cache_slice_k = temp_cache_slice + slice * K * conv_in_spatial_dim_;
						Dtype* cache_slice_k = cache_slice + slice * K;
						for (int j = 0; j < K; ++j) {
							*cache_slice_k = temp_cache_slice_k[j * conv_in_spatial_dim_];
							++cache_slice_k;
						}
					}
				}

				for (int output_row = 0; output_row < output_h; ++output_row) {
					Dtype* top_data_row = out_cache.mutable_cpu_data() + output_row * output_w * this->num_output_;

					for (int output_col = 0; output_col < output_w; ++output_col) {
						Dtype* top_data_row_col = top_data_row + output_col * this->num_output_;
						for (int out_channel = 0; out_channel < this->num_output_; ++out_channel)
							top_data_row_col[out_channel] = (Dtype)0.f;

						for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
							const unsigned char* B_row = B + kernel_row * kernel_w * this->num_output_ * slice_num;
							const int input_row = -pad_h + kernel_row * dilation_h + output_row * stride_h;

							if (static_cast<unsigned>(input_row) < static_cast<unsigned>(height)) {
								const Dtype* cache_row = cache_.cpu_data() + input_row * width * K * slice_num;
								for (int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
									const unsigned char* B_row_col = B_row + kernel_col * this->num_output_ * slice_num;
									const int input_col = -pad_h + kernel_col * dilation_h + output_col * stride_w;

									if (static_cast<unsigned>(input_col) < static_cast<unsigned>(width)) {
										const Dtype* cache_row_col = cache_row + input_col * K * slice_num;
										for (int slice = 0; slice < slice_num; ++slice) {
											const unsigned char* B_row_col_slice = B_row_col + slice * this->num_output_;
											const Dtype* cache_row_col_slice = cache_row_col + slice * K;
											top_data_row_col = top_data_row + output_col * this->num_output_;

											// attempt to implement the whole cycle on asm, it works the same time
											/*int num_output = this->num_output_ / 16;
											asm (
															 "CYCLE%=:"
															 "movaps (%1), %%xmm1; movaps (%2), %%xmm1; addps %%xmm0, %%xmm1; movaps %%xmm1, (%1);"
															 "movaps 16(%1), %%xmm1; movaps 16(%2), %%xmm1; addps %%xmm0, %%xmm1; movaps %%xmm1, 16(%1);"
															 "movaps 32(%1), %%xmm1; movaps 32(%2), %%xmm1; addps %%xmm0, %%xmm1; movaps %%xmm1, 32(%1);"
															 "movaps 48(%1), %%xmm1; movaps 48(%2), %%xmm1; addps %%xmm0, %%xmm1; movaps %%xmm1, 48(%1);"
															 "add $64, %1;"
															 "add $64, %2;"
															 "dec %0;"
															 "jne CYCLE%=;"
															 : : "r" (num_output), "r" (top_data_row_col), "r" (cache_row_col_slice) : "xmm0", "xmm1", "memory" );*/

											for (int out_channel = 0; out_channel < this->num_output_ / 16; ++out_channel) {
												Dtype ds[] = {cache_row_col_slice[B_row_col_slice[0]], cache_row_col_slice[B_row_col_slice[1]], cache_row_col_slice[B_row_col_slice[2]],
																		cache_row_col_slice[B_row_col_slice[3]], cache_row_col_slice[B_row_col_slice[4]], cache_row_col_slice[B_row_col_slice[5]],
																		cache_row_col_slice[B_row_col_slice[6]], cache_row_col_slice[B_row_col_slice[7]],
																		cache_row_col_slice[B_row_col_slice[8]], cache_row_col_slice[B_row_col_slice[9]], cache_row_col_slice[B_row_col_slice[10]],
																		cache_row_col_slice[B_row_col_slice[11]], cache_row_col_slice[B_row_col_slice[12]], cache_row_col_slice[B_row_col_slice[13]],
																		cache_row_col_slice[B_row_col_slice[14]], cache_row_col_slice[B_row_col_slice[15]]};
												asm (
													"movaps (%0), %%xmm0; movaps (%1), %%xmm1; addps %%xmm0, %%xmm1; movaps %%xmm1, (%0);"
													"movaps 16(%0), %%xmm0; movaps 16(%1), %%xmm1; addps %%xmm0, %%xmm1; movaps %%xmm1, 16(%0);"
													"movaps 32(%0), %%xmm0; movaps 32(%1), %%xmm1; addps %%xmm0, %%xmm1; movaps %%xmm1, 32(%0);"
													"movaps 48(%0), %%xmm0; movaps 48(%1), %%xmm1; addps %%xmm0, %%xmm1; movaps %%xmm1, 48(%0);"
													: : "r" (top_data_row_col), "r" (ds) : "xmm0", "xmm1", "memory" );

												top_data_row_col += 16;
												B_row_col_slice += 16;
											}
										}
									}
								}
							}
						}
					}
				}

				for (int output_row = 0; output_row < output_h; ++output_row) {
					const Dtype* out_cache_row = out_cache.cpu_data() + output_row * output_w * this->num_output_;
					Dtype* top_data_row = top_data_image + output_row * output_w;

					for (int output_col = 0; output_col < output_w; ++output_col) {
						const Dtype* out_cache_row_col = out_cache_row + output_col * this->num_output_;
						Dtype* top_data_row_col = top_data_row + output_col;

						for (int out_channel = 0; out_channel < this->num_output_; ++out_channel) {
							top_data_row_col[out_channel * output_h * output_w] = out_cache_row_col[out_channel];
						}
					}
				}

				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->cpu_data();
					this->forward_cpu_bias(top_data_image, bias);
				}
			}
		}
	}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionQLayer);
#endif

INSTANTIATE_CLASS(ConvolutionQLayer);

}  // namespace caffe
