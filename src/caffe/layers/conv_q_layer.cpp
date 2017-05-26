#include <vector>
#include <time.h>

#include "caffe/layers/conv_q_layer.hpp"
#include "vectorial/simd4f.h"

namespace caffe {

	template <typename Dtype>
	void ConvolutionQLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		BaseConvolutionLayer<Dtype>::Reshape(bottom, top);

		conv_out_spatial_dim_ =  top[0]->count(this->channel_axis_ + 1);
		conv_in_spatial_dim_ = bottom[0]->count(this->channel_axis_ + 1);
		K = this->layer_param_.convolution_param().k();
		M = this->layer_param_.convolution_param().m();
		const int BITS = (int)log2(K);
		const int TOTAL_BITS = 32;
		const int REST_BITS = TOTAL_BITS - BITS;
		const int D_DIVIDER = 10000;
		vector<int> cache_shape_(1, K * conv_in_spatial_dim_);
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
		}
	}

	// TODO: SPEC for kernel_w = 5,7 pad_w = 0,1
	template <int kernel, int pad, typename Dtype> class kernel_row_processor {
	public:
		kernel_row_processor(int width, int stride_w, int output_w, int conv_in_spatial_dim_)
		: width(width), stride_w(stride_w), output_w(output_w), conv_in_spatial_dim_(conv_in_spatial_dim_)
		{}

		inline void process_row(Dtype* d, const Dtype * cpu_row, const int* b_kernel_row) {
			for (int kernel_col = 0; kernel_col < kernel; ++kernel_col) {
				const Dtype* cache_row_row = cpu_row + b_kernel_row[kernel_col]*conv_in_spatial_dim_;

				int input_col = -pad + kernel_col;

				for (int output_col = 0; output_col < output_w; ++output_col) {
					if (static_cast<unsigned>(input_col) < static_cast<unsigned>(width)) {
						d[output_col] += cache_row_row[input_col];
					}
					input_col += stride_w;
				}
			}
		}

	private:
		const int width;
		const int stride_w;
		const int output_w;
		const int conv_in_spatial_dim_;
	};

	template <typename Dtype>
	class kernel_row_processor<1, 0, Dtype> {
	public:
		kernel_row_processor(int, int stride_w, int output_w, int conv_in_spatial_dim_)
		: stride_w(stride_w), output_w(output_w), conv_in_spatial_dim_(conv_in_spatial_dim_)
		{}

		inline void process_row(Dtype* d, const Dtype * cpu_row, const int* b_kernel_row) {
			const Dtype * src0 = cpu_row + b_kernel_row[0] * conv_in_spatial_dim_;

			Dtype *D = d+output_w;
			while (d < D) {
				*(d++) += src0[0];
				src0+=stride_w;
			}
		}

	private:
		const int stride_w;
		const int output_w;
		const int conv_in_spatial_dim_;
	};

	template <typename Dtype>
	class kernel_row_processor<3, 0, Dtype> {
	public:
		kernel_row_processor(int width, int stride_w, int output_w, int conv_in_spatial_dim_)
		: width(width), stride_w(stride_w), output_w(output_w), conv_in_spatial_dim_(conv_in_spatial_dim_)
		{}

		inline void process_row(Dtype* d, const Dtype * cpu_row, const int* b_kernel_row) {
			const Dtype * src0 = cpu_row + b_kernel_row[0] * conv_in_spatial_dim_;
			const Dtype * src1 = cpu_row + b_kernel_row[1] * conv_in_spatial_dim_;
			const Dtype * src2 = cpu_row + b_kernel_row[2] * conv_in_spatial_dim_;

			Dtype *D = d+output_w;
			while (d < D) {
				*(d++) += src0[0] + src1[1] + src2[2];
				src0+=stride_w;
				src1+=stride_w;
				src2+=stride_w;
			}
		}

	private:
		const int width;
		const int stride_w;
		const int output_w;
		const int conv_in_spatial_dim_;
	};

	template <typename Dtype>
	class kernel_row_processor<3, 1, Dtype> {
	public:
		kernel_row_processor(int width, int stride_w, int output_w, int conv_in_spatial_dim_)
		: width(width), stride_w(stride_w), output_w(output_w), conv_in_spatial_dim_(conv_in_spatial_dim_)
		{}

		inline void process_row(Dtype* d, const Dtype * cpu_row, const int* b_kernel_row) {

			const Dtype * src0 = cpu_row + b_kernel_row[0] * conv_in_spatial_dim_;
			const Dtype * src1 = cpu_row + b_kernel_row[1] * conv_in_spatial_dim_;
			const Dtype * src2 = cpu_row + b_kernel_row[2] * conv_in_spatial_dim_;

			*d += src1[0]+src2[1];
			Dtype *D = d+output_w-1;
			int icol = stride_w;
			while (++d < D) {
				*d += src0[icol-1] + src1[icol] + src2[icol+1];
				icol+=stride_w;
			}
			*d += src0[icol-1]+src1[icol];
		}

	private:
		const int width;
		const int stride_w;
		const int output_w;
		const int conv_in_spatial_dim_;
	};
	template <>
	class kernel_row_processor<3, 1, float> {
	public:
		typedef float Dtype;
		kernel_row_processor(int width, int stride_w, int output_w, int conv_in_spatial_dim_)
		: width(width), stride_w(stride_w), output_w(output_w), conv_in_spatial_dim_(conv_in_spatial_dim_)
		{}

		inline void process_row(Dtype* d, const Dtype * cpu_row, const int* b_kernel_row) {

			const Dtype * src0 = cpu_row + b_kernel_row[0] * conv_in_spatial_dim_;
			const Dtype * src1 = cpu_row + b_kernel_row[1] * conv_in_spatial_dim_;
			const Dtype * src2 = cpu_row + b_kernel_row[2] * conv_in_spatial_dim_;

			*d += src1[0]+src2[1];
			Dtype *D = d+output_w-1;
			++d;
			int icol = 1;
			while (d < D-3) {
				simd4f a = simd4f_uload4(d);
				a = simd4f_add(a, simd4f_uload4(&src0[icol-1]));
				a = simd4f_add(a, simd4f_uload4(&src1[icol  ]));
				a = simd4f_add(a, simd4f_uload4(&src2[icol+1]));
				simd4f_ustore4(a, d);

				d+=4;
				icol+=4;
			}
			while (d < D) {
				*d += src0[icol-1] + src1[icol] + src2[icol+1];
				++icol;
				++d;
			}
			*d += src0[icol-1]+src1[icol];
		}
		inline void process_row(Dtype* d, const Dtype * src0, const Dtype * src1, const Dtype * src2) {

			//const Dtype * src0 = cpu_row + b_kernel_row[0] * conv_in_spatial_dim_;
			//const Dtype * src1 = cpu_row + b_kernel_row[1] * conv_in_spatial_dim_;
			//const Dtype * src2 = cpu_row + b_kernel_row[2] * conv_in_spatial_dim_;

			*d += src1[0]+src2[1];
			Dtype *D = d+output_w-1;
			++d;
			int icol = 1;
			while (d < D-3) {
				simd4f a = simd4f_uload4(d);
				a = simd4f_add(a, simd4f_uload4(&src0[icol-1]));
				a = simd4f_add(a, simd4f_uload4(&src1[icol  ]));
				a = simd4f_add(a, simd4f_uload4(&src2[icol+1]));
				simd4f_ustore4(a, d);

				d+=4;
				icol+=4;
			}
			while (d < D) {
				*d += src0[icol-1] + src1[icol] + src2[icol+1];
				++icol;
				++d;
			}
			*d += src0[icol-1]+src1[icol];
		}

	private:
		const int width;
		const int stride_w;
		const int output_w;
		const int conv_in_spatial_dim_;
	};

	template <int kernel, int pad, typename Dtype> class kernel_cell_processor {
	public:
		kernel_cell_processor(int height,int width,int stride,int output_h,int output_w, int conv_in_spatial_dim_)
		: height(height),width(width),stride(stride),output_h(output_h),output_w(output_w),conv_in_spatial_dim_(conv_in_spatial_dim_)
		{}

		inline void process_image(Dtype* top_data_channel, const Dtype * cpu_data, const int* b_out_channel) {
			kernel_row_processor<kernel, pad, Dtype>  krp(width, stride, output_w, conv_in_spatial_dim_);
			
			for (int output_row = 0; output_row < output_h; ++output_row) {
				Dtype* top_data_row = &top_data_channel[output_row * output_w];

				for (int kernel_row = 0; kernel_row < kernel; ++kernel_row) {
					int input_row = -pad + kernel_row + output_row*stride;
					if (static_cast<unsigned>(input_row) < static_cast<unsigned>(height)) {
						krp.process_row(top_data_row, cpu_data + input_row*width, b_out_channel + kernel_row*kernel);
					}
				}
			}
		}

	private:
		const int height;
		const int width;
		const int stride;
		const int output_h;
		const int output_w;
		const int conv_in_spatial_dim_;
	};

	template <> class kernel_cell_processor<1,0,float> {
	public:
		typedef float Dtype;
		kernel_cell_processor(int height,int width,int stride,int output_h,int output_w, int conv_in_spatial_dim_)
		: height(height),width(width),stride(stride),output_h(output_h),output_w(output_w),conv_in_spatial_dim_(conv_in_spatial_dim_)
		{}

		inline void process_image(Dtype* top_data_channel, const Dtype * cpu_data, const int* b_out_channel) {
			kernel_row_processor<1,0,Dtype>  krp(width, stride, output_w, conv_in_spatial_dim_);

			const Dtype *cpu_data_b = cpu_data + b_out_channel[0] * conv_in_spatial_dim_;

			int N = output_h*output_w;
			if (stride==1 && (N&3)==0 && (uintptr_t(top_data_channel)&0xF)==0 && (uintptr_t(cpu_data_b)&0xF)==0) {
				N/=4;
				simd4f *d = (simd4f*)top_data_channel;
				simd4f *s = (simd4f*)cpu_data_b;
				for (int i = 0; i<N; ++i)
				{
					d[i] = simd4f_add(d[i], s[i]);
				}
			} else {
				const int off = stride*width;
				for (int output_row = 0; output_row < output_h; ++output_row) {

					Dtype* d = top_data_channel + output_row*output_w;
					const Dtype * src0 = cpu_data_b;

					Dtype *D = d+output_w;
					while (d < D) {
						*(d++) += src0[0];
						src0+=stride;
					}
					cpu_data_b+=off;
				}
			}
		}

	private:
		const int height;
		const int width;
		const int stride;
		const int output_h;
		const int output_w;
		const int conv_in_spatial_dim_;
	};

	template <typename Dtype> class kernel_cell_processor<3,1,Dtype> {
	public:
		kernel_cell_processor(int height,int width,int stride,int output_h,int output_w, int conv_in_spatial_dim_)
		: height(height),width(width),stride(stride),output_h(output_h),output_w(output_w),conv_in_spatial_dim_(conv_in_spatial_dim_)
		{}

		inline void process_image(Dtype* top_data_channel, const Dtype * cpu_data, const int* b_out_channel) {
			kernel_row_processor<3, 1, Dtype>  krp(width, stride, output_w, conv_in_spatial_dim_);

			Dtype* top_data_row = top_data_channel;
			krp.process_row(top_data_row, cpu_data, b_out_channel + 3);
			krp.process_row(top_data_row, cpu_data+width, b_out_channel + 6);
			top_data_row += output_w;

			int input_row = stride;
			for (int output_row = 1; output_row < output_h-1; ++output_row) {
				krp.process_row(top_data_row, cpu_data + (input_row-1)*width, b_out_channel + 0);
				krp.process_row(top_data_row, cpu_data + (input_row)*width, b_out_channel + 3);
				krp.process_row(top_data_row, cpu_data + (input_row+1)*width, b_out_channel + 6);

				input_row+=stride;
				top_data_row += output_w;
			}

			krp.process_row(top_data_row, cpu_data + (input_row-1)*width, b_out_channel + 0);
			krp.process_row(top_data_row, cpu_data + (input_row)*width, b_out_channel + 3);
		}

	private:
		const int height;
		const int width;
		const int stride;
		const int output_h;
		const int output_w;
		const int conv_in_spatial_dim_;
	};

	// for stride 1
	template <> class kernel_cell_processor<3,1,float> {
	public:
		typedef float Dtype;

		kernel_cell_processor(int height,int width,int stride,int output_h,int output_w, int conv_in_spatial_dim_)
		: height(height),width(width),stride(stride),output_h(output_h),output_w(output_w),conv_in_spatial_dim_(conv_in_spatial_dim_)
		{}

		inline void process_image(Dtype* top_data_channel, const Dtype * cpu_data, const int* b_out_channel) {
			const Dtype *src00 = cpu_data+b_out_channel[0]*conv_in_spatial_dim_-width;
			const Dtype *src01 = cpu_data+b_out_channel[1]*conv_in_spatial_dim_-width;
			const Dtype *src02 = cpu_data+b_out_channel[2]*conv_in_spatial_dim_-width;

			const Dtype *src10 = cpu_data+b_out_channel[3]*conv_in_spatial_dim_;
			const Dtype *src11 = cpu_data+b_out_channel[4]*conv_in_spatial_dim_;
			const Dtype *src12 = cpu_data+b_out_channel[5]*conv_in_spatial_dim_;

			const Dtype *src20 = cpu_data+b_out_channel[6]*conv_in_spatial_dim_+width;
			const Dtype *src21 = cpu_data+b_out_channel[7]*conv_in_spatial_dim_+width;
			const Dtype *src22 = cpu_data+b_out_channel[8]*conv_in_spatial_dim_+width;

			kernel_row_processor<3, 1, Dtype>  krp(width, stride, output_w, conv_in_spatial_dim_);

			Dtype* top_data_row = top_data_channel;
			krp.process_row(top_data_row, src10, src11, src12);
			krp.process_row(top_data_row, src20, src21, src22);

			top_data_row += output_w;

			int input_row = stride;
			for (int output_row = 1; output_row < output_h-1; ++output_row) {

				Dtype *d = top_data_row;
				Dtype *D = d+output_w-1;

				int icol = input_row*width;
				*d += src01[icol]+src02[icol+1]
				    + src11[icol]+src12[icol+1]
				    + src21[icol]+src22[icol+1];
				++icol;
				++d;

				while (d < D-3) {

					simd4f a = simd4f_uload4(d);
					a = simd4f_add(a, simd4f_uload4(&src00[icol-1]));
					a = simd4f_add(a, simd4f_uload4(&src10[icol-1]));
					a = simd4f_add(a, simd4f_uload4(&src20[icol-1]));

					a = simd4f_add(a, simd4f_uload4(&src01[icol]));
					a = simd4f_add(a, simd4f_uload4(&src11[icol]));
					a = simd4f_add(a, simd4f_uload4(&src21[icol]));

					a = simd4f_add(a, simd4f_uload4(&src02[icol+1]));
					a = simd4f_add(a, simd4f_uload4(&src12[icol+1]));
					a = simd4f_add(a, simd4f_uload4(&src22[icol+1]));

					simd4f_ustore4(a, d);
					icol+=4;
					d+=4;
				}
				//while(D
				while (d < D) {
					*d += src00[icol-1] + src01[icol] + src02[icol+1]
					    + src10[icol-1] + src11[icol] + src12[icol+1]
					    + src20[icol-1] + src21[icol] + src22[icol+1];
					icol+=stride;
					++d;
				}
				*d += src00[icol-1]+src01[icol]
				    + src10[icol-1]+src11[icol]
				    + src20[icol-1]+src21[icol];
				//krp.process_row(top_data_row, cpu_data + (input_row-1)*width, b_out_channel + 0);
				//krp.process_row(top_data_row, cpu_data + (input_row)*width, b_out_channel + 3);
				//krp.process_row(top_data_row, cpu_data + (input_row+1)*width, b_out_channel + 6);
				input_row+=stride;
				top_data_row += output_w;
			}
			int off = input_row*width;
			krp.process_row(top_data_row, src00+off, src01+off, src02+off);
			krp.process_row(top_data_row, src10+off, src11+off, src12+off);
		}

	private:
		const int height;
		const int width;
		const int stride;
		const int output_h;
		const int output_w;
		const int conv_in_spatial_dim_;
	};
	template <typename Dtype>
	void ConvolutionQLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* D = D_.cpu_data();
		// hash with indexes of D columns, each number uses log2(K) bits
		const int* B = B_.cpu_data();

		const int height = this->conv_input_shape_.cpu_data()[1];
		const int width = this->conv_input_shape_.cpu_data()[2];
		const int kernel_h = this->kernel_shape_.cpu_data()[0];
		const int kernel_w = this->kernel_shape_.cpu_data()[1];
		const int kernel_dim_ = kernel_h * kernel_w;
		const int pad_h = this->pad_.cpu_data()[0];
		const int pad_w = this->pad_.cpu_data()[1];
		const int stride_h = this->stride_.cpu_data()[0];
		const int stride_w = this->stride_.cpu_data()[1];
		const int dilation_h = this->dilation_.cpu_data()[0];
		const int dilation_w = this->dilation_.cpu_data()[1];
		const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int output_w = (width + 2 * pad_w -(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

		// check if we can use optimazed kernel processor
		//const bool can_use_fast_path = (kernel_h==kernel_w) && (stride_h==stride_w);

		// TODO: support for multiple bottoms and tops
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();

			for (int n = 0; n < this->num_; ++n) {
				Dtype* top_data_image = &top_data[n * this->top_dim_];
				caffe_set(this->top_dim_, (Dtype)0., top_data_image);
				const Dtype* bottom_data_image = &bottom_data[n * this->bottom_dim_];

				for (int slice = 0; slice < this->channels_ / M; ++slice) {
					caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, K, conv_in_spatial_dim_, M, (Dtype)1.,
							D + slice * M * K, bottom_data_image + slice * M * conv_in_spatial_dim_, (Dtype)0.,
							this->cache_.mutable_cpu_data());
					const int* b_slice = &B[slice * this->num_output_ * kernel_dim_];

					for (int out_channel = 0; out_channel < this->num_output_; ++out_channel) {
						const int* b_out_channel = &b_slice[out_channel * kernel_dim_];
						Dtype* top_data_channel = &top_data_image[out_channel * this->conv_out_spatial_dim_];

						if (kernel_w==3 && pad_w==1) {
							kernel_cell_processor<3,1,Dtype> kcp(height, width, stride_h, output_h, output_w, conv_in_spatial_dim_);
							kcp.process_image(top_data_channel, this->cache_.cpu_data(), b_out_channel);
						} else if (kernel_w==3 && pad_w==0) {
							kernel_cell_processor<3,0,Dtype> kcp(height, width, stride_h, output_h, output_w, conv_in_spatial_dim_);
							kcp.process_image(top_data_channel, this->cache_.cpu_data(), b_out_channel);
						} else if (kernel_w==1 && pad_w==0) {
							kernel_cell_processor<1,0,Dtype> kcp(height, width, stride_h, output_h, output_w, conv_in_spatial_dim_);
							kcp.process_image(top_data_channel, this->cache_.cpu_data(), b_out_channel);
						} else { // universal alg
							for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
								const int* b_kernel_row = &b_out_channel[kernel_row * kernel_w];

								for (int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
									int b = b_kernel_row[kernel_col];
									int input_row = -pad_h + kernel_row * dilation_h;
									const Dtype* cache_row = &this->cache_.cpu_data()[b * conv_in_spatial_dim_];
								
									for (int output_row = 0; output_row < output_h; ++output_row) {
										if (static_cast<unsigned>(input_row) < static_cast<unsigned>(height)) {
											const Dtype* cache_row_row = &cache_row[input_row * width];
											Dtype* top_data_row = &top_data_channel[output_row * output_w];
											int input_col = -pad_w + kernel_col * dilation_w;
											
											for (int output_col = 0; output_col < output_w; ++output_col) {
												if (static_cast<unsigned>(input_col) < static_cast<unsigned>(width)) {
													top_data_row[output_col] += cache_row_row[input_col];
												}
												input_col += stride_w;
											}
										}
										input_row += stride_h;
									}
								}
							}
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
