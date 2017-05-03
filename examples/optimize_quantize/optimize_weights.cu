#undef USE_CUDNN
#include "caffe/util/device_alternate.hpp"

using namespace caffe;

__global__ void kernel(int num_images, int out_channels, int out_height, int out_width, int k, 
							  int m, int kernel_size, int in_height, int in_width, int pad, int stride,
							  const float* input_slice_, const int* b_unpacked_slice, float* A_) {
	const int input_hw_size = in_height * in_width;
	const int out_hw_size = out_height * out_width;
	const int in_slice_image_size = m * input_hw_size;
	const int out_image_size = out_channels * out_height * out_width;

	CUDA_KERNEL_LOOP(n_ct, num_images * out_channels) {
		int n = n_ct / out_channels;
		float* A_n = A_ + n * out_image_size * m;
		const float* input_slice_n = input_slice_ + n * in_slice_image_size;

		int ct = n_ct % out_channels;
		float* A_n_ct = A_n + ct * out_hw_size * m;
		for (int i = 0; i < out_hw_size * m; ++i)
			A_n_ct[i] = 0.f;
		const int* b_ct = b_unpacked_slice + ct * kernel_size * kernel_size;

		for (int kh = 0; kh < kernel_size; ++kh) {
			const int* b_ct_kh = b_ct + kh * kernel_size;

			for (int kw = 0; kw < kernel_size; ++kw) {
				const int b = b_ct_kh[kw];
				if (k == b) {
					int pt_h = -pad + kh;
					for (int h = 0; h < out_height; ++h) {
						if (pt_h >= 0 && pt_h < in_height) {
							float* A_n_ct_h = A_n_ct + h * out_width * m;
							const float* input_slice_n_pkh = input_slice_n + pt_h * in_width;
							int pt_w = -pad + kw;

							for (int w = 0; w < out_width; ++w) {
								if (pt_w >= 0 && pt_w < in_width) {
									float* A_n_ct_h_w = A_n_ct_h + w * m;
									const float* input_slice_n_pkhw = input_slice_n_pkh + pt_w;

									for (int cs = 0; cs < m; ++cs) {
										A_n_ct_h_w[cs] += input_slice_n_pkhw[cs * input_hw_size];
									}
								}
								pt_w += stride;
							}
						}
						pt_h += stride;
					}
				}
			}
		}
	}
}

void build_A(int num_images, int out_channels, int out_height, int out_width,
	int k, int m, int kernel_size, int in_height, int in_width, int pad, int stride,
	const float* input_slice_, const int* b_unpacked_slice, float* A_) {
	kernel<<<CAFFE_GET_BLOCKS(num_images), CAFFE_CUDA_NUM_THREADS>>>(num_images, out_channels, out_height,
		out_width, k, m, kernel_size, in_height, in_width, pad, stride, input_slice_,
		b_unpacked_slice,  A_);
}
