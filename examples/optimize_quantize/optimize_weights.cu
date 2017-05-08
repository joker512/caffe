#undef USE_CUDNN
#include "caffe/util/device_alternate.hpp"

using namespace caffe;

__global__ void kernel_3(int m, int k, const float* A, const float* B, float* C) {
	const int x = blockIdx.x;
	const int y = blockIdx.y;
	const int SLICE_SIZE = (k + blockDim.x - 1) / blockDim.x;
	extern __shared__ float partial_sum[];
	partial_sum[threadIdx.x] = 0.f;

	const int start = SLICE_SIZE * threadIdx.x;
	const int end = min(start + SLICE_SIZE, k);
	const float* A_x = A + x * k;
	const float* B_y = B + y * k;
	for (int i = start; i < end; ++i) {
		partial_sum[threadIdx.x] += A_x[i] * B_y[i];
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		float* C_xy = C + x * m + y;
		for (int i = 0; i < blockDim.x; ++i) {
			*C_xy += partial_sum[i];
		}
	}
}

void multiply(int n, int k, int m, const float* A, const float* B_trans, float* C) {
	dim3 grid(n, m);
	kernel_3<<<grid, CAFFE_CUDA_NUM_THREADS, sizeof(float) * CAFFE_CUDA_NUM_THREADS>>>(
			m, k, A, B_trans, C);
	cudaDeviceSynchronize();
}

__global__ void kernel_2(int num_images, int out_channels, int out_height, int out_width, int k,
								 int m, int kernel_size, int in_height, int in_width, int pad, int stride,
								 int pos, const float* input_slice_, const float* d_slice_,
								 const float* P_, float* bs_diffs) {
	const int input_hw_size = in_height * in_width;
	const int in_slice_image_size = m * input_hw_size;
	const int out_hw_size = out_height * out_width;
	const int out_image_size = out_channels * out_height * out_width;
	const int kernel_h = pos / kernel_size;
	const int kernel_w = pos % kernel_size;

	CUDA_KERNEL_LOOP(j_ct, k * out_channels) {
		int j = j_ct / out_channels;
		int ct = j_ct % out_channels;
		const float* d_slice_j = d_slice_ + j * m;
		float bs_diff = 0.f;

		for (int n = 0; n < num_images; ++n) {
			const float* input_slice_n = input_slice_ + n * in_slice_image_size;
			const float* P_n_ct = P_ + n * out_image_size + ct * out_hw_size;
			int pt_h = -pad + kernel_h;

			for (int h = 0; h < out_height; ++h) {
				if (pt_h >= 0 && pt_h < in_height) {
					const float* P_n_ct_h = P_n_ct + h * out_width;
					const float* input_slice_n_h = input_slice_n + pt_h * in_width;
					int pt_w = -pad + kernel_w;

					for (int w = 0; w < out_width; ++w) {
						if (pt_w >= 0 && pt_w < in_width) {
							float P_n_ct_h_w = P_n_ct_h[w];
							const float* input_slice_n_h_w = input_slice_n_h + pt_w;
							float p = 0.f;

							for (int cs = 0; cs < m; ++cs)
								p += input_slice_n_h_w[cs * input_hw_size] * d_slice_j[cs];
							float diff = p - P_n_ct_h_w;
							bs_diff += diff * diff;
						}
						pt_w += stride;
					}
				}
				pt_h += stride;
			}
		}
		bs_diffs[ct * k + j] = bs_diff;
	}
}

void build_bsdiffs(int num_images, int out_channels, int out_height, int out_width, int k,
						 int m, int kernel_size, int in_height, int in_width, int pad, int stride,
						 int pos, const float* input_slice_, const float* d_slice_,
						 const float* P_, float* bs_diffs) {
	kernel_2<<<CAFFE_GET_BLOCKS(k * out_channels), CAFFE_CUDA_NUM_THREADS>>>(
		num_images, out_channels, out_height, out_width, k, m, kernel_size, in_height, in_width,
		pad, stride, pos, input_slice_, d_slice_, P_, bs_diffs);
	cudaDeviceSynchronize();
}

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
	kernel<<<CAFFE_GET_BLOCKS(num_images * out_channels), CAFFE_CUDA_NUM_THREADS>>>(num_images, out_channels, out_height,
		out_width, k, m, kernel_size, in_height, in_width, pad, stride, input_slice_,
		b_unpacked_slice,  A_);
		cudaDeviceSynchronize();
}
