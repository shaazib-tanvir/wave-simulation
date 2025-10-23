#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <types.hpp>
#include <glad/gl.h>
#include <GLFW/glfw3.h>

constexpr f32 SCALE = 1.0f;
constexpr f32 DX = 1.0f;
constexpr f32 WAVE_SPEED = 20.0f;
constexpr f32 ANGULAR_FREQUENCY = 2.5f;
constexpr f32 STANDARD_DEVIATION = 1e-2f;

struct WindowData {
	f32 cursor_x;
	f32 cursor_y;
	int framebuffer_width;
	int framebuffer_height;
};

struct Complex {
	f32 x;
	f32 y;
	__device__ Complex(f32 x, f32 y) : x(x), y(y) {}
	__device__ f32 magnitude_squared() {
		return x*x + y*y;
	}
	__device__ Complex operator+(Complex other) {
		return Complex(x + other.x, y + other.y);
	}
	__device__ Complex operator*(Complex other) {
		return Complex(x * other.x - y * other.y, x * other.y + other.x * y);
	}
};

__device__
f32 julia(f32 x, f32 y, f32 time) {
	constexpr size_t iterations = 200;
	const Complex c = Complex(0.088, -.62)*Complex(__cosf(time), __sinf(time));
	Complex z = Complex(x, y);
	for (size_t i = 0; i < iterations; i++) {
		if (z.magnitude_squared() > 1e3) {
			return (f32) i / iterations;
		}
		z = z*z + c;
	}

	return 0.0f;
}

__device__ int idx_2d_to_1d(int i, int j, int width) {
	return width * j + i;
}

__device__ f32 wave_step(f32 const* u_last, f32 const* u_second_last, f32 delta, int i, int j, int width, int height, f32 force) {
	if (i == 0 || i == width - 1 || j == 0 || j == height - 1) {
		return 0.0f;
	}

	return 2.0f*u_last[3*idx_2d_to_1d(i, j, width)]
		-u_second_last[3*idx_2d_to_1d(i, j, width)]
		+(((WAVE_SPEED * delta) / DX) * ((WAVE_SPEED * delta) / DX)) *
			(u_last[3*idx_2d_to_1d(i+1, j, width)]+u_last[3*idx_2d_to_1d(i-1, j, width)]
			 +u_last[3*idx_2d_to_1d(i, j+1, width)]+u_last[3*idx_2d_to_1d(i, j-1, width)]
			 -4.0f*u_last[3*idx_2d_to_1d(i, j, width)])
		+delta*delta*force;
}

__device__ f32 distance_squared(f32 x0, f32 y0, f32 x1, f32 y1) {
	return (x0-x1)*(x0-x1)+(y0-y1)*(y0-y1);
}

__device__ f32 __fabs(f32 x) {
	return x > 0.0f ? x : -x;
}

__global__
void update(f32 const* data, f32* prev_data, f32 time, f32 delta, f32 cursor_x, f32 cursor_y, int width, int height) {
	const f32 FACTOR = 1.0f / sqrt(6.283185307179586f * STANDARD_DEVIATION);
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < width*height; i += gridDim.x * blockDim.x) {
		int screen_x = i % width; 
		int screen_y = i / width;
		f32 x = (f32) screen_x / width;
		f32 y = (f32) screen_y / width;
		x = (x - .5) * SCALE;
		y = (y - .5*((f32)height / width)) * SCALE;
		f32 force = FACTOR * (__expf(-.5*distance_squared(x, y, cursor_x, cursor_y) / (STANDARD_DEVIATION*STANDARD_DEVIATION)))
			* -__sinf(ANGULAR_FREQUENCY*time);
		f32 value = wave_step(data, prev_data, delta, screen_x, screen_y, width, height, force);
		prev_data[3*i] = value;
		prev_data[3*i+1] = 0.0f;
		prev_data[3*i+2] = 0.0f;
	}
}

struct RGBColor {
	f32 r;
	f32 g;
	f32 b;
};

__device__
RGBColor HSVtoRGB(f32 h, f32 s, f32 v) {
    f32 r, g, b, f, p, q, t;
	int i;
    i = floor(h * 6);
    f = h * 6 - i;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

	auto result = RGBColor{};
	result.r = r;
	result.g = g;
	result.b = b;
	return result;
}

__global__
void update_framebuffer(f32 const* data, f32* buffer, int width, int height) {
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < width*height; i += gridDim.x * blockDim.x) {
		auto color = HSVtoRGB(.67*((1.+data[3*i]) / 2.), fabs(data[3*i])*.5+.5, 1.0f);
		buffer[3*i] = color.r;
		buffer[3*i+1] = color.g;
		buffer[3*i+2] = color.b;
		// if (data[3*i] >= 0.0f) {
		// 	buffer[3*i] = fabs(data[3*i]);
		// 	buffer[3*i+1] = 0.0f;
		// 	buffer[3*i+2] = 0.0f;
		// } else {
		// 	buffer[3*i] = 0.0f;
		// 	buffer[3*i+1] = 0.0f;
		// 	buffer[3*i+2] = fabs(data[3*i]);
		// }
	}
}

struct CPUBuffer {
	f32* data;
	int w;
	int h;
	int size;
	CPUBuffer(int w, int h) : data(new f32[3 * w * h]), w(w), h(h), size(3 * w * h * sizeof(*data)) {
		for (int i = 0; i < 3*w*h; i++) {
			data[i] = 0.0f;
		}
	}
	~CPUBuffer() {
		delete[] data;
	}
};

void panic(char* message, ...);
void panic_err(cudaError_t error);
void print_device_info();

void panic(const char* message, ...) {
	std::va_list args;
	va_start(args, message);
	std::vprintf(message, args);
	va_end(args);
	exit(1);
}

void panic_err(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("error: %s\n", cudaGetErrorString(error));
		exit(1);
	}
}

void print_device_info() {
	int device_count;
	panic_err(cudaGetDeviceCount(&device_count));
	for (int i = 0; i < device_count; i++) {
		cudaDeviceProp prop;
		panic_err(cudaGetDeviceProperties(&prop, i));
		printf("=============================================\n");
		printf("Device Name: %s\n", prop.name);
		printf("Total Constant Memory: %zuB\n", prop.totalConstMem);
		printf("Total Global Memory: %zuB\n", prop.totalGlobalMem);
		printf("Bus Width: %db\n", prop.memoryBusWidth);
		printf("L2 Cache: %dB\n", prop.l2CacheSize);
	}
	printf("=============================================\n");
}

static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
	WindowData* window_data = (WindowData*) glfwGetWindowUserPointer(window);
	window_data->cursor_x = (((f32) xpos / window_data->framebuffer_width) - .5) * SCALE;
	window_data->cursor_y = (((f32) (window_data->framebuffer_height-ypos) / window_data->framebuffer_width) - .5*((f32) window_data->framebuffer_height / window_data->framebuffer_width)) * SCALE;
}

int main() {
	constexpr int WIDTH = 1280;
	constexpr int HEIGHT = 720;

	print_device_info();
	if (!glfwInit()) {
		panic("error: failed to initialize glfw");
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Cuda Sandbox", nullptr, nullptr);
	assert(window != nullptr);

	WindowData window_data;
	glfwGetFramebufferSize(window, &window_data.framebuffer_width, &window_data.framebuffer_height);
	glfwSetWindowUserPointer(window, &window_data);
	glfwSetCursorPosCallback(window, cursor_pos_callback);

	glfwMakeContextCurrent(window);
	assert(gladLoadGL(glfwGetProcAddress) != 0);

	CPUBuffer buffer(window_data.framebuffer_width, window_data.framebuffer_height);

	GLuint framebuffer;
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_data.framebuffer_width, window_data.framebuffer_height, 0, GL_RGB, GL_FLOAT, buffer.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	f32* data_gpu_0;
	f32* data_gpu_1;
	f32* framebuffer_gpu;
	cudaMalloc(&data_gpu_0, buffer.size);
	cudaMalloc(&data_gpu_1, buffer.size);
	cudaMalloc(&framebuffer_gpu, buffer.size);
	cudaMemcpy(data_gpu_0, buffer.data, buffer.size, cudaMemcpyHostToDevice);
	cudaMemcpy(data_gpu_1, buffer.data, buffer.size, cudaMemcpyHostToDevice);
	cudaMemcpy(framebuffer_gpu, buffer.data, buffer.size, cudaMemcpyHostToDevice);

	f32 delta = 0.0f;
	uint8_t idx = 0;
	while (!glfwWindowShouldClose(window)) {
		auto start = glfwGetTime();
		glfwPollEvents();

		// buffer.update();
		if (idx == 0) {
			update<<<32, 256>>>(data_gpu_0, data_gpu_1, glfwGetTime(), delta, window_data.cursor_x, window_data.cursor_y, window_data.framebuffer_width, window_data.framebuffer_height);
			update_framebuffer<<<32, 256>>>(data_gpu_1, framebuffer_gpu, window_data.framebuffer_width, window_data.framebuffer_height);
		} else {
			update<<<32, 256>>>(data_gpu_1, data_gpu_0, glfwGetTime(), delta, window_data.cursor_x, window_data.cursor_y, window_data.framebuffer_width, window_data.framebuffer_height);
			update_framebuffer<<<32, 256>>>(data_gpu_0, framebuffer_gpu, window_data.framebuffer_width, window_data.framebuffer_height);
		}

		panic_err(cudaMemcpy(buffer.data, framebuffer_gpu, buffer.size, cudaMemcpyDeviceToHost));

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glBindTexture(GL_TEXTURE_2D, texture);
		glActiveTexture(texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_data.framebuffer_width, window_data.framebuffer_height, GL_RGB, GL_FLOAT, buffer.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glBlitFramebuffer(0, 0, window_data.framebuffer_width, window_data.framebuffer_height, 0, 0, window_data.framebuffer_width, window_data.framebuffer_height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		glfwSwapBuffers(window);
		delta = glfwGetTime() - start;
		idx = (idx + 1) % 2;
	}

	cudaFree(data_gpu_0);
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
