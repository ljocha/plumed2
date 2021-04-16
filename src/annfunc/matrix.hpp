#include <cstdlib>
#include <vector>
#include <array>
#include <cstdio>
#include <ostream>

class matrix {
    using R_VEC = std::vector<double>;
    using L_VEC = std::vector<double>;
    using DATA = std::vector<double>;

    int height, width;
    DATA _data;

public:
    matrix(int height, int width)
    : height(height)
    , width(width)
    , _data(DATA(height * width))
    {}

    /**
     * row major
     * row length =: width
     * i \in [0, height-1]
     * j \in [0, width-1]
     **/

    /**
     * output += M . input
     */
    void add_right_vector_multiply(const R_VEC& input, L_VEC& output) const {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;
                output[i] += input[j] * _data[index];
            }
        }
    }

    /**
     * OUTPUT = M . INPUT
     */
    template<class I, class O>
    void right_matrix_multiply(const I& INPUT, O& OUTPUT) const {
        fill(OUTPUT._data.begin(), OUTPUT._data.end(), 0);
        int input_width = INPUT.width;
        for (int i = 0; i < height; i++) {
            for (int k = 0; k < width; k++) {
                for (int j = 0; j < input_width; j++) {
                    OUTPUT._data[i * input_width + j] += INPUT._data[k * input_width + j] * _data[i * width + k];
                }
            }
        }
    }

    /**
     * output += input . M
     */
    void add_left_vector_multiply(const L_VEC& input, R_VEC& output) const {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;
                output[j] += input[i] * _data[index];
            }
        }
    }

    /**
     * output = input . M
     */
    void left_vector_multiply(const L_VEC& input, R_VEC& output) const {
        fill(output.begin(), output.end(), 0);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = i * width + j;
                output[j] += input[i] * _data[index];
            }
        }
    }

    /**
     * row(output) = row(M) % input (where % is pointwise product)
     */
    void pointwise_multiply(const R_VEC& input, matrix& output) const {
        for (int i = 0; i < width * height; i++) {
            output._data[i] = input[i % width] * _data[i];
        }
    }

    double& get(int i, int j) {
        return _data[i * width + j];
    }
    
    double get(int i, int j) const {
        return _data[i * width + j];
    }

    void print(std::ostream& out) const {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                out << get(i, j) << "\t";
            }
            out << "\n";
        }
    }

    DATA& data() {
        return _data;
    }
};
