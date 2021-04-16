#include "./matrix.hpp"

class Network
{
using vector = std::vector<double>;

private:
    int input_size;
    int hidden_size;

    double rescale_factor;

    vector input_layer;
    vector hidden_layer;
    double output_layer;

    matrix w_input_hidden;
    vector w_hidden_output;

    vector b_hidden;
    double b_output;

    vector grad_input; // input layer: output(i) | x
    vector grad_hidden; // hidden layer: output(i) | y

    void set_weights(const vector& w_input_hidden, const vector& w_hidden_output) {
        std::copy_n(w_input_hidden.begin(), input_size * hidden_size, this->w_input_hidden.data().begin());
        std::copy_n(w_hidden_output.begin(), hidden_size, this->w_hidden_output.begin());
    }

    void set_biases(const vector& b_hidden, double b_output) {
        std::copy_n(b_hidden.begin(), hidden_size, this->b_hidden.begin());
        this->b_output = b_output;
    }

public:
    explicit Network(
        int input_size, int hidden_size, double rescale_factor,
        vector& w_input_hidden, const vector& w_hidden_output,
        const vector& b_hidden, double b_output
    )
    : input_size(input_size)
    , hidden_size(hidden_size)
    , rescale_factor(rescale_factor)
    , input_layer(vector(input_size))
    , hidden_layer(vector(hidden_size))
    , w_input_hidden(matrix(hidden_size, input_size))
    , w_hidden_output(vector(hidden_size))
    , b_hidden(vector(hidden_size))
    , grad_input(vector(input_size))
    , grad_hidden(vector(hidden_size))
    {
        set_weights(w_input_hidden, w_hidden_output);
        set_biases(b_hidden, b_output);
    }

    double forth_and_back_propagation() {
        /* rescale the input */
        for (double& i : input_layer ) {
            i *= rescale_factor;
        }

        /* compute innier potentials in the hidden layer */
        hidden_layer = b_hidden;
        w_input_hidden.add_right_vector_multiply(input_layer, hidden_layer);

        /* compute
        *  - activations in the hidden layer
        *  - the gradient with respect to output value in the hidden layer
        */
        for (int k = 0; k < hidden_size; k++) {
            hidden_layer[k] = tanh(hidden_layer[k]);
            grad_hidden[k] = (1 - hidden_layer[k] * hidden_layer[k]) * w_hidden_output[k];
        }

        /* compute the output value */
        output_layer = b_output;
        for (int k = 0; k < hidden_size; k++) {
            output_layer += w_hidden_output[k] * hidden_layer[k];
        }

        /* compute the gradient with respect to output value in the input layer */
        w_input_hidden.left_vector_multiply(grad_hidden, grad_input);

        /* rescale the input gradient */
        for (double& g : grad_input) {
            g *= rescale_factor;
        }
        
        return output_layer;
    }

    void print_params(std::ostream& out) {
        out << "weights0:\n";
        w_input_hidden.print(out);
        out << "weights1:\n";
        for (int i = 0; i < hidden_size; i++) {
            out << w_hidden_output[i] << " ";
        }
        out << "biases0:\n";
        for (int i = 0; i < hidden_size; i++) {
            out << b_hidden[i] << " ";
        }
        out << "\nbiases1:\n";
        out << b_output << " ";
        out << "\n";
    }

    void print_output(std::ostream& out) {
        out << "input layer: \n";
        for (int i = 0; i < input_size; i++) {
            out << input_layer[i] << " ";
        }
        out << "\n\nhidden layer: \n";
        for (int i = 0; i < hidden_size; i++) {
            out << hidden_layer[i] << " ";
        }
        out << "\n\nouput layer: ";
        out << output_layer << " ";
        out << "\n";
        out << "\n";
    }

    vector& get_input() {
        return input_layer;
    }

    vector& get_input_gradient() {
        return grad_input;
    }
};
