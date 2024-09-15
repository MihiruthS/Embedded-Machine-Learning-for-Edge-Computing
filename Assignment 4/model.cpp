#include <math.h>
#include "model.h"
#include "model_data.h"

// Function for ReLU activation

inline float relu(float x) {
    return fmaxf(0.0f, x);
}

// Define layer sizes
#define DENSE1_SIZE 16
#define DENSE2_SIZE 16

float predict(float x) {
    // Activation arrays
    float h1[DENSE1_SIZE];
    float h2[DENSE2_SIZE];
    
    // Compute the activation of the first hidden layer
    for (int i = 0; i < DENSE1_SIZE; ++i) {
        h1[i] = 0.0f;
        for (int j = 0; j < 1; ++j) {
            // W1_data is stored in PROGMEM and accessed with pgm_read_float_near()
            h1[i] += pgm_read_float_near(W1_data + i) * x;
        }
        h1[i] += pgm_read_float_near(b1_data + i);
        h1[i] = relu(h1[i]);
    }

    // Compute the activation of the second hidden layer
    float h2_sum;
    for (int i = 0; i < DENSE2_SIZE; ++i) {
        h2_sum = 0.0f;
        for (int j = 0; j < DENSE1_SIZE; ++j) {
            h2_sum += pgm_read_float_near(W2_data + i * DENSE1_SIZE + j) * h1[j];
        }
        h2_sum += pgm_read_float_near(b2_data + i);
        h2[i] = relu(h2_sum);
    }

    // Compute the output layer
    float y = 0.0f;
    for (int i = 0; i < DENSE2_SIZE; ++i) {
        y += pgm_read_float_near(W3_data + i) * h2[i];
    }
    y += pgm_read_float_near(b3_data);

    return y;
}
