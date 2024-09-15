#include "model.h"

// Define LED pin
const int LED_PIN = 11;

// Variables to keep count
int inference_count = 0;
const int kInferencesPerCycle = 1000;

// Define 2*Pi
const float kXrange = 2.f * 3.14159265359f;

void setup() {
    Serial.begin(9600);
    // Set LED pin as output
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // Get x value
    float position = static_cast<float>(inference_count) / static_cast<float>(kInferencesPerCycle);
    float x_val = position * kXrange;

    // Call the model's predict function with the x value
    float y_val = predict(x_val);

    // The model may actually predict values smaller than -1 or larger than 1, so clamp the results.
    y_val = constrain(y_val, -1.0f, 1.0f);

    // Calculate the brightness of the LED such that y = -1 is fully off and y = 1 is fully on. The LED's brightness can range from 0-255.
    int brightness = (int)(127.5f * (y_val + 1.0f));
    
    // Analog write the brightness value to the LED pin
    analogWrite(LED_PIN, brightness);

    // Serial print the brightness value
    Serial.print("Brightness: ");
    Serial.println(brightness);

    // Update x value (Reset reaching 1000 inferences)
    inference_count += 1;
    if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
