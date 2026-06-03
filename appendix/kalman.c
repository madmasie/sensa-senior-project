#include <stdio.h>
#include <stdlib.h> // For rand()

static float P_k1_k1 = 1.0;   // Error covariance (initialized to non-zero)
static float Kg = 0;          // Kalman Gain
static float P_k_k1 = 1;
static float x_k1_k1 = 0;     // Previous Estimate (State)

/*****************************************************
*Function name: kalman_filter
*Entry parameter: ADC_Value
*****************************************************/
unsigned long kalman_filter(unsigned long ADC_Value)
{
    static float Q = 0.0001;
    static float R = 0.005;

    /* 1. Predict */
    P_k_k1 = P_k1_k1 + Q;
    // Note: We assume x_k_k1 = x_k1_k1 (prediction remains previous state)

    /* 2. Update gain */
    Kg = P_k_k1 / (P_k_k1 + R);

    /* Update the estimate based on the measurement */
    /* Current Est = Previous Est + Gain * (Measurement - Previous Est) */
    x_k1_k1 = x_k1_k1 + Kg * (ADC_Value - x_k1_k1);

    /* Update covariance */
    P_k1_k1 = (1 - Kg) * P_k_k1;

    return (unsigned long)x_k1_k1;
}

int main() {
    printf("Iteration | Raw ADC | Filtered Output\n");
    printf("-------------------------------------\n");

    // Simulate 20 readings
    for (int i = 0; i < 20; i++) {
        // 1. Create a noisy signal (True value 1000, noise +/- 50)
        // using a simple pseudo-random generation for testing
        unsigned long noisy_input = 1000 + (rand() % 100 - 50); 
        
        // 2. Run the filter
        unsigned long filtered = kalman_filter(noisy_input);

        // 3. Print the comparison
        printf("%9d | %7lu | %15lu\n", i, noisy_input, filtered);
    }

    return 0;
}
