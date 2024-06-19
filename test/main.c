#include <stdio.h>
#include <stdlib.h>
#include "mes_savgol.h"

void printData(const MqsRawDataPoint_t data[], size_t dataSize) {
    printf("%lu yourSavgolData = [", dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        if (i == dataSize - 1) {
            printf("%f", data[i].phaseAngle);
        } else {
            printf("%f, ", data[i].phaseAngle);
        }
    }
    printf("];\n");
}

int main() {
    // Define a sample dataset
    double dataset[] = { 11.272, 11.254, 11.465, 11.269, 11.31, 11.388, 11.385, 11.431, 11.333, 11.437, 11.431, 11.527, 11.483, 11.449, 11.544, 11.39, 11.469, 11.526, 11.498, 11.522, 11.709, 11.503, 11.564, 11.428, 11.714, 11.707, 11.619, 11.751, 11.626, 11.681, 11.838, 11.658, 11.859, 11.916, 11.814, 11.833, 12.046, 11.966, 12.031, 12.079, 11.958, 12.114, 12.041, 12.186, 12.048, 12.258, 12.312, 12.126, 12.159, 12.393, 12.221, 12.45, 12.439, 12.282, 12.373, 12.573, 12.647, 12.545, 12.467, 12.629, 12.686, 12.668, 12.748, 12.71, 12.852, 13.02, 12.848, 13.144, 13.225, 13.211, 13.496, 13.311, 13.33, 13.634, 13.189, 13.623, 13.671, 13.618, 13.645, 13.779, 14.006, 14.13, 14.071, 14.277, 14.223, 14.457, 14.378, 14.698, 14.599, 14.84, 15.143, 15.106, 15.343, 15.506, 15.665, 15.889, 15.878, 16.055, 16.153, 15.966, 16.637, 16.783, 16.746, 17.193, 16.877, 17.656, 17.522, 17.842, 18.086, 18.336, 18.863, 18.977, 19.534, 19.308, 19.626, 19.956, 20.221, 20.673, 20.59, 21.229, 21.767, 22.225, 22.477, 22.695, 22.828, 23.586, 23.776, 24.39, 25.316, 24.639, 25.767, 26.469, 26.976, 27.651, 27.807, 28.089, 28.869, 29.964, 30.367, 30.159, 31.133, 32.034, 33.131, 32.775, 34.372, 34.516, 35.603, 36.214, 37.742, 38.868, 38.702, 39.811, 40.818, 41.422, 41.521, 42.57, 42.819, 42.871, 42.944, 43.851, 44.086, 44.272, 44.466, 44.274, 44.473, 44.348, 43.932, 43.817, 43.48, 42.943, 42.491, 41.793, 41.071, 39.491, 39.231, 38.365, 37.833, 36.583, 35.787, 34.949, 33.006, 32.827, 32.266, 31.012, 30.436, 29.737, 28.097, 28.76, 27.068, 26.195, 25.262, 24.677, 24.211, 23.574, 22.868, 22.781, 22.258, 21.475, 21.247, 21.982, 20.771, 20.383, 20.349, 19.866, 19.433, 18.573, 18.723, 18.325, 18.084, 18.226, 17.492, 17.505, 16.762, 16.907, 16.606, 16.265, 16.234, 15.983, 16.147, 15.811, 15.667, 15.509, 15.325, 15.031, 14.884, 14.881, 14.836, 14.814, 14.706, 14.158, 14.399, 14.123, 14.084, 14.173, 13.963, 13.981, 14.218, 13.898, 13.869, 13.701, 13.397, 13.528, 13.321, 13.071, 13.393, 13.164, 12.876, 13.021, 12.989, 12.869, 13.004, 12.833, 12.795, 12.661, 12.761, 12.547, 12.775, 12.388, 12.425, 12.564, 12.408, 12.301, 12.469, 12.173, 12.323, 12.248, 12.281, 12.208, 11.887, 12.149, 12.073, 12.053, 11.88, 12.066, 11.958, 12.007, 11.868, 11.921, 11.898, 11.804, 11.7, 11.81, 11.758, 11.717, 11.715, 11.611, 11.719, 11.679, 11.619, 11.58, 11.576, 11.589, 11.491, 11.659, 11.506, 11.431, 11.535, 11.349, 11.464, 11.343, 11.492, 11.407, 11.479, 11.269, 11.355, 11.323, 11.341, 11.238, 11.32, 11.333, 11.262, 11.31, 11.221, 11.302, 11.135, 11.139, 11.217, 11.343, 11.225, 11.089, 11.079, 11.127, 11.082, 11.141, 11.186, 11.184, 11.231, 11.025, 11.058, 11.076, 11.087, 11.047, 11.02, 10.996, 10.906, 11.144, 11.005, 10.911, 10.993, 10.858, 11.086, 10.954, 10.906, 11.026, 11.005, 10.934, 10.922, 10.914, 10.955, 11.057, 10.967, 10.811, 10.833, 10.747, 10.821, 10.946, 10.844, 10.838, 10.848, 10.847};
    size_t dataSize = sizeof(dataset)/sizeof(dataset[0]);
    
    MqsRawDataPoint_t rawData[dataSize];
    for (int i = 0; i < dataSize; ++i) {
        rawData[i].phaseAngle = dataset[i];
        rawData[i].impedance = 0.0;  // You can set the impedance to a default value
    }
    MqsRawDataPoint_t filteredData[360] = {0.0};

    // Set parameters for the Savitzky-Golay filter
    uint8_t halfWindowSize = 7; // Example half window size
    uint8_t polynomialOrder = 3; // Example polynomial order
    uint8_t targetPoint = 0; // Target point for the filter
    uint8_t derivativeOrder = 0; // Derivative order for the filter

    // Apply the Savitzky-Golay filter
    mes_savgolFilter(rawData, dataSize, halfWindowSize, filteredData, polynomialOrder, targetPoint, derivativeOrder);

    // Print the filtered data
   
    printData(filteredData, dataSize);

    return 0;
}
