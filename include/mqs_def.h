#ifndef MQS_DEF_H
#define MQS_DEF_H

#define MAX_ATTEMPTS 3

typedef struct {
    float phaseAngle;
    float impedance;
} MqsRawDataPoint_t;

typedef struct {
    float startFrequency;
    float frequencyIncrement;
    uint16_t dataCount;
} MqsRawDataSweep_t;

typedef struct {
    uint32_t idMeasurement;
    MqsRawDataSweep_t base;
    MqsRawDataSweep_t afterExposure;
    MqsRawDataPoint_t data[1];    // Size is base.dataCount + afterExposure.dataCount
        // Max size for data is 2 x MQS_SWEEP_MAX_NUMBER_OF_SAMPLES
} MqsRawDataSet_t;

#define MQS_RAW_DATA_SET_SIZE_MAX    (sizeof(MqsRawDataSet_t)                  \
                                      + (2 * 501   \
                                      * sizeof(MqsRawDataPoint_t)))

// Other declarations...

#endif // MQS_DEF_H