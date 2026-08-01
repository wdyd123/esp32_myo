#ifndef MYO_CONTROLLER_H
#define MYO_CONTROLLER_H

#include <Arduino.h>
#include <BLEClient.h>
#include <BLEDevice.h>
#include <BLERemoteCharacteristic.h>
#include <BLERemoteService.h>
#include <BLEUtils.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

typedef struct __attribute__((packed)) {
    int16_t w, x, y, z;
    int16_t acc[3];
    int16_t gyro[3];
} MyoIMU;

typedef struct __attribute__((packed)) {
    int8_t sample1[8];       ///< 1st sample of EMG data.
    int8_t sample2[8];       ///< 2nd sample of EMG data.
} MyoEMG;

static constexpr int EMG_CHANNEL_COUNT = 4;  ///< Myo has 4 EMG characteristics

class MyoController {
public:
    MyoController();
    void setup();
    void loop();

private:
    // --- BLE connection ---
    String myoAddressStr;
    BLEAddress* myoAddress;
    BLEClient* gClient;
    BLERemoteCharacteristic* pCmdChar;
    BLERemoteCharacteristic* pImuChar;
    BLERemoteCharacteristic* pEmgChars[EMG_CHANNEL_COUNT];  ///< 4 EMG characteristics
    BLERemoteCharacteristic* pClassifierChar;
    BLERemoteCharacteristic* pBatteryChar;

    // --- Thread-safe data buffers ---
    SemaphoreHandle_t imuMutex;
    SemaphoreHandle_t emgMutex;
    volatile bool hasNewImuData;
    volatile bool hasNewEmgData;
    volatile bool bleDisconnected;
    uint8_t imuData[20];
    uint8_t emgData[EMG_CHANNEL_COUNT * 16];  ///< 4 channels x 16 bytes

    // --- Timing ---
    uint32_t lastImuDataMs;
    uint32_t lastEmgDataMs;
    uint32_t lastLoopAliveMs;
    uint32_t lastReconnectAttemptMs;

    // --- State ---
    String currentTag;
    bool isCollecting;
    bool armSynced;
    int8_t batteryLevel;
    bool headerPrinted;

    static constexpr uint32_t LOOP_ALIVE_INTERVAL_MS = 1000;
    static constexpr uint32_t RECONNECT_INTERVAL_MS = 2000;
    static constexpr uint32_t IMU_TIMEOUT_MS = 3000;
    static constexpr uint32_t BATTERY_POLL_MS = 30000;  ///< Poll battery every 30s

    class MyClientCallbacks : public BLEClientCallbacks {
    public:
        MyClientCallbacks(MyoController* controller);
        void onConnect(BLEClient* pClient) override;
        void onDisconnect(BLEClient* pClient) override;
    private:
        MyoController* controller;
    };

    MyClientCallbacks* gClientCallbacks;

    // --- Methods ---
    void logLine(const String& message);
    void imuNotify(BLERemoteCharacteristic*, uint8_t* data, size_t length, bool);
    void emgNotify(int channel, uint8_t* data, size_t length);
    void classifierNotify(BLERemoteCharacteristic*, uint8_t* data, size_t length, bool);
    bool enableDataStreaming();
    void disconnectMyo();
    bool connectMyo();
    void maintainMyoConnection();
    void processImuData();
    void processEmgData();
    void vibrate(uint8_t type);
    void readBatteryLevel();
    void printCsvHeader();
    void handleSerialCommand(const String& input);
};

#endif