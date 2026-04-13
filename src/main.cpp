#include <Arduino.h>
#include <BLEClient.h>
#include <BLEDevice.h>
#include <BLERemoteCharacteristic.h>
#include <BLERemoteService.h>
#include <BLEUtils.h>

typedef struct __attribute__((packed)) {
    int16_t w, x, y, z;
    int16_t acc[3];
    int16_t gyro[3];
} MyoIMU;

static BLEAddress myoAddress("d5:5a:8e:39:d6:95");
static BLEClient* gClient = nullptr;
static BLERemoteCharacteristic* pCmdChar = nullptr;
static BLERemoteCharacteristic* pImuChar = nullptr;

static volatile bool hasNewData = false;
static volatile bool bleDisconnected = false;
static uint8_t imuData[20];

static uint32_t lastImuDataMs = 0;
static uint32_t lastLoopAliveMs = 0;
static uint32_t lastReconnectAttemptMs = 0;

static constexpr uint32_t LOOP_ALIVE_INTERVAL_MS = 1000;
static constexpr uint32_t RECONNECT_INTERVAL_MS = 2000;
static constexpr uint32_t IMU_TIMEOUT_MS = 3000;

static void logLine(const String& message) {
    if (Serial) {
        Serial.println(message);
    }
}

class MyClientCallbacks : public BLEClientCallbacks {
    void onConnect(BLEClient* pClient) override {
        bleDisconnected = false;
        lastImuDataMs = millis();
        logLine("BLE connected");
    }

    void onDisconnect(BLEClient* pClient) override {
        bleDisconnected = true;
        pCmdChar = nullptr;
        pImuChar = nullptr;
        logLine("BLE disconnected");
    }
};

static MyClientCallbacks gClientCallbacks;

void imuNotify(
    BLERemoteCharacteristic*,
    uint8_t* data,
    size_t length,
    bool) {

    if (length == sizeof(imuData)) {
        memcpy(imuData, data, sizeof(imuData));
        hasNewData = true;
        lastImuDataMs = millis();
    }
}

static bool enableImuStreaming() {
    auto* controlService = gClient->getService(BLEUUID("d5060001-a904-deb9-4748-2c7f4a124842"));
    auto* imuService = gClient->getService(BLEUUID("d5060002-a904-deb9-4748-2c7f4a124842"));

    if (!controlService) {
        logLine("Control service not found!");
        return false;
    }
    if (!imuService) {
        logLine("IMU service not found!");
        return false;
    }

    pCmdChar = controlService->getCharacteristic(BLEUUID("d5060401-a904-deb9-4748-2c7f4a124842"));
    pImuChar = imuService->getCharacteristic(BLEUUID("d5060402-a904-deb9-4748-2c7f4a124842"));

    if (!pCmdChar) {
        logLine("Command characteristic not found!");
        return false;
    }
    if (!pImuChar) {
        logLine("IMU characteristic not found!");
        return false;
    }

    pImuChar->registerForNotify(imuNotify);

    uint8_t setMode[] = {
        0x01, 0x03,
        0x00,
        0x01,
        0x00
    };
    pCmdChar->writeValue(setMode, sizeof(setMode), true);

    lastImuDataMs = millis();
    logLine("IMU enabled");
    return true;
}

static void disconnectMyo() {
    pCmdChar = nullptr;
    pImuChar = nullptr;
    hasNewData = false;

    if (gClient && gClient->isConnected()) {
        gClient->disconnect();
    }
}

static bool connectMyo() {
    if (gClient == nullptr) {
        gClient = BLEDevice::createClient();
        gClient->setClientCallbacks(&gClientCallbacks);
    }

    logLine("Connecting to Myo...");
    if (!gClient->connect(myoAddress)) {
        logLine("Connect failed!");
        return false;
    }

    return enableImuStreaming();
}

static void maintainMyoConnection() {
    uint32_t now = millis();
    bool connected = gClient != nullptr && gClient->isConnected();
    bool imuTimedOut = connected && (now - lastImuDataMs > IMU_TIMEOUT_MS);

    if (!connected || bleDisconnected || imuTimedOut) {
        if (imuTimedOut) {
            logLine("IMU data timeout, reconnecting...");
        }

        if (now - lastReconnectAttemptMs < RECONNECT_INTERVAL_MS) {
            return;
        }

        lastReconnectAttemptMs = now;
        disconnectMyo();
        bleDisconnected = false;
        connectMyo();
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    logLine("Start");

    BLEDevice::init("");
    connectMyo();
}

void loop() {
    maintainMyoConnection();

    if (hasNewData) {
        hasNewData = false;

        MyoIMU* imu = (MyoIMU*)imuData;

        float qw = imu->w / 16384.0f;
        float qx = imu->x / 16384.0f;
        float qy = imu->y / 16384.0f;
        float qz = imu->z / 16384.0f;

        float ax = imu->acc[0] / 2048.0f;
        float ay = imu->acc[1] / 2048.0f;
        float az = imu->acc[2] / 2048.0f;

        float gx = imu->gyro[0] / 16.0f;
        float gy = imu->gyro[1] / 16.0f;
        float gz = imu->gyro[2] / 16.0f;

        if (Serial) {
            Serial.print("Q: ");
            Serial.print(qw); Serial.print(" ");
            Serial.print(qx); Serial.print(" ");
            Serial.print(qy); Serial.print(" ");
            Serial.print(qz);

            Serial.print(" | ACC: ");
            Serial.print(ax); Serial.print(" ");
            Serial.print(ay); Serial.print(" ");
            Serial.print(az);

            Serial.print(" | GYRO: ");
            Serial.print(gx); Serial.print(" ");
            Serial.print(gy); Serial.print(" ");
            Serial.print(gz);

            Serial.println();
        }
    }

    // uint32_t now = millis();
    // if (now - lastLoopAliveMs >= LOOP_ALIVE_INTERVAL_MS) {
    //     lastLoopAliveMs = now;
    //     logLine("loop alive");
    // }

    delay(10);
}
