#include "MyoController.h"

// EMG characteristic short UUIDs: 0x0105, 0x0205, 0x0305, 0x0405
static const char* EMG_CHAR_UUIDS[EMG_CHANNEL_COUNT] = {
    "d5060105-a904-deb9-4748-2c7f4a124842",
    "d5060205-a904-deb9-4748-2c7f4a124842",
    "d5060305-a904-deb9-4748-2c7f4a124842",
    "d5060405-a904-deb9-4748-2c7f4a124842",
};

MyoController::MyoController() :
    myoAddressStr("d5:5a:8e:39:d6:95"),
    myoAddress(nullptr),
    gClient(nullptr),
    pCmdChar(nullptr),
    pImuChar(nullptr),
    pClassifierChar(nullptr),
    pBatteryChar(nullptr),
    imuMutex(nullptr),
    emgMutex(nullptr),
    hasNewImuData(false),
    hasNewEmgData(false),
    bleDisconnected(false),
    lastImuDataMs(0),
    lastEmgDataMs(0),
    lastLoopAliveMs(0),
    lastReconnectAttemptMs(0),
    currentTag(""),
    isCollecting(false),
    armSynced(false),
    batteryLevel(-1),
    headerPrinted(false),
    gClientCallbacks(nullptr) {
    for (int i = 0; i < EMG_CHANNEL_COUNT; i++) {
        pEmgChars[i] = nullptr;
    }
}

void MyoController::setup() {
    Serial.begin(115200);
    delay(1000);

    // Create FreeRTOS mutexes for thread-safe data access
    imuMutex = xSemaphoreCreateMutex();
    emgMutex = xSemaphoreCreateMutex();

    myoAddress = new BLEAddress(std::string(myoAddressStr.c_str()));

    logLine("=== Myo Data Collector ===");
    logLine("Commands: 1-6 = collect gesture, A = set BLE address, B = battery");
    logLine("Waiting for connection...");

    BLEDevice::init("");
    connectMyo();
}

void MyoController::loop() {
    maintainMyoConnection();

    // Serial command handling
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        handleSerialCommand(input);
    }

    // Process IMU data (thread-safe)
    if (hasNewImuData && isCollecting) {
        if (xSemaphoreTake(imuMutex, portMAX_DELAY) == pdTRUE) {
            hasNewImuData = false;
            printCsvHeader();
            processImuData();
            xSemaphoreGive(imuMutex);
        }
    }

    // Process EMG data (thread-safe)
    if (hasNewEmgData && isCollecting) {
        if (xSemaphoreTake(emgMutex, portMAX_DELAY) == pdTRUE) {
            hasNewEmgData = false;
            printCsvHeader();
            processEmgData();
            xSemaphoreGive(emgMutex);
        }
    }

    // Periodic battery check — only log when NOT collecting to avoid polluting CSV
    uint32_t now = millis();
    if (gClient && gClient->isConnected() && pBatteryChar &&
        (now - lastLoopAliveMs > BATTERY_POLL_MS)) {
        lastLoopAliveMs = now;
        if (!isCollecting) {
            readBatteryLevel();
        }
    }
}

void MyoController::handleSerialCommand(const String& input) {
    if (input.length() == 0) return;

    // Gesture collection: 1-6
    if (input.length() == 1 && input >= "1" && input <= "6") {
        if (currentTag == input) {
            // Stop collecting
            isCollecting = false;
            currentTag = "";
            vibrate(0x01);  // short vibration to confirm stop
            logLine("Stopped collecting");
        } else {
            if (!armSynced) {
                logLine("WARNING: Myo arm not synced! Data may be unreliable.");
            }
            currentTag = input;
            isCollecting = true;
            headerPrinted = false;
            vibrate(0x02);  // medium vibration to confirm start
            logLine("Collecting gesture: " + currentTag);
        }
        return;
    }

    // Set BLE address: "A xx:xx:xx:xx:xx:xx"
    if (input.length() > 2 && (input[0] == 'A' || input[0] == 'a')) {
        String addr = input.substring(2);
        addr.trim();
        if (addr.length() == 17) {
            myoAddressStr = addr;
            delete myoAddress;
            myoAddress = new BLEAddress(std::string(myoAddressStr.c_str()));
            disconnectMyo();
            logLine("BLE address set to: " + myoAddressStr + ", reconnecting...");
            connectMyo();
        } else {
            logLine("Invalid address format. Use: A xx:xx:xx:xx:xx:xx");
        }
        return;
    }

    // Battery check: "B"
    if (input == "B" || input == "b") {
        readBatteryLevel();
        return;
    }

    logLine("Unknown command: " + input);
}

void MyoController::logLine(const String& message) {
    if (Serial) {
        Serial.println(message);
    }
}

// --- BLE Client Callbacks ---

MyoController::MyClientCallbacks::MyClientCallbacks(MyoController* controller) : controller(controller) {}

void MyoController::MyClientCallbacks::onConnect(BLEClient* pClient) {
    controller->bleDisconnected = false;
    controller->lastImuDataMs = millis();
    controller->lastEmgDataMs = millis();
    controller->logLine("BLE connected");
}

void MyoController::MyClientCallbacks::onDisconnect(BLEClient* pClient) {
    controller->bleDisconnected = true;
    controller->pCmdChar = nullptr;
    controller->pImuChar = nullptr;
    controller->pClassifierChar = nullptr;
    controller->pBatteryChar = nullptr;
    for (int i = 0; i < EMG_CHANNEL_COUNT; i++) {
        controller->pEmgChars[i] = nullptr;
    }
    controller->armSynced = false;
    controller->logLine("BLE disconnected");
}

// --- BLE Notification Handlers ---

void MyoController::imuNotify(BLERemoteCharacteristic*, uint8_t* data, size_t length, bool) {
    if (length == sizeof(imuData)) {
        if (xSemaphoreTake(imuMutex, 0) == pdTRUE) {
            memcpy(imuData, data, sizeof(imuData));
            hasNewImuData = true;
            lastImuDataMs = millis();
            xSemaphoreGive(imuMutex);
        }
    }
}

void MyoController::emgNotify(int channel, uint8_t* data, size_t length) {
    if (channel < 0 || channel >= EMG_CHANNEL_COUNT) return;
    if (length == 16) {
        if (xSemaphoreTake(emgMutex, 0) == pdTRUE) {
            memcpy(emgData + channel * 16, data, 16);
            hasNewEmgData = true;
            lastEmgDataMs = millis();
            xSemaphoreGive(emgMutex);
        }
    }
}

void MyoController::classifierNotify(BLERemoteCharacteristic*, uint8_t* data, size_t length, bool) {
    if (length < 1) return;
    uint8_t eventType = data[0];
    switch (eventType) {
        case 0x01:  // arm_synced
            armSynced = true;
            logLine("Arm synced");
            break;
        case 0x02:  // arm_unsynced
            armSynced = false;
            logLine("Arm unsynced");
            break;
        case 0x03:  // pose
            if (length >= 3) {
                uint16_t pose = data[1] | (data[2] << 8);
                logLine("Pose: " + String(pose));
            }
            break;
        case 0x06:  // sync_failed
            armSynced = false;
            logLine("Arm sync failed");
            break;
        default:
            break;
    }
}

// --- Connection Management ---

bool MyoController::enableDataStreaming() {
    auto* controlService = gClient->getService(BLEUUID("d5060001-a904-deb9-4748-2c7f4a124842"));
    auto* imuService = gClient->getService(BLEUUID("d5060002-a904-deb9-4748-2c7f4a124842"));
    auto* emgService = gClient->getService(BLEUUID("d5060005-a904-deb9-4748-2c7f4a124842"));
    auto* classifierService = gClient->getService(BLEUUID("d5060003-a904-deb9-4748-2c7f4a124842"));

    if (!controlService) { logLine("Control service not found!"); return false; }
    if (!imuService) { logLine("IMU service not found!"); return false; }
    if (!emgService) { logLine("EMG service not found!"); return false; }

    // Command characteristic
    pCmdChar = controlService->getCharacteristic(BLEUUID("d5060401-a904-deb9-4748-2c7f4a124842"));
    if (!pCmdChar) { logLine("Command characteristic not found!"); return false; }

    // IMU characteristic
    pImuChar = imuService->getCharacteristic(BLEUUID("d5060402-a904-deb9-4748-2c7f4a124842"));
    if (!pImuChar) { logLine("IMU characteristic not found!"); return false; }

    // Subscribe to IMU notifications
    pImuChar->registerForNotify([this](BLERemoteCharacteristic* chr, uint8_t* data, size_t length, bool isNotify) {
        this->imuNotify(chr, data, length, isNotify);
    });

    // Subscribe to ALL 4 EMG characteristics
    for (int i = 0; i < EMG_CHANNEL_COUNT; i++) {
        pEmgChars[i] = emgService->getCharacteristic(BLEUUID(EMG_CHAR_UUIDS[i]));
        if (!pEmgChars[i]) {
            logLine("EMG characteristic " + String(i) + " not found!");
            return false;
        }
        int ch = i;  // capture by value for lambda
        pEmgChars[i]->registerForNotify([this, ch](BLERemoteCharacteristic* chr, uint8_t* data, size_t length, bool isNotify) {
            this->emgNotify(ch, data, length);
        });
    }

    // Subscribe to Classifier events (arm sync detection)
    if (classifierService) {
        pClassifierChar = classifierService->getCharacteristic(BLEUUID("d5060103-a904-deb9-4748-2c7f4a124842"));
        if (pClassifierChar) {
            pClassifierChar->registerForNotify([this](BLERemoteCharacteristic* chr, uint8_t* data, size_t length, bool isNotify) {
                this->classifierNotify(chr, data, length, isNotify);
            });
            logLine("Classifier events enabled");
        }
    }

    // Battery characteristic
    auto* batteryService = gClient->getService(BLEUUID((uint16_t)0x180F));
    if (batteryService) {
        pBatteryChar = batteryService->getCharacteristic(BLEUUID((uint16_t)0x2A19));
    }

    // Enable data streaming: EMG=raw(0x03), IMU=data+events(0x03), Classifier=disabled(0x00)
    uint8_t setMode[] = {0x01, 0x03, 0x03, 0x03, 0x00};
    pCmdChar->writeValue(setMode, sizeof(setMode), true);

    lastImuDataMs = millis();
    lastEmgDataMs = millis();
    logLine("IMU + EMG (4ch) streaming enabled");

    // Read initial battery level
    readBatteryLevel();

    return true;
}

void MyoController::disconnectMyo() {
    pCmdChar = nullptr;
    pImuChar = nullptr;
    pClassifierChar = nullptr;
    pBatteryChar = nullptr;
    for (int i = 0; i < EMG_CHANNEL_COUNT; i++) {
        pEmgChars[i] = nullptr;
    }
    hasNewImuData = false;
    hasNewEmgData = false;
    armSynced = false;
    if (gClient && gClient->isConnected()) {
        gClient->disconnect();
    }
}

bool MyoController::connectMyo() {
    if (gClient == nullptr) {
        gClient = BLEDevice::createClient();
        gClientCallbacks = new MyClientCallbacks(this);
        gClient->setClientCallbacks(gClientCallbacks);
    }

    logLine("Connecting to Myo (" + myoAddressStr + ")...");
    if (!gClient->connect(*myoAddress)) {
        logLine("Connect failed!");
        return false;
    }

    return enableDataStreaming();
}

void MyoController::maintainMyoConnection() {
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

// --- Data Processing ---

void MyoController::processImuData() {
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
        Serial.print(currentTag); Serial.print(",");
        Serial.print(millis()); Serial.print(",IMU,");
        Serial.print(qw); Serial.print(",");
        Serial.print(qx); Serial.print(",");
        Serial.print(qy); Serial.print(",");
        Serial.print(qz); Serial.print(",");
        Serial.print(ax); Serial.print(",");
        Serial.print(ay); Serial.print(",");
        Serial.print(az); Serial.print(",");
        Serial.print(gx); Serial.print(",");
        Serial.print(gy); Serial.print(",");
        Serial.println(gz);
    }
}

void MyoController::processEmgData() {
    // Output all 4 EMG channels (32 values total)
    if (!Serial) return;

    Serial.print(currentTag); Serial.print(",");
    Serial.print(millis()); Serial.print(",EMG,");

    for (int ch = 0; ch < EMG_CHANNEL_COUNT; ch++) {
        MyoEMG* emg = (MyoEMG*)(emgData + ch * 16);
        for (int i = 0; i < 8; i++) {
            Serial.print(emg->sample1[i]); Serial.print(",");
        }
        for (int i = 0; i < 8; i++) {
            Serial.print(emg->sample2[i]);
            // Comma after every value except the very last
            if (!(ch == EMG_CHANNEL_COUNT - 1 && i == 7)) {
                Serial.print(",");
            }
        }
    }
    Serial.println();
}

// --- Utilities ---

void MyoController::vibrate(uint8_t type) {
    if (!pCmdChar) return;
    uint8_t cmd[] = {0x03, 0x01, type};  // command=vibrate, payload_size=1, type
    pCmdChar->writeValue(cmd, sizeof(cmd), true);
}

void MyoController::readBatteryLevel() {
    if (!pBatteryChar) {
        logLine("Battery: N/A");
        return;
    }
    std::string val = pBatteryChar->readValue();
    if (val.length() >= 1) {
        batteryLevel = (int8_t)val[0];
        logLine("Battery: " + String(batteryLevel) + "%");
        if (batteryLevel < 20) {
            logLine("WARNING: Low battery!");
        }
    }
}

void MyoController::printCsvHeader() {
    if (headerPrinted) return;
    headerPrinted = true;

    // Print metadata
    Serial.println("# META: device=myo, imu_rate=50, emg_rate=200, emg_values=32, tag=" + currentTag);

    // Print CSV header
    Serial.print("label,timestamp,type,qw,qx,qy,qz,ax,ay,az,gx,gy,gz");
    for (int i = 0; i < EMG_CHANNEL_COUNT * 8; i++) {
        Serial.print(",emg"); Serial.print(i);
    }
    Serial.println();
}