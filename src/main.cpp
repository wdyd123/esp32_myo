#include <Arduino.h>
#include "MyoController.h"

// 全局实例
MyoController* gController = nullptr;

void setup() {
    gController = new MyoController();
    gController->setup();
}

void loop() {
    gController->loop();
}
