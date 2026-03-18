#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "esp_log.h"

#define WIFI_SSID "OnePlus Nord CE 3 Lite 5G"
#define WIFI_PASS "5G82610992"

static const char *TAG = "CSI_PROJECT";

// 🔥 CSI CALLBACK
static void wifi_csi_cb(void *ctx, wifi_csi_info_t *info)
{
    printf("CSI_DATA,");

    printf("RSSI:%d,", info->rx_ctrl.rssi);
    printf("CHANNEL:%d,", info->rx_ctrl.channel);
    printf("LEN:%d,", info->len);

    printf("DATA:[");

    for (int i = 0; i < info->len; i++) {
        printf("%d", info->buf[i]);
        if (i != info->len - 1) printf(",");
    }

    printf("]\n");
}

// WIFI INIT
void wifi_init_sta()
{
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
    esp_wifi_connect();
}

// MAIN
void app_main(void)
{
    nvs_flash_init();

    wifi_init_sta();

    // 🔥 IMPORTANT FOR CSI
    esp_wifi_set_promiscuous(true);

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = 0,
    };

    esp_wifi_set_csi_config(&csi_config);
    esp_wifi_set_csi_rx_cb(wifi_csi_cb, NULL);
    esp_wifi_set_csi(true);

    ESP_LOGI(TAG, "CSI STARTED");

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}