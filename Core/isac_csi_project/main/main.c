#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_timer.h"   // 🔥 for timestamp

#define WIFI_SSID "OnePlus Nord CE 3 Lite 5G"
#define WIFI_PASS "5G82610992"

static const char *TAG = "CSI";

/* ===================== CSI CALLBACK ===================== */

void csi_callback(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) return;

    int len = info->len;
    int rssi = info->rx_ctrl.rssi;

    // 🔥 Get time in microseconds
    int64_t ts_us = esp_timer_get_time();

    // 🔥 Convert to seconds
    int seconds = ts_us / 1000000;

    int hrs = seconds / 3600;
    int mins = (seconds % 3600) / 60;
    int secs = seconds % 60;

    printf("TS:%02d:%02d:%02d , RSSI:%d , LEN:%d , DATA:[",
           hrs, mins, secs, rssi, len);

    for (int i = 0; i < len; i++)
    {
        printf("%d", info->buf[i]);
        if (i < len - 1) printf(",");
    }

    printf("]\n");
}

/* ===================== WIFI EVENT HANDLER ===================== */
static void event_handler(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
    {
        esp_wifi_connect();
    }
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
    {
        ESP_LOGI(TAG, "Reconnecting...");
        esp_wifi_connect();
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
    {
        ESP_LOGI(TAG, "WiFi Connected - CSI Started");
    }
}

/* ===================== WIFI INIT ===================== */
void wifi_init()
{
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL, NULL);

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);

    esp_wifi_set_ps(WIFI_PS_NONE);  // stability

    esp_wifi_start();

    /* ===================== ENABLE CSI ===================== */
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = false,
    };

    esp_wifi_set_csi_config(&csi_config);
    esp_wifi_set_csi_rx_cb(&csi_callback, NULL);
    esp_wifi_set_csi(true);
}

/* ===================== MAIN ===================== */
void app_main(void)
{
    nvs_flash_init();
    wifi_init();
}