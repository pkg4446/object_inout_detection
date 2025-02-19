import os
import subprocess
from wifi import Cell, Scheme

ssid = 'YOUR_SSID'  # 연결하고자 하는 WiFi 네트워크의 SSID
password = 'YOUR_PASSWORD'  # WiFi 네트워크의 비밀번호

def wifi_scan():
    # 사용 가능한 WiFi 네트워크 검색
    networks = Cell.all('wlan0')
    # 연결할 네트워크 찾기
    for network in networks:
        if network.ssid == ssid:
            scheme = Scheme.for_cell('wlan0', 'home', network, password)
            scheme.save()
            scheme.activate()
            print(f"Connected to {ssid}")
            break
        else:
            print(f"SSID: {network.ssid}, 신호 강도: {network.signal}, 암호화: {network.encryption_type}")

# wpa_supplicant.conf 파일에 네트워크 정보를 추가하는 함수
def update_wpa_supplicant(ssid, password):
    with open('/etc/wpa_supplicant/wpa_supplicant.conf', 'a') as file:
        file.write(f'\nnetwork={{\n    ssid="{ssid}"\n    psk="{password}"\n    key_mgmt=WPA-PSK\n}}\n')
    print("wpa_supplicant.conf 파일이 업데이트되었습니다.")

# 시스템 명령어를 실행하는 함수
def restart_wpa_supplicant():
    try:
        subprocess.run(['sudo', 'systemctl', 'restart', 'wpa_supplicant'], check=True)
        print("wpa_supplicant 서비스가 재시작되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"명령어 실행에 실패했습니다: {e}")

# wpa_supplicant.conf 파일 업데이트 및 서비스 재시작
wifi_scan()
# update_wpa_supplicant(ssid, password)
# restart_wpa_supplicant()

os._exit(0)