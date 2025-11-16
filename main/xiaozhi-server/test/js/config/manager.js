// 配置管理模块
import { log } from '../utils/logger.js';

// 生成随机MAC地址
function generateRandomMac() {
    const hexDigits = '0123456789ABCDEF';
    let mac = '';
    for (let i = 0; i < 6; i++) {
        if (i > 0) mac += ':';
        for (let j = 0; j < 2; j++) {
            mac += hexDigits.charAt(Math.floor(Math.random() * 16));
        }
    }
    return mac;
}

// 加载配置
export function loadConfig() {
    const deviceMacInput = document.getElementById('deviceMac');
    const deviceNameInput = document.getElementById('deviceName');
    const clientIdInput = document.getElementById('clientId');
    const tokenInput = document.getElementById('token');
    const otaUrlInput = document.getElementById('otaUrl');

    // 从localStorage加载MAC地址，如果没有则生成新的
    let savedMac = localStorage.getItem('deviceMac');
    if (!savedMac) {
        savedMac = generateRandomMac();
        localStorage.setItem('deviceMac', savedMac);
    }
    deviceMacInput.value = savedMac;

    // 从localStorage加载其他配置
    const savedDeviceName = localStorage.getItem('deviceName');
    if (savedDeviceName) {
        deviceNameInput.value = savedDeviceName;
    }

    const savedClientId = localStorage.getItem('clientId');
    if (savedClientId) {
        clientIdInput.value = savedClientId;
    }

    const savedToken = localStorage.getItem('token');
    if (savedToken) {
        tokenInput.value = savedToken;
    }

    const savedOtaUrl = localStorage.getItem('otaUrl');
    if (savedOtaUrl) {
        otaUrlInput.value = savedOtaUrl;
    }
}

// 保存配置
export function saveConfig() {
    const deviceMacInput = document.getElementById('deviceMac');
    const deviceNameInput = document.getElementById('deviceName');
    const clientIdInput = document.getElementById('clientId');
    const tokenInput = document.getElementById('token');

    localStorage.setItem('deviceMac', deviceMacInput.value);
    localStorage.setItem('deviceName', deviceNameInput.value);
    localStorage.setItem('clientId', clientIdInput.value);
    localStorage.setItem('token', tokenInput.value);
}

// 获取配置值
export function getConfig() {
    const deviceMac = document.getElementById('deviceMac').value.trim();
    return {
        deviceId: deviceMac,  // 使用MAC地址作为deviceId
        deviceName: document.getElementById('deviceName').value.trim(),
        deviceMac: deviceMac,
        clientId: document.getElementById('clientId').value.trim(),
        token: document.getElementById('token').value.trim()
    };
}

// 保存连接URL
export function saveConnectionUrls() {
    const otaUrl = document.getElementById('otaUrl').value.trim();
    const wsUrl = document.getElementById('serverUrl').value.trim();
    localStorage.setItem('otaUrl', otaUrl);
    localStorage.setItem('wsUrl', wsUrl);
}
