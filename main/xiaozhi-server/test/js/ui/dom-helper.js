// ota 是否连接成功，修改成对应的样式
export function otaStatusStyle(flan) {
    const otaStatusElement = document.getElementById('otaStatus');
    if (otaStatusElement) {
        if (flan) {
            otaStatusElement.textContent = 'OTA已连接';
            otaStatusElement.style.color = 'green';
        } else {
            otaStatusElement.textContent = 'OTA未连接';
            otaStatusElement.style.color = 'red';
        }
    }
}

// 更新Opus库状态显示
export function updateScriptStatus(message, type) {
    const statusElement = document.getElementById('scriptStatus');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `script-status ${type}`;
        statusElement.style.display = 'block';
        statusElement.style.width = 'auto';
    }
}

