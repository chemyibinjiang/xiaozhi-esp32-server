// DOM元素
const connectButton = document.getElementById('connectButton');
const serverUrlInput = document.getElementById('serverUrl');
const connectionStatus = document.getElementById('connectionStatus');
const messageInput = document.getElementById('messageInput');
const sendTextButton = document.getElementById('sendTextButton');
const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const conversationDiv = document.getElementById('conversation');
const logContainer = document.getElementById('logContainer');
let visualizerCanvas = document.getElementById('audioVisualizer');

// ota 是否连接成功，修改成对应的样式
export function otaStatusStyle (flan) {
    if(flan){
        document.getElementById('otaStatus').textContent = 'ota已连接';
        document.getElementById('otaStatus').style.color = 'green';
    }else{
        document.getElementById('otaStatus').textContent = 'ota未连接';
        document.getElementById('otaStatus').style.color = 'red';
    }
}

// ota 是否连接成功，修改成对应的样式
export function getLogContainer (flan) {
    return  logContainer;
}

