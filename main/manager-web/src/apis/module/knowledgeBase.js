import { getServiceUrl } from '../api';
import RequestService from '../httpRequest';

/**
 * 获取认证token
 */
function getAuthToken() {
  return localStorage.getItem('token') || '';
}

/**
 * 知识库管理相关API
 */
export default {
  /**
   * 获取知识库列表
   * @param {Object} params - 查询参数
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  getKnowledgeBaseList(params, callback, errorCallback) {
    const token = getAuthToken();
    const queryParams = new URLSearchParams({
      page: params.page,
      page_size: params.page_size,
      name: params.name || ''
    }).toString();

    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets?${queryParams}`)
      .method('GET')
      .header({ 'Authorization': `Bearer ${token}` })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('获取知识库列表失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.getKnowledgeBaseList(params, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 创建知识库
   * @param {Object} data - 知识库数据
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  createKnowledgeBase(data, callback, errorCallback) {
    console.log('createKnowledgeBase called with data:', data);
    const token = getAuthToken();
    console.log('Token exists:', !!token);
    console.log('API URL:', `${getServiceUrl()}/api/v1/datasets`);
    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets`)
      .method('POST')
      .data(data)
      .header({ 'Authorization': `Bearer ${token}` })
      .success((res) => {
        console.log('createKnowledgeBase success response:', res);
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('创建知识库失败:', err);
        if (err.response) {
          console.error('Error response data:', err.response.data);
          console.error('Error response status:', err.response.status);
        }
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        console.log('Network failure, retrying...');
        const self = this;
        RequestService.reAjaxFun(() => {
          self.createKnowledgeBase(data, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 更新知识库
   * @param {string} datasetId - 知识库ID
   * @param {Object} data - 更新数据
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  updateKnowledgeBase(datasetId, data, callback, errorCallback) {
    console.log('updateKnowledgeBase called with datasetId:', datasetId, 'data:', data);
    const token = getAuthToken();
    console.log('Token exists:', !!token);
    console.log('API URL:', `${getServiceUrl()}/api/v1/datasets/${datasetId}`);
    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets/${datasetId}`)
      .method('PUT')
      .data(data)
      .header({ 'Authorization': `Bearer ${token}` })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('更新知识库失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.updateKnowledgeBase(datasetId, data, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 删除单个知识库
   * @param {string} datasetId - 知识库ID
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  deleteKnowledgeBase(datasetId, callback, errorCallback) {
    console.log('deleteKnowledgeBase called with datasetId:', datasetId);
    const token = getAuthToken();
    console.log('Token exists:', !!token);
    console.log('API URL:', `${getServiceUrl()}/api/v1/datasets/${datasetId}`);
    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets/${datasetId}`)
      .method('DELETE')
      .header({ 'Authorization': `Bearer ${token}` })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('删除知识库失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.deleteKnowledgeBase(datasetId, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 批量删除知识库
   * @param {string|Array} ids - 知识库ID字符串或数组
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  deleteKnowledgeBases(ids, callback, errorCallback) {
    const token = getAuthToken();
    // 确保ids是正确格式的字符串
    const idsStr = Array.isArray(ids) ? ids.join(',') : ids;
    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets/batch?ids=${idsStr}`)
      .method('DELETE')
      .header({ 'Authorization': `Bearer ${token}` })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('批量删除知识库失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.deleteKnowledgeBases(ids, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 获取文档列表
   * @param {string} datasetId - 知识库ID
   * @param {Object} params - 查询参数
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  getDocumentList(datasetId, params, callback, errorCallback) {
    const token = getAuthToken();
    const queryParams = new URLSearchParams({
      page: params.page,
      page_size: params.page_size,
      name: params.name || ''
    }).toString();

    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets/${datasetId}/documents?${queryParams}`)
      .method('GET')
      .header({ 'Authorization': `Bearer ${token}` })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('获取文档列表失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.getDocumentList(datasetId, params, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 上传文档
   * @param {string} datasetId - 知识库ID
   * @param {Object} formData - 表单数据
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  uploadDocument(datasetId, formData, callback, errorCallback) {
    const token = getAuthToken();
    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets/${datasetId}/documents`)
      .method('POST')
      .data(formData)
      .header({ 
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'multipart/form-data'
      })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('上传文档失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.uploadDocument(datasetId, formData, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 解析文档
   * @param {string} datasetId - 知识库ID
   * @param {string} documentId - 文档ID
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  parseDocument(datasetId, documentId, callback, errorCallback) {
    const token = getAuthToken();
    const requestBody = {
      document_ids: [documentId]
    };
    
    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets/${datasetId}/chunks`)
      .method('POST')
      .data(requestBody)
      .header({ 
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('解析文档失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.parseDocument(datasetId, documentId, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 删除文档
   * @param {string} datasetId - 知识库ID
   * @param {string} documentId - 文档ID
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  deleteDocument(datasetId, documentId, callback, errorCallback) {
    const token = getAuthToken();
    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets/${datasetId}/documents/${documentId}`)
      .method('DELETE')
      .header({ 'Authorization': `Bearer ${token}` })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('删除文档失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.deleteDocument(datasetId, documentId, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 获取文档切片列表
   * @param {string} datasetId - 知识库ID
   * @param {string} documentId - 文档ID
   * @param {Object} params - 查询参数
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  listChunks(datasetId, documentId, params, callback, errorCallback) {
    const token = getAuthToken();
    const queryParams = new URLSearchParams({
      page: params.page || 1,
      page_size: params.page_size || 10
    }).toString();

    // 添加关键词搜索参数
    if (params.keywords) {
      queryParams += `&keywords=${encodeURIComponent(params.keywords)}`;
    }

    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets/${datasetId}/documents/${documentId}/chunks?${queryParams}`)
      .method('GET')
      .header({ 'Authorization': `Bearer ${token}` })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('获取切片列表失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.listChunks(datasetId, documentId, params, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 添加切片
   * @param {string} datasetId - 知识库ID
   * @param {string} documentId - 文档ID
   * @param {Object} data - 切片数据
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  addChunk(datasetId, documentId, data, callback, errorCallback) {
    const token = getAuthToken();
    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets/${datasetId}/documents/${documentId}/chunks`)
      .method('POST')
      .data(data)
      .header({ 'Authorization': `Bearer ${token}` })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('添加切片失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.addChunk(datasetId, documentId, data, callback, errorCallback);
        });
      }).send();
  },

  /**
   * 召回测试
   * @param {string} datasetId - 知识库ID
   * @param {Object} data - 召回测试参数
   * @param {Function} callback - 回调函数
   * @param {Function} errorCallback - 错误回调
   */
  retrievalTest(datasetId, data, callback, errorCallback) {
    const token = getAuthToken();
    RequestService.sendRequest()
      .url(`${getServiceUrl()}/api/v1/datasets/${datasetId}/retrieval-test`)
      .method('POST')
      .data(data)
      .header({ 'Authorization': `Bearer ${token}` })
      .success((res) => {
        RequestService.clearRequestTime();
        callback(res);
      })
      .fail((err) => {
        console.error('召回测试失败:', err);
        if (errorCallback) {
          errorCallback(err);
        }
      })
      .networkFail(() => {
        const self = this;
        RequestService.reAjaxFun(() => {
          self.retrievalTest(datasetId, data, callback, errorCallback);
        });
      }).send();
  }

};