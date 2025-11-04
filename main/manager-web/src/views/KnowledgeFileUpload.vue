<template>
  <div class="welcome">
    <HeaderBar />

    <div class="operation-bar">
      <div class="left-operations">
          <el-button class="btn-back" type="text" @click="$router.back()">
            &lt;
          </el-button>
          <h2 class="knowledge-base-title">文档管理-{{ knowledgeBaseName }}</h2>
        </div>
      <div class="right-operations">
        <el-input :placeholder="$t('knowledgeFileUpload.searchPlaceholder')" v-model="searchName" class="search-input"
          @keyup.enter.native="handleSearch" clearable />
        <el-button class="btn-search" @click="handleSearch">{{ $t('knowledgeFileUpload.search') }}</el-button>
      </div>
    </div>

    <div class="main-wrapper">
      <div class="content-panel">
        <div class="content-area">
          <el-card class="params-card" shadow="never">
            <div class="table-wrapper">
              <el-table ref="fileTable" :data="fileList" class="transparent-table" v-loading="loading"
                element-loading-text="Loading" element-loading-spinner="el-icon-loading"
                element-loading-background="rgba(255, 255, 255, 0.7)"
                :header-cell-class-name="headerCellClassName"
                @selection-change="handleSelectionChange">
                <el-table-column type="selection" width="55" align="center"></el-table-column>
                <el-table-column :label="$t('knowledgeFileUpload.documentName')" prop="name" align="center">
                  <template slot-scope="scope">
                    <span class="document-name">{{ scope.row.name }}</span>
                  </template>
                </el-table-column>
                <el-table-column :label="$t('knowledgeFileUpload.uploadTime')" prop="createdAt" align="center">
                  <template slot-scope="scope">
                    <span>{{ formatDate(scope.row.createdAt) }}</span>
                  </template>
                </el-table-column>
                <el-table-column :label="$t('knowledgeFileUpload.sliceCount')" align="center">
                  <template slot-scope="scope">
                    <span>{{ scope.row.sliceCount || 0 }}</span>
                  </template>
                </el-table-column>
                <el-table-column :label="$t('knowledgeFileUpload.operation')" align="center">
                  <template slot-scope="scope">
                    <el-button size="mini" type="text" @click="handleParse(scope.row)" 
                      :disabled="scope.row.status === 1"
                      v-if="!scope.row.sliceCount || scope.row.sliceCount <= 0">
                      {{ $t('knowledgeFileUpload.parse') }}
                    </el-button>
                    <el-button size="mini" type="text" @click="handleViewSlices(scope.row)"
                      v-if="scope.row.sliceCount && scope.row.sliceCount > 0">
                      {{ $t('knowledgeFileUpload.viewSlices') }}
                    </el-button>
                    <el-button size="mini" type="text" @click="handleDelete(scope.row)">
                      {{ $t('knowledgeFileUpload.delete') }}
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>
            </div>

            <div class="table_bottom">
              <div class="ctrl_btn">
                <el-button type="primary" @click="handleSelectAll">
                  {{ isAllSelected ? $t('knowledgeFileUpload.deselectAll') : $t('knowledgeFileUpload.selectAll') }}
                </el-button>
                <el-button type="success" @click="showUploadDialog">
                  {{ $t('knowledgeFileUpload.add') }}
                </el-button>
                <el-button type="danger" @click="handleBatchDelete" :disabled="selectedFiles.length === 0">
                  {{ $t('knowledgeFileUpload.batchDelete') }}
                </el-button>
                <el-button 
                  type="primary" 
                  icon="el-icon-search"
                  @click="showRetrievalTestDialog"
                  style="background-color: #409EFF; border-color: #409EFF;">
                  {{ $t('knowledgeFileUpload.retrievalTest') }}
                </el-button>
              </div>
              <div class="custom-pagination">
                <el-select v-model="pageSize" @change="handlePageSizeChange" class="page-size-select">
                  <el-option v-for="item in pageSizeOptions" :key="item"
                    :label="`${item}${$t('knowledgeFileUpload.itemsPerPage')}`" :value="item">
                  </el-option>
                </el-select>
                <button class="pagination-btn" :disabled="currentPage === 1" @click="goFirst">
                  {{ $t('knowledgeFileUpload.firstPage') }}
                </button>
                <button class="pagination-btn" :disabled="currentPage === 1" @click="goPrev">
                  {{ $t('knowledgeFileUpload.prevPage') }}
                </button>
                <button v-for="page in visiblePages" :key="page" class="pagination-btn"
                  :class="{ active: page === currentPage }" @click="goToPage(page)">
                  {{ page }}
                </button>
                <button class="pagination-btn" :disabled="currentPage === pageCount" @click="goNext">
                  {{ $t('knowledgeFileUpload.nextPage') }}
                </button>
                <span class="total-text">{{ $t('knowledgeFileUpload.totalRecords', { total }) }}</span>
              </div>
            </div>
          </el-card>
        </div>
      </div>
    </div>

    <!-- 上传文档对话框 -->
    <el-dialog :title="$t('knowledgeFileUpload.uploadDocument')" :visible.sync="uploadDialogVisible" width="800px">
      <el-upload
        class="document-uploader"
        drag
        :action="uploadUrl"
        :auto-upload="false"
        :on-change="handleFileChange"
        :multiple="true"
        :show-file-list="false"
        accept=".doc,.docx,.pdf,.txt,.md,.mdx,.csv,.xls,.xlsx,.ppt,.pptx">
        <i class="el-icon-upload"></i>
        <div class="el-upload__text">{{ $t('knowledgeFileUpload.dragOrClick') }}</div>
        <div class="el-upload__tip">{{ $t('knowledgeFileUpload.uploadTip') }}</div>
      </el-upload>
      
      <!-- 已选择文件列表 -->
      <div class="selected-files-section" v-if="selectedFilesList.length > 0">
        <h4>{{ $t('knowledgeFileUpload.selectedFiles') }} ({{ selectedFilesList.length }})</h4>
        <div class="selected-files-list">
          <div 
            v-for="(file, index) in selectedFilesList" 
            :key="index" 
            class="selected-file-item"
          >
            <div class="file-info">
              <i class="el-icon-document"></i>
              <span class="file-name">{{ file.name }}</span>
              <span class="file-size">{{ formatFileSize(file.size) }}</span>
            </div>
            <el-button 
              type="text" 
              class="remove-btn" 
              @click="removeSelectedFile(index)"
            >
              <i class="el-icon-close"></i>
            </el-button>
          </div>
        </div>
      </div>
      
      <div slot="footer" class="dialog-footer">
        <el-button @click="uploadDialogVisible = false">{{ $t('knowledgeFileUpload.cancel') }}</el-button>
        <el-button type="primary" @click="handleBatchUploadSubmit" :loading="uploading" :disabled="selectedFilesList.length === 0">
          {{ $t('knowledgeFileUpload.confirm') }} {{ selectedFilesList.length > 0 ? `(${selectedFilesList.length}${$t('knowledgeFileUpload.itemsPerPage').replace('条/页', '个文件')})` : '' }}
        </el-button>
      </div>
    </el-dialog>

    <!-- 切片管理弹窗 -->
    <el-dialog :title="`${$t('knowledgeFileUpload.viewSlices')} - ${currentDocumentName}`" :visible.sync="sliceDialogVisible" width="1200px" class="slice-dialog">
      <div class="slice-management">
        <!-- 切片列表 -->
        <div class="slice-list-section">
          <div class="slice-header">
          <div class="slice-info">
            <h3>{{ $t('knowledgeFileUpload.documentName') }}：{{ currentDocumentName }}</h3>
          </div>
        </div>
          
          <!-- 切片内容卡片式布局 -->
          <div v-loading="sliceLoading" class="slice-content-container">
            <div v-if="sliceList.length > 0" class="slice-cards-container">
              <div v-for="(slice, index) in sliceList" :key="index" class="slice-card">
                <div class="slice-header-info">
                  <p><strong>{{ $t('knowledgeFileUpload.slice') }} {{ (sliceCurrentPage - 1) * slicePageSize + index + 1 }}</strong></p>
                </div>
                <div class="slice-card-content">
                  <div class="content-text">{{ slice.content }}</div>
                </div>
              </div>
            </div>
            <div v-else class="no-slice-data">
              <el-alert type="info" :title="$t('knowledgeFileUpload.noSliceData')"></el-alert>
            </div>
          </div>
          
          <!-- 切片分页 -->
          <div class="slice-pagination" style="margin-top: 20px; text-align: right;">
            <div class="custom-pagination">
              <!-- 条/页选择器 -->
              <el-select 
                v-model="slicePageSize" 
                @change="handleSliceSizeChange"
                class="page-size-select"
                :popper-append-to-body="false"
              >
                <el-option v-for="item in pageSizeOptions" :key="item" 
                  :label="`${item}${$t('knowledgeFileUpload.itemsPerPage')}`" :value="item">
                </el-option>
              </el-select>
              
              <!-- 首页按钮 -->
              <button class="pagination-btn" 
                :disabled="sliceCurrentPage === 1" 
                @click="goToSliceFirstPage">
                {{ $t('knowledgeFileUpload.firstPage') }}
              </button>
              
              <!-- 上一页按钮 -->
              <button class="pagination-btn" 
                :disabled="sliceCurrentPage === 1" 
                @click="goToSlicePrevPage">
                {{ $t('knowledgeFileUpload.prevPage') }}
              </button>
              
              <!-- 页码按钮 -->
              <button v-for="page in sliceVisiblePages" :key="page" class="pagination-btn"
                :class="{ active: page === sliceCurrentPage }" @click="goToSlicePage(page)">
                {{ page }}
              </button>
              
              <!-- 下一页按钮 -->
              <button class="pagination-btn" 
                :disabled="sliceCurrentPage === slicePageCount" 
                @click="goToSliceNextPage">
                {{ $t('knowledgeFileUpload.nextPage') }}
              </button>
              
              <!-- 总记录数 -->
              <span class="total-text">{{ $t('knowledgeFileUpload.totalRecords', { total: sliceTotal }) }}</span>
            </div>
          </div>
        </div>
      </div>
    </el-dialog>

    <!-- 召回测试弹窗 -->
    <el-dialog :title="$t('knowledgeFileUpload.retrievalTest')" :visible.sync="retrievalTestDialogVisible" width="1200px" class="retrieval-test-dialog">
      <div class="retrieval-test-form">
        <el-form :model="retrievalTestForm" label-width="80px">
            <el-form-item :label="$t('knowledgeFileUpload.testQuestion')" required>
              <el-input
                v-model="retrievalTestForm.question"
                type="textarea"
                :rows="2"
                :placeholder="$t('knowledgeFileUpload.testQuestionPlaceholder')"
                style="width: 100%; max-height: 80px;">
              </el-input>
            </el-form-item>
          </el-form>
        
        <div class="retrieval-test-actions" style="text-align: center; margin-top: 20px;">
            <el-button type="primary" @click="runRetrievalTest" :loading="retrievalTestLoading">
              {{ $t('knowledgeFileUpload.runTest') }}
            </el-button>
            <el-button @click="retrievalTestDialogVisible = false">{{ $t('knowledgeFileUpload.cancel') }}</el-button>
          </div>
        
        <div v-if="retrievalTestResult" class="retrieval-test-result">
            <h4>{{ $t('knowledgeFileUpload.testResult') }}</h4>
            <el-card v-if="retrievalTestResult.chunks && retrievalTestResult.chunks.length > 0">
              <div class="result-chunk-container">
                <div v-for="(chunk, index) in retrievalTestResult.chunks" :key="index" class="result-chunk">
                  <p><strong>{{ $t('knowledgeFileUpload.slice') }} {{ index + 1 }}</strong></p>
                  <div class="similarity-scores">
                    <div class="score-item">
                      <span class="score-label">{{ $t('knowledgeFileUpload.comprehensiveSimilarity') }}</span>
                      <span class="score-value">{{ (chunk.similarity || 0).toFixed(4) }}</span>
                    </div>
                  </div>
                  <div class="chunk-content">
                    <p><strong>{{ $t('knowledgeFileUpload.content') }}</strong></p>
                    <p>{{ chunk.content }}</p>
                  </div>
                  <el-divider v-if="index < retrievalTestResult.chunks.length - 1"></el-divider>
                </div>
              </div>
            </el-card>
            <el-alert v-else type="info" :title="$t('knowledgeFileUpload.noRelatedSlices')"></el-alert>
          </div>
      </div>
    </el-dialog>

    <el-footer>
      <version-footer />
    </el-footer>
  </div>
</template>

<script>
import Api from "@/apis/api";
import HeaderBar from "@/components/HeaderBar.vue";
import VersionFooter from "@/components/VersionFooter.vue";
import KnowledgeBaseAPI from "@/apis/module/knowledgeBase";

export default {
  components: { HeaderBar, VersionFooter },
  data() {
    return {
      datasetId: '',
      knowledgeBaseName: '',
      searchName: "",
      fileList: [],
      selectedFiles: [],
      currentPage: 1,
      loading: false,
      pageSize: 10,
      pageSizeOptions: [10, 20, 50, 100],
      total: 0,
      uploadDialogVisible: false,
      uploading: false,
      uploadForm: {
        name: '',
        file: null
      },
      uploadUrl: '',
      isAllSelected: false,
      selectedFilesList: [], // 批量上传选择的文件列表
      
      // 切片管理相关数据
      sliceDialogVisible: false,
      currentDocumentId: '',
      currentDocumentName: '',
      sliceList: [],
      sliceLoading: false,
      sliceSearchKeyword: '',
      sliceCurrentPage: 1,
      slicePageSize: 10,
      sliceTotal: 0,
      
      // 召回测试相关数据
      retrievalTestDialogVisible: false,
      retrievalTestForm: {
        question: ''
      },
      retrievalTestResult: null,
      retrievalTestLoading: false
    };
  },
  created() {
    this.datasetId = this.$route.query.datasetId || '';
    this.knowledgeBaseName = this.$route.query.knowledgeBaseName || '';
    this.uploadUrl = `${Api.getServiceUrl()}/api/v1/documents/upload`;
    this.fetchFileList();
  },
  computed: {
    pageCount() {
      return Math.ceil(this.total / this.pageSize);
    },
    visiblePages() {
      const pages = [];
      const maxVisible = 3;
      let start = Math.max(1, this.currentPage - 1);
      let end = Math.min(this.pageCount, start + maxVisible - 1);

      if (end - start + 1 < maxVisible) {
        start = Math.max(1, end - maxVisible + 1);
      }

      for (let i = start; i <= end; i++) {
        pages.push(i);
      }
      return pages;
    },
    // 切片分页页数计算
    slicePageCount() {
      return Math.ceil(this.sliceTotal / this.slicePageSize);
    },
    // 切片分页可见页码计算（最多显示3个页码）
    sliceVisiblePages() {
      const pages = [];
      const maxVisible = 3;
      let start = Math.max(1, this.sliceCurrentPage - 1);
      let end = Math.min(this.slicePageCount, start + maxVisible - 1);

      if (end - start + 1 < maxVisible) {
        start = Math.max(1, end - maxVisible + 1);
      }

      for (let i = start; i <= end; i++) {
        pages.push(i);
      }
      return pages;
    },
  },
  methods: {
    goBack() {
      this.$router.push('/knowledge-base-management');
    },
    handlePageSizeChange: function(val) {
      this.pageSize = val;
      this.currentPage = 1;
      this.fetchFileList();
    },
    fetchFileList: function() {
      this.loading = true;
      const params = {
        page: this.currentPage,
        page_size: this.pageSize,
        name: this.searchName
      };
      
      KnowledgeBaseAPI.getDocumentList(this.datasetId, params, 
        async ({data}) => {
          this.loading = false;
          if (data && data.code === 0) {
            this.fileList = data.data.list;
            this.total = data.data.total;
            
            // 为每个文档获取切片数量
            await this.fetchSliceCountsForDocuments();
          } else {
            this.$message.error(data?.msg || this.$t('knowledgeFileUpload.getListFailed'));
            this.fileList = [];
            this.total = 0;
          }
        },
        (err) => {
          this.loading = false;
          this.$message.error(this.$t('knowledgeFileUpload.getListFailed'));
          console.error('获取文档列表失败:', err);
          this.fileList = [];
          this.total = 0;
        }
      );
    },
    
    // 为文档列表中的每个文档获取切片数量
    fetchSliceCountsForDocuments: async function() {
      if (!this.fileList || this.fileList.length === 0) {
        return;
      }
      
      // 为每个文档创建获取切片数量的Promise
      const sliceCountPromises = this.fileList.map(async (document) => {
        try {
          const params = {
            page: 1,
            page_size: 1 // 只需要获取总数，不需要实际数据
          };
          
          return new Promise((resolve) => {
            KnowledgeBaseAPI.listChunks(this.datasetId, document.id, params,
              ({data}) => {
                if (data && data.code === 0) {
                  // 获取切片总数
                  const sliceCount = data.data.total || 0;
                  resolve({ documentId: document.id, sliceCount });
                } else {
                  console.warn(`获取文档 ${document.name} 切片数量失败:`, data?.msg);
                  resolve({ documentId: document.id, sliceCount: 0 });
                }
              },
              (err) => {
                console.warn(`获取文档 ${document.name} 切片数量失败:`, err);
                resolve({ documentId: document.id, sliceCount: 0 });
              }
            );
          });
        } catch (error) {
          console.warn(`获取文档 ${document.name} 切片数量失败:`, error);
          return { documentId: document.id, sliceCount: 0 };
        }
      });
      
      try {
        // 等待所有切片数量获取完成
        const sliceCountResults = await Promise.all(sliceCountPromises);
        
        // 更新文档列表中的切片数量
        sliceCountResults.forEach(result => {
          const document = this.fileList.find(doc => doc.id === result.documentId);
          if (document) {
            // 使用Vue.set确保响应式更新
            this.$set(document, 'sliceCount', result.sliceCount);
          }
        });
        
        // 强制更新视图
        this.$forceUpdate();
      } catch (error) {
        console.error('获取切片数量失败:', error);
      }
    },
    
    // 获取单个文档的切片数量
    fetchSliceCountForSingleDocument: function(documentId) {
      const document = this.fileList.find(doc => doc.id === documentId);
      if (!document) {
        console.warn('未找到文档:', documentId);
        return;
      }
      
      const params = {
        page: 1,
        page_size: 1  // 只需要获取总数，所以每页1条记录即可
      };
      
      KnowledgeBaseAPI.listChunks(this.datasetId, documentId, params,
        ({data}) => {
          if (data && data.code === 0) {
            const sliceCount = data.data.total || 0;
            // 更新文档的切片数量
            this.$set(document, 'sliceCount', sliceCount);
            // 强制更新视图
            this.$forceUpdate();
            console.log(`文档 ${document.name} 切片数量已更新为:`, sliceCount);
          } else {
            console.warn(`获取文档 ${document.name} 切片数量失败:`, data?.msg);
          }
        },
        (err) => {
          console.warn(`获取文档 ${document.name} 切片数量失败:`, err);
        }
      );
    },
    
    // 智能检测切片生成状态并自动刷新
    smartRefreshSliceCount: function(documentId, maxRetries = 10, interval = 2000) {
      const document = this.fileList.find(doc => doc.id === documentId);
      if (!document) {
        console.warn('未找到文档:', documentId);
        return;
      }
      
      let retryCount = 0;
      let lastSliceCount = document.sliceCount || 0;
      
      const checkSliceStatus = () => {
        const params = {
          page: 1,
          page_size: 1
        };
        
        KnowledgeBaseAPI.listChunks(this.datasetId, documentId, params,
          ({data}) => {
            if (data && data.code === 0) {
              const currentSliceCount = data.data.total || 0;
              
              if (currentSliceCount > 0 && currentSliceCount !== lastSliceCount) {
                // 切片数量有变化，更新显示
                this.$set(document, 'sliceCount', currentSliceCount);
                this.$forceUpdate();
                console.log(`文档 ${document.name} 切片数量已自动更新为:`, currentSliceCount);
                
                // 如果切片数量稳定且大于0，停止检测
                if (currentSliceCount > 0) {
                  return;
                }
              }
              
              lastSliceCount = currentSliceCount;
              
              // 继续检测直到达到最大重试次数
              if (retryCount < maxRetries) {
                retryCount++;
                setTimeout(checkSliceStatus, interval);
              } else {
                console.log(`文档 ${document.name} 切片检测已达到最大重试次数`);
              }
            } else {
              console.warn(`获取文档 ${document.name} 切片数量失败:`, data?.msg);
              
              // 失败时也继续重试
              if (retryCount < maxRetries) {
                retryCount++;
                setTimeout(checkSliceStatus, interval);
              }
            }
          },
          (err) => {
            console.warn(`获取文档 ${document.name} 切片数量失败:`, err);
            
            // 失败时也继续重试
            if (retryCount < maxRetries) {
              retryCount++;
              setTimeout(checkSliceStatus, interval);
            }
          }
        );
      };
      
      // 开始检测
      setTimeout(checkSliceStatus, 1000); // 1秒后开始第一次检测
    },
    handleSearch: function() {
      this.currentPage = 1;
      this.fetchFileList();
    },
    headerCellClassName: function({ row, column, rowIndex, columnIndex }) {
      if (columnIndex === 0) {
        return 'header-cell-first';
      }
      return 'header-cell';
    },
    showUploadDialog: function() {
      this.uploadForm = {
        name: '',
        file: null
      };
      this.selectedFilesList = []; // 清空已选择文件列表
      this.uploadDialogVisible = true;
    },
    handleFileChange: function(file, fileList) {
      if (!file || !file.raw) return;
      
      // 文件上传前的验证
      const isLt10M = file.size / 1024 / 1024 < 10;
      if (!isLt10M) {
        this.$message.error('文件大小不能超过10MB!');
        return;
      }
      
      // 添加到已选择文件列表
      this.selectedFilesList.push({
        name: file.name,
        size: file.size,
        raw: file.raw
      });
    },
    beforeUpload: function(file) {
      // 文件上传前的验证
      const isLt10M = file.size / 1024 / 1024 < 10;
      if (!isLt10M) {
        this.$message.error('文件大小不能超过10MB!');
        return false;
      }
      // 保存文件到uploadForm
      this.uploadForm.file = file;
      return false; // 阻止自动上传，使用自定义上传逻辑
    },
    // 移除已选择的文件
    removeSelectedFile: function(index) {
      this.selectedFilesList.splice(index, 1);
    },
    
    // 格式化文件大小
    formatFileSize: function(bytes) {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    // 批量上传提交
    handleBatchUploadSubmit: function() {
      if (this.selectedFilesList.length === 0) {
        this.$message.error('请选择要上传的文件');
        return;
      }
      
      this.uploading = true;
      
      // 创建上传任务数组
      const uploadPromises = this.selectedFilesList.map(file => {
        return new Promise((resolve, reject) => {
          const formData = new FormData();
          formData.append('file', file.raw);
          
          KnowledgeBaseAPI.uploadDocument(this.datasetId, formData,
            ({data}) => {
              if (data && data.code === 0) {
                resolve({ success: true, fileName: file.name });
              } else {
                reject({ success: false, fileName: file.name, error: data?.msg || this.$t('knowledgeFileUpload.uploadFailed') });
              }
            },
            (err) => {
              reject({ success: false, fileName: file.name, error: this.$t('knowledgeFileUpload.uploadFailed') });
              console.error('上传文档失败:', err);
            }
          );
        });
      });
      
      // 执行所有上传任务
      Promise.all(uploadPromises.map(p => p.catch(e => e)))
        .then(results => {
          this.uploading = false;
          
          const successCount = results.filter(r => r.success).length;
          const failedCount = results.filter(r => !r.success).length;
          
          if (successCount > 0) {
            this.$message.success(`成功上传 ${successCount} 个文件`);
          }
          
          if (failedCount > 0) {
            const failedFiles = results.filter(r => !r.success).map(r => r.fileName);
            this.$message.error(`上传失败 ${failedCount} 个文件: ${failedFiles.join(', ')}`);
          }
          
          if (successCount > 0) {
            this.uploadDialogVisible = false;
            this.fetchFileList();
          }
        })
        .catch(error => {
          this.uploading = false;
          this.$message.error('批量上传失败');
          console.error('批量上传失败:', error);
        });
    },
    
    // 单文件上传（保留原有功能）
    handleUploadSubmit: function() {
      if (!this.uploadForm.file) {
        this.$message.error(this.$t('knowledgeFileUpload.fileRequired'));
        return;
      }
      
      this.uploading = true;
      
      const formData = new FormData();
      formData.append('file', this.uploadForm.file);
      
      KnowledgeBaseAPI.uploadDocument(this.datasetId, formData,
        ({data}) => {
          this.uploading = false;
          if (data && data.code === 0) {
            this.$message.success(this.$t('knowledgeFileUpload.uploadSuccess'));
            this.uploadDialogVisible = false;
            this.fetchFileList();
          } else {
            this.$message.error(data?.msg || this.$t('knowledgeFileUpload.uploadFailed'));
          }
        },
        (err) => {
          this.uploading = false;
          this.$message.error(this.$t('knowledgeFileUpload.uploadFailed'));
          console.error('上传文档失败:', err);
        }
      );
    },
    handleParse: function(row) {
      this.$confirm(this.$t('knowledgeFileUpload.confirmParse'), this.$t('warning'), {
        confirmButtonText: this.$t('knowledgeFileUpload.confirm'),
        cancelButtonText: this.$t('knowledgeFileUpload.cancel'),
        type: 'warning'
      }).then(() => {
        KnowledgeBaseAPI.parseDocument(this.datasetId, row.id,
          ({data}) => {
            if (data && data.code === 0) {
              this.$message.success('请求已提交，解析中');
              // 使用智能检测自动刷新切片数量
              this.smartRefreshSliceCount(row.id);
            } else {
              this.$message.error(data?.msg || this.$t('knowledgeFileUpload.parseFailed'));
            }
          },
          (err) => {
            this.$message.error(this.$t('knowledgeFileUpload.parseFailed'));
            console.error('解析文档失败:', err);
          }
        );
      }).catch(() => {
        this.$message.info(this.$t('knowledgeFileUpload.parseCancelled'));
      });
    },
    handleViewSlices: function(row) {
      // 查看切片
      this.currentDocumentId = row.id;
      this.currentDocumentName = row.name;
      this.sliceDialogVisible = true;
      this.sliceCurrentPage = 1;
      this.sliceSearchKeyword = '';
      this.fetchSlices();
    },
    handleDelete: function(row) {
      this.$confirm(this.$t('knowledgeFileUpload.confirmDelete'), this.$t('warning'), {
        confirmButtonText: this.$t('knowledgeFileUpload.confirm'),
        cancelButtonText: this.$t('knowledgeFileUpload.cancel'),
        type: 'warning'
      }).then(() => {
        KnowledgeBaseAPI.deleteDocument(this.datasetId, row.id,
          ({data}) => {
            if (data && data.code === 0) {
              this.$message.success(this.$t('knowledgeFileUpload.deleteSuccess'));
              this.fetchFileList();
            } else {
              this.$message.error(data?.msg || this.$t('knowledgeFileUpload.deleteFailed'));
            }
          },
          (err) => {
            this.$message.error(this.$t('knowledgeFileUpload.deleteFailed'));
            console.error('删除文档失败:', err);
          }
        );
      }).catch(() => {
        this.$message.info(this.$t('knowledgeFileUpload.deleteCancelled'));
      });
    },
    handleSelectionChange: function(selection) {
      this.selectedFiles = selection;
    },
    handleSelectAll: function() {
      if (this.isAllSelected) {
        this.$refs.fileTable.clearSelection();
        this.isAllSelected = false;
      } else {
        this.$refs.fileTable.clearSelection();
        this.fileList.forEach(row => {
          this.$refs.fileTable.toggleRowSelection(row, true);
        });
        this.isAllSelected = true;
      }
    },
    handleBatchDelete: function() {
      if (this.selectedFiles.length === 0) {
        this.$message.warning(this.$t('knowledgeFileUpload.selectFilesFirst'));
        return;
      }
      
      this.$confirm(this.$t('knowledgeFileUpload.confirmBatchDelete', { count: this.selectedFiles.length }), this.$t('warning'), {
        confirmButtonText: this.$t('knowledgeFileUpload.confirm'),
        cancelButtonText: this.$t('knowledgeFileUpload.cancel'),
        type: 'warning'
      }).then(() => {
        const deletePromises = this.selectedFiles.map(file => {
          return new Promise((resolve, reject) => {
            KnowledgeBaseAPI.deleteDocument(this.datasetId, file.id,
              ({data}) => {
                if (data && data.code === 0) {
                  resolve();
                } else {
                  reject(data?.msg || this.$t('knowledgeFileUpload.deleteFailed'));
                }
              },
              (err) => {
                reject(this.$t('knowledgeFileUpload.deleteFailed'));
                console.error('删除文档失败:', err);
              }
            );
          });
        });
        
        Promise.all(deletePromises)
          .then(() => {
            this.$message.success(this.$t('knowledgeFileUpload.batchDeleteSuccess', { count: this.selectedFiles.length }));
            this.selectedFiles = [];
            this.fetchFileList();
          })
          .catch((error) => {
            this.$message.error(error || this.$t('knowledgeFileUpload.batchDeleteFailed'));
          });
      }).catch(() => {
        this.$message.info(this.$t('knowledgeFileUpload.deleteCancelled'));
      });
    },
    getStatusType: function(status) {
      switch(status) {
        case 0:
          return 'info'; // 灰色 - 待解析
        case 1:
          return 'success'; // 绿色 - 解析成功
        case 2:
          return 'primary'; // 蓝色 - 解析中
        default:
          return 'danger'; // 红色 - 解析失败
      }
    },
    getStatusText: function(status) {
      switch(status) {
        case 0:
          return this.$t('knowledgeFileUpload.statusPending');
        case 1:
          return this.$t('knowledgeFileUpload.statusSuccess');
        case 2:
          return this.$t('knowledgeFileUpload.statusParsing');
        default:
          return this.$t('knowledgeFileUpload.statusFailed');
      }
    },
    goToPage: function(page) {
      if (page !== this.currentPage) {
        this.currentPage = page;
        this.fetchFileList();
      }
    },
    goFirst: function() {
      if (this.currentPage !== 1) {
        this.currentPage = 1;
        this.fetchFileList();
      }
    },
    goPrev: function() {
      if (this.currentPage > 1) {
        this.currentPage--;
        this.fetchFileList();
      }
    },
    goNext: function() {
      if (this.currentPage < this.pageCount) {
        this.currentPage++;
        this.fetchFileList();
      }
    },
    formatDate: function(dateString) {
      if (!dateString) return '';
      const date = new Date(dateString);
      return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
    },
    
    // 切片管理相关方法
    fetchSlices: function() {
      this.sliceLoading = true;
      
      const params = {
        page: this.sliceCurrentPage,
        page_size: this.slicePageSize
      };
      
      if (this.sliceSearchKeyword) {
        params.keywords = this.sliceSearchKeyword;
      }
      
      KnowledgeBaseAPI.listChunks(this.datasetId, this.currentDocumentId, params,
        ({data}) => {
          this.sliceLoading = false;
          if (data && data.code === 0) {
            // 解析切片列表数据
            this.parseSliceData(data.data);
          } else {
            this.$message.error(data?.msg || '获取切片列表失败');
            this.sliceList = [];
            this.sliceTotal = 0;
          }
        },
        (err) => {
          this.sliceLoading = false;
          this.$message.error('获取切片列表失败');
          console.error('获取切片列表失败:', err);
          this.sliceList = [];
          this.sliceTotal = 0;
        }
      );
    },
    
    parseSliceData: function(data) {
      try {
        if (data && data.list) {
          // 后端已经解析过的格式
          this.sliceList = data.list;
          this.sliceTotal = data.total || data.list.length;
        } else if (data && data.chunks && Array.isArray(data.chunks)) {
          // RAGFlow API原始格式
          this.sliceList = data.chunks;
          this.sliceTotal = data.total || data.chunks.length;
        } else if (data && Array.isArray(data)) {
          this.sliceList = data;
          this.sliceTotal = data.length;
        } else {
          this.sliceList = [];
          this.sliceTotal = 0;
        }
        
        console.log('解析后的切片数据:', {
          list: this.sliceList,
          total: this.sliceTotal
        });
      } catch (error) {
        console.error('解析切片数据失败:', error);
        this.sliceList = [];
        this.sliceTotal = 0;
      }
    },
    
    handleSliceSizeChange: function(pageSize) {
      this.slicePageSize = pageSize;
      this.sliceCurrentPage = 1;
      this.fetchSlices();
    },
    
    handleSlicePageChange: function(page) {
      this.sliceCurrentPage = page;
      this.fetchSlices();
    },
    
    handleSliceDialogClose: function(done) {
      this.sliceDialogVisible = false;
      this.currentDocumentId = null;
      this.currentDocumentName = '';
      this.sliceList = [];
      this.sliceTotal = 0;
      if (done) {
        done();
      }
    },
    
    // 跳转到切片管理第一页
    goToSliceFirstPage: function() {
      if (this.sliceCurrentPage !== 1) {
        this.sliceCurrentPage = 1;
        this.fetchSlices();
      }
    },
    
    // 切片管理上一页
    goToSlicePrevPage: function() {
      if (this.sliceCurrentPage > 1) {
        this.sliceCurrentPage--;
        this.fetchSlices();
      }
    },
    
    // 切片管理跳转到指定页
    goToSlicePage: function(page) {
      if (page !== this.sliceCurrentPage) {
        this.sliceCurrentPage = page;
        this.fetchSlices();
      }
    },
    
    // 切片管理下一页
    goToSliceNextPage: function() {
      if (this.sliceCurrentPage < this.slicePageCount) {
        this.sliceCurrentPage++;
        this.fetchSlices();
      }
    },
    
    // 召回测试相关方法
    showRetrievalTestDialog: function() {
      // 初始化召回测试表单
      this.retrievalTestForm = {
        question: ''
      };
      this.retrievalTestResult = null;
      this.retrievalTestDialogVisible = true;
    },
    
    runRetrievalTest: function() {
      if (!this.retrievalTestForm.question.trim()) {
        this.$message.error(this.$t('knowledgeFileUpload.testQuestionRequired'));
        return;
      }
      
      this.retrievalTestLoading = true;
      this.retrievalTestResult = null;
      
      // 准备请求数据
      const requestData = {
        question: this.retrievalTestForm.question.trim()
      };
      
      // 调用召回测试API
      KnowledgeBaseAPI.retrievalTest(this.datasetId, requestData,
        ({data}) => {
          this.retrievalTestLoading = false;
          if (data && data.code === 0) {
            this.retrievalTestResult = data.data || data;
            this.$message.success('召回测试完成');
          } else {
            this.$message.error(data?.msg || '召回测试失败');
          }
        },
        (err) => {
          this.retrievalTestLoading = false;
          this.$message.error('召回测试失败');
          console.error('召回测试失败:', err);
        }
      );
    },
    
    handleRetrievalTestDialogClose: function() {
      this.retrievalTestDialogVisible = false;
      this.retrievalTestResult = null;
    }
  }
};
</script>

<style lang="scss" scoped>
.welcome {
    min-width: 900px;
    min-height: 506px;
    height: 100vh;
    display: flex;
    position: relative;
    flex-direction: column;
    background-size: cover;
    background: linear-gradient(to bottom right, #dce8ff, #e4eeff, #e6cbfd) center;
    -webkit-background-size: cover;
    -o-background-size: cover;
    overflow: hidden;
}

.main-wrapper {
    margin: 5px 22px;
    border-radius: 15px;
    min-height: calc(100vh - 24vh);
    height: auto;
    max-height: 80vh;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    position: relative;
    background: rgba(237, 242, 255, 0.5);
    display: flex;
    flex-direction: column;
}

.operation-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
}

.left-operations {
    display: flex;
    align-items: center;
    gap: 16px;
}

.btn-back {
    font-size: 20px;
    font-weight: bold;
    color: #606266;
    padding: 8px 12px;
    border-radius: 4px;
    transition: all 0.3s ease;
}

.btn-back:hover {
    background-color: #f5f7fa;
    color: #409eff;
}

.knowledge-base-title {
    font-size: 24px;
    margin: 0;
    color: #303133;
    font-weight: 600;
}

.right-operations {
    display: flex;
    gap: 10px;
    margin-left: auto;
}

.search-input {
    width: 240px;
}

.btn-search {
    background: linear-gradient(135deg, #6b8cff, #a966ff);
    border: none;
    color: white;
}

.content-panel {
    flex: 1;
    display: flex;
    overflow: hidden;
    height: 100%;
    border-radius: 15px;
    background: transparent;
    border: 1px solid #fff;
}

.content-area {
    flex: 1;
    height: 100%;
    min-width: 600px;
    overflow: auto;
    background-color: white;
    display: flex;
    flex-direction: column;
}

.params-card {
    background: white;
    flex: 1;
    display: flex;
    flex-direction: column;
    border: none;
    box-shadow: none;
    overflow: hidden;

    ::v-deep .el-card__body {
        padding: 15px;
        display: flex;
        flex-direction: column;
        flex: 1;
        overflow: hidden;
    }
}

.table_bottom {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: auto; 
    padding-bottom: 10px;
    width: 100%;
}

.ctrl_btn {
    display: flex;
    gap: 8px;
    padding-left: 26px;

    .el-button {
        min-width: 72px;
        height: 32px;
        padding: 7px 12px 7px 10px;
        font-size: 12px;
        border-radius: 4px;
        line-height: 1;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);

        &:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
    }

    .el-button--primary {
        background: #5f70f3;
        color: white;
    }

    .el-button--danger {
        background: #fd5b63;
        color: white;
    }
}

.custom-pagination {
    display: flex;
    align-items: center;
    gap: 5px;

    .el-select {
        margin-right: 8px;
    }

    .pagination-btn:first-child,
    .pagination-btn:nth-child(2),
    .pagination-btn:nth-last-child(2),
    .pagination-btn:nth-child(3) {
        min-width: 60px;
        height: 32px;
        padding: 0 12px;
        border-radius: 4px;
        border: 1px solid #e4e7ed;
        background: #dee7ff;
        color: #606266;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;

        &:hover {
            background: #d7dce6;
        }

        &:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
    }

    .pagination-btn:not(:first-child):not(:nth-child(3)):not(:nth-child(2)):not(:nth-last-child(2)) {
        min-width: 28px;
        height: 32px;
        padding: 0;
        border-radius: 4px;
        border: 1px solid transparent;
        background: transparent;
        color: #606266;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;

        &:hover {
            background: rgba(245, 247, 250, 0.3);
        }
    }

    .pagination-btn.active {
        background: #5f70f3 !important;
        color: #ffffff !important;
        border-color: #5f70f3 !important;

        &:hover {
            background: #6d7cf5 !important;
        }
    }
}

.total-text {
    margin-left: 10px;
    color: #606266;
    font-size: 14px;
}

.page-size-select {
    width: 100px;
    margin-right: 10px;

    :deep(.el-input__inner) {
        height: 32px;
        line-height: 32px;
        border-radius: 4px;
        border: 1px solid #e4e7ed;
        background: #dee7ff;
        color: #606266;
        font-size: 14px;
    }

    :deep(.el-input__suffix) {
        right: 6px;
        width: 15px;
        height: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        top: 6px;
        border-radius: 4px;
    }

    :deep(.el-input__suffix-inner) {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }

    :deep(.el-icon-arrow-up:before) {
        content: "";
        display: inline-block;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 9px solid #606266;
        position: relative;
        transform: rotate(0deg);
        transition: transform 0.3s;
    }
}

:deep(.transparent-table) {
    background: white;
    flex: 1;
    width: 100%;
    display: flex;
    flex-direction: column;

    .el-table__body-wrapper {
        flex: 1;
        overflow-y: auto;
        max-height: none !important;
    }

    .el-table__header-wrapper {
        flex-shrink: 0;
    }

    .el-table__header th {
        background: white !important;
        color: black;
        font-weight: 600;
        height: 40px;
        padding: 8px 0;
        font-size: 14px;
        border-bottom: 1px solid #e4e7ed;
    }

    .el-table__body tr {
        background-color: white;

        td {
            border-top: 1px solid rgba(0, 0, 0, 0.04);
            border-bottom: 1px solid rgba(0, 0, 0, 0.04);
            padding: 8px 0;
            height: 40px;
            color: #606266;
            font-size: 14px;
        }
    }

    .el-table__row:hover>td {
        background-color: #f5f7fa !important;
    }

    &::before {
        display: none;
    }
}

:deep(.el-table .el-button--text) {
    color: #7079aa !important;
}

:deep(.el-table .el-button--text:hover) {
    color: #5a64b5 !important;
}

:deep(.el-checkbox__inner) {
    background-color: #eeeeee !important;
    border-color: #cccccc !important;
}

:deep(.el-checkbox__inner:hover) {
    border-color: #cccccc !important;
}

:deep(.el-checkbox__input.is-checked .el-checkbox__inner) {
    background-color: #5f70f3 !important;
    border-color: #5f70f3 !important;
}

:deep(.el-loading-mask) {
    background-color: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(2px);
}

:deep(.el-loading-spinner .path) {
    stroke: #6b8cff;
}

:deep(.el-table__empty-block) {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 300px;
}

/* 拖拽上传区域样式 */
.document-uploader {
    :deep(.el-upload-dragger) {
        width: 100%;
        height: 200px;
        min-height: 200px;
        border: 2px dashed #c0c4cc;
        border-radius: 16px;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease;
        
        &:hover {
            border-color: #409eff;
        }
        
        .el-icon-upload {
            font-size: 48px;
            color: #c0c4cc;
            margin: 20px 0 16px;
            line-height: 1;
        }
        
        .el-upload__text {
            font-size: 16px;
            color: #606266;
            margin-bottom: 8px;
            line-height: 1.5;
        }
        
        .el-upload__tip {
            font-size: 14px;
            color: #909399;
            line-height: 1.5;
        }
    }
}

/* 召回测试弹窗样式 */
.retrieval-test-dialog {
  :deep(.el-dialog) {
    width: 1000px !important;
    max-height: 850px !important;
    min-height: 600px !important;
  }
  
  :deep(.el-dialog__body) {
    height: calc(100% - 90px) !important;
    max-height: calc(100% - 90px) !important;
    overflow: hidden;
    padding: 15px 25px;
  }
}

.retrieval-test-result {
  margin-top: 20px;
  height: 300px;
  max-height: 300px;
  min-height: 200px;
  display: flex;
  flex-direction: column;
  
  h4 {
    margin: 0 0 12px 0;
    font-size: 16px;
    font-weight: 600;
    color: #303133;
    flex-shrink: 0;
  }
  
  :deep(.el-card) {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    
    .el-card__body {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      padding: 0;
    }
  }
}

.result-chunk-container {
  flex: 1;
  overflow-y: auto;
  max-height: 100%;
  padding: 16px;
  
  /* 滚动条样式 */
  &::-webkit-scrollbar {
    width: 6px;
    height: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: #fafafa;
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #e0e0e0;
    border-radius: 3px;
    border: 1px solid #f0f0f0;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: #d0d0d0;
  }
}

.result-chunk {
  background-color: #f8f9fa;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
  
  &:last-child {
    margin-bottom: 0;
  }
  
  p {
    margin: 0 0 8px 0;
    font-size: 14px;
    line-height: 1.5;
    
    &:last-child {
      margin-bottom: 0;
    }
    
    strong {
      color: #303133;
      font-weight: 600;
    }
  }
  
  .similarity-scores {
    display: flex;
    gap: 16px;
    margin-bottom: 12px;
    
    .score-item {
      display: flex;
      flex-direction: column;
      
      .score-label {
        font-size: 12px;
        color: #909399;
        margin-bottom: 4px;
      }
      
      .score-value {
        font-size: 14px;
        font-weight: 600;
        color: #409eff;
      }
    }
  }
  
  .chunk-content {
    background-color: white;
    border: 1px solid #f0f0f0;
    border-radius: 4px;
    padding: 12px;
    margin-top: 8px;
    max-height: 120px;
    overflow-y: auto;
    
    /* 内容滚动条样式 */
    &::-webkit-scrollbar {
      width: 4px;
    }
    
    &::-webkit-scrollbar-track {
      background: #fafafa;
    }
    
    &::-webkit-scrollbar-thumb {
      background: #e0e0e0;
      border-radius: 2px;
    }
    
    &::-webkit-scrollbar-thumb:hover {
      background: #d0d0d0;
    }
  }
  
  :deep(.el-divider) {
    margin: 12px 0;
  }
}

/* 已选择文件列表样式 */
.selected-files-section {
  margin-top: 20px;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 16px;
  background-color: #f8f9fa;
  
  h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    color: #606266;
  }
}

.selected-files-list {
  max-height: 200px;
  overflow-y: auto;
}

.selected-file-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background-color: white;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  margin-bottom: 8px;
  
  &:last-child {
    margin-bottom: 0;
  }
  
  .file-info {
    display: flex;
    align-items: center;
    flex: 1;
    
    .el-icon-document {
      color: #409eff;
      margin-right: 8px;
      font-size: 16px;
    }
    
    .file-name {
      font-size: 14px;
      color: #303133;
      margin-right: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 300px;
    }
    
    .file-size {
      font-size: 12px;
      color: #909399;
    }
  }
  
  .remove-btn {
    color: #f56c6c;
    padding: 4px;
    
    &:hover {
      color: #f78989;
      background-color: #fef0f0;
      border-radius: 4px;
    }
  }
}

/* 上传对话框容器样式 */
:deep(.el-dialog) {
    border-radius: 16px !important;
    overflow: hidden;
}

.el-table {
    --table-max-height: calc(100vh - 40vh);
    max-height: var(--table-max-height);
    flex: 1;

    .el-table__body-wrapper {
        max-height: calc(var(--table-max-height) - 40px);
    }
}

  .slice-dialog-content {
    max-height: 70vh;
    overflow-y: auto;
  }
  
  /* 切片管理弹窗固定容器大小 */
  .slice-dialog {
    /* 切片管理弹窗滚动条样式 */
    :deep(.el-dialog)::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    
    :deep(.el-dialog)::-webkit-scrollbar-track {
      background: #f8f9fa;
      border-radius: 4px;
    }
    
    :deep(.el-dialog)::-webkit-scrollbar-thumb {
      background: #f0f0f0;
      border-radius: 4px;
      border: 1px solid #e8e8e8;
    }
    
    :deep(.el-dialog)::-webkit-scrollbar-thumb:hover {
      background: #e8e8e8;
    }
    
    :deep(.el-dialog) {
      width: 1200px !important;
      height: 800px !important;
      max-height: 600px !important;
      min-height: 600px !important;
    }
    
    :deep(.el-dialog__body) {
      height: calc(100% - 90px) !important;
      max-height: calc(100% - 90px) !important;
      overflow: hidden;
      padding: 15px 20px;
    }
    
    .slice-management {
      height: 100%;
      display: flex;
      flex-direction: column;
    }
    
    .slice-list-section {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
    }
    
    /* 切片内容容器样式 */
  .slice-content-container {
    flex: 1;
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  
  .slice-cards-container {
    flex: 1;
    overflow-y: auto;
    max-height: 100%;
    padding: 16px;
    
    /* 滚动条样式 */
    &::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    
    &::-webkit-scrollbar-track {
      background: #fafafa;
      border-radius: 3px;
    }
    
    &::-webkit-scrollbar-thumb {
      background: #e0e0e0;
      border-radius: 3px;
      border: 1px solid #f0f0f0;
    }
    
    &::-webkit-scrollbar-thumb:hover {
      background: #d0d0d0;
    }
  }
  
  .slice-card {
    background-color: #f8f9fa;
    border: 1px solid #e4e7ed;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    
    &:last-child {
      margin-bottom: 0;
    }
    
    .slice-header-info {
      margin-bottom: 12px;
      
      p {
        margin: 0;
        font-size: 14px;
        line-height: 1.5;
        
        strong {
          color: #303133;
          font-weight: 600;
        }
      }
    }
    
    .slice-card-content {
      background-color: white;
      border: 1px solid #f0f0f0;
      border-radius: 4px;
      padding: 12px;
      max-height: 280px;
      min-height: 120px;
      overflow-y: auto;
      
      /* 内容滚动条样式 */
      &::-webkit-scrollbar {
        width: 4px;
      }
      
      &::-webkit-scrollbar-track {
        background: #fafafa;
      }
      
      &::-webkit-scrollbar-thumb {
        background: #e0e0e0;
        border-radius: 2px;
      }
      
      &::-webkit-scrollbar-thumb:hover {
        background: #d0d0d0;
      }
      
      .content-text {
        font-size: 14px;
        line-height: 1.6;
        text-align: left;
        color: #333;
        word-wrap: break-word;
        white-space: pre-wrap;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
        
        /* 确保文本正常显示，包括空格和换行 */
        white-space: pre-wrap;
        word-break: break-word;
        overflow-wrap: break-word;
        
        /* 段落样式 */
        p {
          margin: 0 0 12px 0;
          line-height: 1.6;
          
          &:last-child {
            margin-bottom: 0;
          }
        }
        
        /* 列表样式 */
        ul, ol {
          margin: 8px 0;
          padding-left: 24px;
          
          li {
            margin-bottom: 4px;
            line-height: 1.5;
          }
        }
        
        /* 标题样式 */
        h1, h2, h3, h4, h5, h6 {
          margin: 16px 0 8px 0;
          font-weight: 600;
          line-height: 1.4;
        }
        
        h1 { font-size: 18px; }
        h2 { font-size: 16px; }
        h3 { font-size: 15px; }
        h4, h5, h6 { font-size: 14px; }
        
        /* 强调文本 */
        strong, b {
          font-weight: 600;
          color: #1a1a1a;
        }
        
        /* 代码样式 */
        code {
          background-color: #f5f5f5;
          padding: 2px 4px;
          border-radius: 3px;
          font-family: 'Courier New', monospace;
          font-size: 13px;
        }
        
        /* 引用样式 */
        blockquote {
          margin: 12px 0;
          padding: 8px 12px;
          border-left: 4px solid #e0e0e0;
          background-color: #f9f9f9;
          font-style: italic;
          color: #666;
        }
      }
    }
  }
  
  .no-slice-data {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
  }
  }
  
  .slice-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
  }
  
  .slice-info {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .slice-count {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 14px;
    color: #606266;
    background-color: #f5f7fa;
    padding: 6px 12px;
    border-radius: 4px;
    border: 1px solid #e4e7ed;
  }
  
  .count-label {
    font-weight: 500;
    color: #303133;
  }
  
  .count-value {
    font-weight: 600;
    color: #409eff;
    font-size: 15px;
  }
  
  .count-unit {
    color: #909399;
  }
  
  .slice-actions {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  
  .slice-table {
    margin-bottom: 20px;
  }
  
  .slice-pagination {
    text-align: right;
    margin-top: 20px;
    
    .custom-pagination {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 5px;
    }
    
    :deep(.el-pagination) {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      
      .el-pagination__sizes {
        margin-right: 8px;
        
        .el-input__inner {
          height: 32px;
          line-height: 32px;
          border-radius: 4px;
          border: 1px solid #e4e7ed;
          background: #dee7ff;
          color: #606266;
          font-size: 14px;
        }
        
        .el-input__suffix {
          right: 6px;
          width: 15px;
          height: 20px;
          display: flex;
          justify-content: center;
          align-items: center;
          top: 6px;
          border-radius: 4px;
        }
        
        .el-input__suffix-inner {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 100%;
        }
        
        .el-icon-arrow-up:before {
          content: "";
          display: inline-block;
          border-left: 6px solid transparent;
          border-right: 6px solid transparent;
          border-top: 9px solid #606266;
          position: relative;
          transform: rotate(0deg);
          transition: transform 0.3s;
        }
      }
      
      .btn-prev, .btn-next {
        min-width: 60px;
        height: 32px;
        padding: 0 12px;
        border-radius: 4px;
        border: 1px solid #e4e7ed;
        background: #dee7ff;
        color: #606266;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
        
        &:hover {
          background: #d7dce6;
        }
        
        &:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
      }
      
      .el-pager {
        display: flex;
        gap: 4px;
        
        .number {
          min-width: 28px;
          height: 32px;
          padding: 0;
          border-radius: 4px;
          border: 1px solid transparent;
          background: transparent;
          color: #606266;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.3s ease;
          
          &:hover {
            background: rgba(245, 247, 250, 0.3);
          }
          
          &.active {
            background: #5f70f3 !important;
            color: #ffffff !important;
            border-color: #5f70f3 !important;
            
            &:hover {
              background: #6d7cf5 !important;
            }
          }
        }
      }
      
      .el-pagination__jump {
        margin-left: 8px;
        color: #606266;
        font-size: 14px;
        
        .el-pagination__editor {
          width: 50px;
          margin: 0 8px;
          
          .el-input__inner {
            height: 32px;
            line-height: 32px;
            border-radius: 4px;
            border: 1px solid #e4e7ed;
            background: #dee7ff;
            color: #606266;
            font-size: 14px;
          }
        }
      }
      
      .el-pagination__total {
        margin-right: 8px;
        color: #606266;
        font-size: 14px;
      }
    }
  }
  
  .slice-content {
    max-height: 300px;
    min-height: 60px;
    overflow-y: auto;
    word-break: break-all;
    line-height: 1.4;
    padding: 8px 12px;
    border: 1px solid #e4e7ed;
    border-radius: 4px;
    background-color: #f8f9fa;
    white-space: pre-wrap;
  }
  
  .slice-keywords, .slice-questions {
    max-height: 60px;
    overflow-y: auto;
  }
  
  .slice-tag {
    margin: 2px;
  }
  
  .add-slice-form {
    padding: 20px 0;
  }
  
  .form-item {
    margin-bottom: 20px;
  }
  
  .form-actions {
    text-align: right;
    margin-top: 20px;
  }

@media (min-width: 1144px) {
    .table_bottom {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: auto;
    }

    :deep(.transparent-table) {
        .el-table__body tr {
            td {
                padding-top: 16px;
                padding-bottom: 16px;
            }

            &+tr {
                margin-top: 10px;
            }
        }
    }
}
</style>
