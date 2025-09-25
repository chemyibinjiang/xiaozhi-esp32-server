<template>
  <div class="welcome" @keyup.enter="register">
    <el-container style="height: 100%;">
      <!-- 保持相同的头部 -->
      <el-header>
        <div style="display: flex;align-items: center;margin-top: 15px;margin-left: 10px;gap: 10px;">
          <img loading="lazy" alt="" src="@/assets/xiaozhi-logo.png" style="width: 45px;height: 45px;" />
          <img loading="lazy" alt="" src="@/assets/xiaozhi-ai.png" style="height: 18px;" />
        </div>
      </el-header>
      <div class="login-person">
        <img loading="lazy" alt="" src="@/assets/login/register-person.png" style="width: 100%;" />
      </div>
      <el-main style="position: relative;">
        <div class="login-box">
          <!-- 修改标题部分 -->
          <div style="display: flex;align-items: center;gap: 20px;margin-bottom: 39px;padding: 0 30px;">
            <img loading="lazy" alt="" src="@/assets/login/hi.png" style="width: 34px;height: 34px;" />
            <div class="login-text">{{ $t('register.title') }}</div>
            <div class="login-welcome">
              {{ $t('register.welcome') }}
            </div>
          </div>

          <div style="padding: 0 30px;">
            <form @submit.prevent="register">
              <!-- 用户名/手机号输入框 -->
              <div class="input-box" v-if="!enableMobileRegister">
                <img loading="lazy" alt="" class="input-icon" src="@/assets/login/username.png" />
                <el-input v-model="form.username" :placeholder="$t('register.usernamePlaceholder')" />
              </div>

              <!-- 手机号注册部分 -->
              <template v-if="enableMobileRegister">
                <div class="input-box">
                  <div style="display: flex; align-items: center; width: 100%;">
                    <el-select v-model="form.areaCode" style="width: 220px; margin-right: 10px;">
                      <el-option v-for="item in mobileAreaList" :key="item.key" :label="`${item.name} (${item.key})`"
                        :value="item.key" />
                    </el-select>
                    <el-input v-model="form.mobile" :placeholder="$t('register.mobilePlaceholder')" />
                  </div>
                </div>

                <div style="display: flex; align-items: center; margin-top: 20px; width: 100%; gap: 10px;">
                  <div class="input-box" style="width: calc(100% - 130px); margin-top: 0;">
                    <img loading="lazy" alt="" class="input-icon" src="@/assets/login/shield.png" />
                    <el-input v-model="form.captcha" :placeholder="$t('register.captchaPlaceholder')" style="flex: 1;" />
                  </div>
                  <img loading="lazy" v-if="captchaUrl" :src="captchaUrl" alt="验证码" 
                    style="width: 150px; height: 40px; cursor: pointer;" @click="fetchCaptcha" />
                </div>

                <!-- 手机验证码 -->

                <div style="display: flex; align-items: center; margin-top: 20px; width: 100%; gap: 10px;">
                  <div class="input-box" style="width: calc(100% - 130px); margin-top: 0;">
                    <img loading="lazy" alt="" class="input-icon" src="@/assets/login/phone.png" />
                    <el-input v-model="form.mobileCaptcha" :placeholder="$t('register.mobileCaptchaPlaceholder')" style="flex: 1;" maxlength="6" />
                  </div>
                  <el-button type="primary" class="send-captcha-btn" :disabled="!canSendMobileCaptcha"
                    @click="sendMobileCaptcha">
                    <span>
                      {{ countdown > 0 ? `${countdown}${$t('register.secondsLater')}` : $t('register.sendCaptcha') }}
                    </span>
                  </el-button>
                </div>
              </template>

              <!-- 密码输入框 -->
              <div class="input-box">
                <img loading="lazy" alt="" class="input-icon" src="@/assets/login/password.png" />
                <el-input v-model="form.password" :placeholder="$t('register.passwordPlaceholder')" type="password" show-password />
              </div>

              <!-- 新增确认密码 -->
              <div class="input-box">
                <img loading="lazy" alt="" class="input-icon" src="@/assets/login/password.png" />
                <el-input v-model="form.confirmPassword" :placeholder="$t('register.confirmPasswordPlaceholder')" type="password" show-password />
              </div>

              <!-- 验证码部分保持相同 -->
              <div v-if="!enableMobileRegister"
                style="display: flex; align-items: center; margin-top: 20px; width: 100%; gap: 10px;">
                <div class="input-box" style="width: calc(100% - 130px); margin-top: 0;">
                  <img loading="lazy" alt="" class="input-icon" src="@/assets/login/shield.png" />
                  <el-input v-model="form.captcha" :placeholder="$t('register.captchaPlaceholder')" style="flex: 1;" />
                </div>
                <img loading="lazy" v-if="captchaUrl" :src="captchaUrl" alt="验证码"
                  style="width: 150px; height: 40px; cursor: pointer;" @click="fetchCaptcha" />
              </div>

              <!-- 修改底部链接 -->
              <div style="font-weight: 400;font-size: 14px;text-align: left;color: #5778ff;margin-top: 20px;">
                <div style="cursor: pointer;" @click="goToLogin">{{ $t('register.goToLogin') }}</div>
              </div>
            </form>
          </div>

          <!-- 修改按钮文本 -->
          <div class="login-btn" @click="register">{{ $t('register.registerButton') }}</div>

          <!-- 保持相同的协议声明 -->
          <div style="font-size: 14px;color: #979db1;">
            {{ $t('register.agreeTo') }}
            <div style="display: inline-block;color: #5778FF;cursor: pointer;">{{ $t('register.userAgreement') }}</div>
            {{ $t('register.and') }}
            <div style="display: inline-block;color: #5778FF;cursor: pointer;">{{ $t('register.privacyPolicy') }}</div>
          </div>
        </div>
      </el-main>

      <!-- 保持相同的页脚 -->
      <el-footer>
        <version-footer />
      </el-footer>
    </el-container>
  </div>
</template>

<script>
import Api from '@/apis/api';
import VersionFooter from '@/components/VersionFooter.vue';
import { getUUID, goToPage, showDanger, showSuccess, validateMobile, generateSm2KeyPairHex, sm2Encrypt, isBase64 } from '@/utils';
import { mapState } from 'vuex';
import Constant from '@/utils/constant';

// 导入语言切换功能
import { changeLanguage } from '@/i18n';

export default {
  name: 'register',
  components: {
    VersionFooter
  },
  computed: {
    ...mapState({
      allowUserRegister: state => state.pubConfig.allowUserRegister,
      enableMobileRegister: state => state.pubConfig.enableMobileRegister,
      mobileAreaList: state => state.pubConfig.mobileAreaList
    }),
    canSendMobileCaptcha() {
      return this.countdown === 0 && validateMobile(this.form.mobile, this.form.areaCode);
    }
  },
  data() {
    return {
      form: {
        username: '',
        password: '',
        confirmPassword: '',
        captcha: '',
        captchaId: '',
        areaCode: '+86',
        mobile: '',
        mobileCaptcha: ''
      },
      captchaUrl: '',
      countdown: 0,
      timer: null,
      serverPublicKey: "", // 服务器公钥
      clientKeyPair: null, // 客户端密钥对
      isGettingPublicKey: false // 获取公钥状态
    }
  },
  mounted() {
    this.$store.dispatch('fetchPubConfig').then(() => {
      if (!this.allowUserRegister) {
        showDanger(this.$t('register.notAllowRegister'));
        setTimeout(() => {
          goToPage('/login');
        }, 1500);
      }
    });
    this.fetchCaptcha();
    // 获取服务器公钥
    this.getServerPublicKey();
    // 生成客户端密钥对
    this.generateClientKeyPair();
  },
  methods: {
    // 获取服务器公钥
    getServerPublicKey() {
      console.log('开始获取服务器公钥...');
      // 先从本地存储获取
      const storedPublicKey = localStorage.getItem(Constant.STORAGE_KEY.PUBLIC_KEY);
      if (storedPublicKey) {
        console.log('从本地存储获取到公钥，长度:', storedPublicKey.length);
        this.serverPublicKey = storedPublicKey;
        return;
      }
      
      console.log('本地存储无公钥，从服务器获取...');
      // 从服务器获取公钥
      Api.user.getSM2PublicKey(
        (res) => {
          if (res.data && res.data.data) {
            console.log('获取到服务器公钥，长度:', res.data.data.length);
            this.serverPublicKey = res.data.data;
            // 存储到本地
            localStorage.setItem(Constant.STORAGE_KEY.PUBLIC_KEY, this.serverPublicKey);
            console.log('公钥已存储到本地');
          } else {
            console.error('服务器返回数据格式异常:', res);
          }
        },
        (err) => {
          console.error("获取SM2公钥失败:", err);
          showDanger(this.$t('sm2.failedToGetPublicKey'));
        }
      );
    },
    
    // 生成客户端密钥对
    generateClientKeyPair() {
      this.clientKeyPair = generateSm2KeyPairHex();
    },

    // 复用验证码获取方法
    fetchCaptcha() {
      this.form.captchaId = getUUID();
      Api.user.getCaptcha(this.form.captchaId, (res) => {
        if (res.status === 200) {
          const blob = new Blob([res.data], { type: res.data.type });
          this.captchaUrl = URL.createObjectURL(blob);

        } else {
          console.error('验证码加载异常:', error);
          showDanger(this.$t('register.captchaLoadFailed'));
        }
      });
    },

    // 封装输入验证逻辑
    validateInput(input, message) {
      if (!input.trim()) {
        showDanger(message);
        return false;
      }
      return true;
    },

    // 发送手机验证码
    sendMobileCaptcha() {
      if (!validateMobile(this.form.mobile, this.form.areaCode)) {
        showDanger(this.$t('register.inputCorrectMobile'));
        return;
      }

      // 验证图形验证码
      if (!this.validateInput(this.form.captcha, this.$t('register.inputCaptcha'))) {
        this.fetchCaptcha();
        return;
      }

      // 清除可能存在的旧定时器
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
      }

      // 开始倒计时
      this.countdown = 60;
      this.timer = setInterval(() => {
        if (this.countdown > 0) {
          this.countdown--;
        } else {
          clearInterval(this.timer);
          this.timer = null;
        }
      }, 1000);

      // 调用发送验证码接口
      Api.user.sendSmsVerification({
        phone: this.form.areaCode + this.form.mobile,
        captcha: this.form.captcha,
        captchaId: this.form.captchaId
      }, (res) => {
        showSuccess(this.$t('register.captchaSendSuccess'));
      }, (err) => {
        showDanger(err.data.msg || this.$t('register.captchaSendFailed'));
        this.countdown = 0;
        this.fetchCaptcha();
      });
    },

    // 注册逻辑
    async register() {
      if (this.enableMobileRegister) {
        // 手机号注册验证
        if (!validateMobile(this.form.mobile, this.form.areaCode)) {
          showDanger(this.$t('register.inputCorrectMobile'));
          return;
        }
        if (!this.form.mobileCaptcha) {
          showDanger(this.$t('register.requiredMobileCaptcha'));
          return;
        }
      } else {
        // 用户名注册验证
        if (!this.validateInput(this.form.username, this.$t('register.requiredUsername'))) {
          return;
        }
      }

      // 验证密码
      if (!this.validateInput(this.form.password, this.$t('register.requiredPassword'))) {
        return;
      }
      if (this.form.password !== this.form.confirmPassword) {
        showDanger(this.$t('register.passwordsNotMatch'))
        return
      }
      // 验证验证码
      if (!this.validateInput(this.form.captcha, this.$t('register.requiredCaptcha'))) {
        return;
      }

      // 检查服务器公钥是否已获取，如果未获取则重新获取
      if (!this.serverPublicKey) {
        try {
          // 等待公钥获取完成
          await new Promise((resolve, reject) => {
            this.getServerPublicKey();
            // 设置超时检查，最多等待3秒
            const checkInterval = setInterval(() => {
              if (this.serverPublicKey) {
                clearInterval(checkInterval);
                resolve();
              }
            }, 100);
            
            setTimeout(() => {
              clearInterval(checkInterval);
              if (!this.serverPublicKey) {
                reject(new Error('获取公钥超时'));
              }
            }, 3000);
          });
        } catch (error) {
          showDanger(this.$t('sm2.failedToGetPublicKey'));
          return;
        }
      }

      // 加密密码
      let encryptedPassword = this.form.password;
      if (!this.isSM2Encrypted(this.form.password)) {
        try {
          encryptedPassword = sm2Encrypt(this.serverPublicKey, this.form.password);
        } catch (error) {
          console.error("密码加密失败:", error);
          showDanger(this.$t('sm2.encryptionFailed'));
          return;
        }
      }

      // 加密用户名
      let encryptedUsername = this.form.username;
      if (this.enableMobileRegister) {
        this.form.username = this.form.areaCode + this.form.mobile;
        encryptedUsername = this.form.username;
      }
      
      if (!this.isSM2Encrypted(encryptedUsername)) {
        try {
          encryptedUsername = sm2Encrypt(this.serverPublicKey, encryptedUsername);
        } catch (error) {
          console.error("用户名加密失败:", error);
          showDanger(this.$t('sm2.encryptionFailed'));
          return;
        }
      }

      // 准备注册数据
      const registerData = {
        ...this.form,
        username: encryptedUsername,
        password: encryptedPassword,
        confirmPassword: encryptedPassword
      };

      Api.user.register(registerData, ({ data }) => {
        showSuccess(this.$t('register.registerSuccess'))
        goToPage('/login')
      }, (err) => {
        showDanger(err.data.msg || this.$t('register.registerFailed'))
        if (err.data != null && err.data.msg != null && err.data.msg.indexOf('图形验证码') > -1) {
          this.fetchCaptcha()
        }
      })
    },

    goToLogin() {
      goToPage('/login')
    },
    
    /**
     * 判断字符串是否为SM2加密格式（十六进制格式）
     * @param {string} str 待判断的字符串
     * @returns {boolean} 是否为SM2加密格式
     */
    isSM2Encrypted(str) {
      if (typeof str !== 'string' || str.trim() === '') {
        return false;
      }
      // 长度大于100且只包含0-9,a-f,A-F字符
      return str.length > 100 && /^[0-9a-fA-F]+$/.test(str);
    }
  },
  beforeDestroy() {
    if (this.timer) {
      clearInterval(this.timer);
    }
  }
}
</script>

<style lang="scss" scoped>
@import './auth.scss';

.send-captcha-btn {
  margin-right: -5px;
  min-width: 100px;
  height: 40px;
  line-height: 40px;
  border-radius: 4px;
  font-size: 14px;
  background: rgb(87, 120, 255);
  border: none;
  padding: 0px;

  &:disabled {
    background: #c0c4cc;
    cursor: not-allowed;
  }
}
</style>
