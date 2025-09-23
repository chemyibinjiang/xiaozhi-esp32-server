<template>
  <div class="welcome">
    <el-container style="height: 100%">
      <el-header>
        <div style="
            display: flex;
            align-items: center;
            margin-top: 15px;
            margin-left: 10px;
            gap: 10px;
          ">
          <img loading="lazy" alt="" src="@/assets/xiaozhi-logo.png" style="width: 45px; height: 45px" />
          <img loading="lazy" alt="" src="@/assets/xiaozhi-ai.png" style="height: 18px" />
        </div>
      </el-header>
      <div class="login-person">
        <img loading="lazy" alt="" src="@/assets/login/login-person.png" style="width: 100%" />
      </div>
      <el-main style="position: relative">
        <div class="login-box" @keyup.enter="login">
          <div style="
              display: flex;
              align-items: center;
              gap: 20px;
              margin-bottom: 39px;
              padding: 0 30px;
            ">
            <img loading="lazy" alt="" src="@/assets/login/hi.png" style="width: 34px; height: 34px" />
            <div class="login-text">{{ $t("login.title") }}</div>

            <div class="login-welcome">
              {{ $t("login.welcome") }}
            </div>

            <!-- 语言切换下拉菜单 -->
            <el-dropdown trigger="click" class="title-language-dropdown"
              @visible-change="handleLanguageDropdownVisibleChange">
              <span class="el-dropdown-link">
                <span class="current-language-text">{{ currentLanguageText }}</span>
                <i class="el-icon-arrow-down el-icon--right" :class="{ 'rotate-down': languageDropdownVisible }"></i>
              </span>
              <el-dropdown-menu slot="dropdown">
                <el-dropdown-item @click.native="changeLanguage('zh_CN')">
                  {{ $t("language.zhCN") }}
                </el-dropdown-item>
                <el-dropdown-item @click.native="changeLanguage('zh_TW')">
                  {{ $t("language.zhTW") }}
                </el-dropdown-item>
                <el-dropdown-item @click.native="changeLanguage('en')">
                  {{ $t("language.en") }}
                </el-dropdown-item>
              </el-dropdown-menu>
            </el-dropdown>
          </div>
          <div style="padding: 0 30px">
            <!-- 用户名登录 -->
            <template v-if="!isMobileLogin">
              <div class="input-box">
                <img loading="lazy" alt="" class="input-icon" src="@/assets/login/username.png" />
                <el-input v-model="form.username" :placeholder="$t('login.usernamePlaceholder')" />
              </div>
            </template>

            <!-- 手机号登录 -->
            <template v-else>
              <div class="input-box">
                <div style="display: flex; align-items: center; width: 100%">
                  <el-select v-model="form.areaCode" style="width: 220px; margin-right: 10px">
                    <el-option v-for="item in mobileAreaList" :key="item.key" :label="`${item.name} (${item.key})`"
                      :value="item.key" />
                  </el-select>
                  <el-input v-model="form.mobile" :placeholder="$t('login.mobilePlaceholder')" />
                </div>
              </div>
            </template>

            <div class="input-box">
              <img loading="lazy" alt="" class="input-icon" src="@/assets/login/password.png" />
              <el-input v-model="form.password" :placeholder="$t('login.passwordPlaceholder')" type="password"
                show-password />
            </div>
            <div style="
                display: flex;
                align-items: center;
                margin-top: 20px;
                width: 100%;
                gap: 10px;
              ">
              <div class="input-box" style="width: calc(100% - 130px); margin-top: 0">
                <img loading="lazy" alt="" class="input-icon" src="@/assets/login/shield.png" />
                <el-input v-model="form.captcha" :placeholder="$t('login.captchaPlaceholder')" style="flex: 1" />
              </div>
              <img loading="lazy" v-if="captchaUrl" :src="captchaUrl" alt="验证码"
                style="width: 150px; height: 40px; cursor: pointer" @click="fetchCaptcha" />
            </div>
            <div style="
                font-weight: 400;
                font-size: 14px;
                text-align: left;
                color: #5778ff;
                display: flex;
                justify-content: space-between;
                margin-top: 20px;
              ">
              <div v-if="allowUserRegister" style="cursor: pointer" @click="goToRegister">
                {{ $t("login.register") }}
              </div>
              <div style="cursor: pointer" @click="goToForgetPassword" v-if="enableMobileRegister">
                {{ $t("login.forgetPassword") }}
              </div>
            </div>
          </div>
          <div class="login-btn" @click="login">{{ $t("login.login") }}</div>

          <!-- 登录方式切换按钮 -->
          <div class="login-type-container" v-if="enableMobileRegister">
            <div style="display: flex; gap: 10px">
              <el-tooltip :content="$t('login.mobileLogin')" placement="bottom">
                <el-button :type="isMobileLogin ? 'primary' : 'default'" icon="el-icon-mobile" circle
                  @click="switchLoginType('mobile')"></el-button>
              </el-tooltip>
              <el-tooltip :content="$t('login.usernameLogin')" placement="bottom">
                <el-button :type="!isMobileLogin ? 'primary' : 'default'" icon="el-icon-user" circle
                  @click="switchLoginType('username')"></el-button>
              </el-tooltip>
            </div>
          </div>
          <div style="font-size: 14px; color: #979db1">
            {{ $t("login.agreeTo") }}
            <div style="display: inline-block; color: #5778ff; cursor: pointer">
              {{ $t("login.userAgreement") }}
            </div>
            {{ $t("login.and") }}
            <div style="display: inline-block; color: #5778ff; cursor: pointer">
              {{ $t("login.privacyPolicy") }}
            </div>
          </div>
        </div>
      </el-main>
      <el-footer>
        <version-footer />
      </el-footer>
    </el-container>
  </div>
</template>

<script>
import Api from "@/apis/api";
import VersionFooter from "@/components/VersionFooter.vue";
import i18n, { changeLanguage } from "@/i18n";
import { getUUID, goToPage, showDanger, showSuccess, validateMobile, generateSm2KeyPairHex, sm2Encrypt, isBase64 } from "@/utils";
import { mapState } from "vuex";
import Constant from "@/utils/constant";

export default {
  name: "login",
  components: {
    VersionFooter,
  },
  computed: {
    ...mapState({
      allowUserRegister: (state) => state.pubConfig.allowUserRegister,
      enableMobileRegister: (state) => state.pubConfig.enableMobileRegister,
      mobileAreaList: (state) => state.pubConfig.mobileAreaList,
    }),
    // 获取当前语言
    currentLanguage() {
      return i18n.locale || "zh_CN";
    },
    // 获取当前语言显示文本
    currentLanguageText() {
      const currentLang = this.currentLanguage;
      switch (currentLang) {
        case "zh_CN":
          return this.$t("language.zhCN");
        case "zh_TW":
          return this.$t("language.zhTW");
        case "en":
          return this.$t("language.en");
        default:
          return this.$t("language.zhCN");
      }
    },
  },
  data() {
    return {
      activeName: "username",
      form: {
        username: "",
        password: "",
        captcha: "",
        captchaId: "",
        areaCode: "+86",
        mobile: "",
      },
      captchaUuid: "",
      captchaUrl: "",
      isMobileLogin: false,
      languageDropdownVisible: false,
      serverPublicKey: "", // 服务器公钥
      clientKeyPair: null, // 客户端密钥对
    };
  },
  mounted() {
    this.fetchCaptcha();
    this.$store.dispatch("fetchPubConfig").then(() => {
      // 根据配置决定默认登录方式
      this.isMobileLogin = this.enableMobileRegister;
    });
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
    
    fetchCaptcha() {
      if (this.$store.getters.getToken) {
        if (this.$route.path !== "/home") {
          this.$router.push("/home");
        }
      } else {
        this.captchaUuid = getUUID();

        Api.user.getCaptcha(this.captchaUuid, (res) => {
          if (res.status === 200) {
            const blob = new Blob([res.data], { type: res.data.type });
            this.captchaUrl = URL.createObjectURL(blob);
          } else {
            showDanger("验证码加载失败，点击刷新");
          }
        });
      }
    },

    // 切换语言下拉菜单的可见状态变化
    handleLanguageDropdownVisibleChange(visible) {
      this.languageDropdownVisible = visible;
    },

    // 切换语言
    changeLanguage(lang) {
      changeLanguage(lang);
      this.languageDropdownVisible = false;
      this.$message.success({
        message: this.$t("message.success"),
        showClose: true,
      });
    },

    // 切换登录方式
    switchLoginType(type) {
      this.isMobileLogin = type === "mobile";
      // 清空表单
      this.form.username = "";
      this.form.mobile = "";
      this.form.password = "";
      this.form.captcha = "";
      this.fetchCaptcha();
    },

    // 封装输入验证逻辑
    validateInput(input, messageKey) {
      if (!input.trim()) {
        showDanger(this.$t(messageKey));
        return false;
      }
      return true;
    },

    async login() {
      if (this.isMobileLogin) {
        // 手机号登录验证
        if (!validateMobile(this.form.mobile, this.form.areaCode)) {
          showDanger(this.$t('login.requiredMobile'));
          return;
        }
        // 拼接手机号作为用户名
        this.form.username = this.form.areaCode + this.form.mobile;
      } else {
        // 用户名登录验证
        if (!this.validateInput(this.form.username, 'login.requiredUsername')) {
          return;
        }
      }

      // 验证密码
      if (!this.validateInput(this.form.password, 'login.requiredPassword')) {
        return;
      }
      // 验证验证码
      if (!this.validateInput(this.form.captcha, 'login.requiredCaptcha')) {
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
      if (!this.isSM2Encrypted(this.form.username)) {
        try {
          encryptedUsername = sm2Encrypt(this.serverPublicKey, this.form.username);
        } catch (error) {
          console.error("用户名加密失败:", error);
          showDanger(this.$t('sm2.encryptionFailed'));
          return;
        }
      }

      this.form.captchaId = this.captchaUuid;
      
      // 使用加密后的用户名和密码
      const loginData = {
        ...this.form,
        username: encryptedUsername,
        password: encryptedPassword
      };

      Api.user.login(
        loginData,
        ({ data }) => {
          showSuccess(this.$t('login.loginSuccess'));
          this.$store.commit("setToken", JSON.stringify(data.data));
          goToPage("/home");
        },
        (err) => {
          // 直接使用后端返回的国际化消息
          let errorMessage = err.data.msg || "登录失败";
          
          showDanger(errorMessage);
          if (
            err.data != null &&
            err.data.msg != null &&
            err.data.msg.indexOf("图形验证码") > -1 || err.data.msg.indexOf("Captcha") > -1
          ) {
            this.fetchCaptcha();
          }
        }
      );

      // 重新获取验证码
      setTimeout(() => {
        this.fetchCaptcha();
      }, 1000);
    },

    goToRegister() {
      goToPage("/register");
    },
    goToForgetPassword() {
      goToPage("/retrieve-password");
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
};
</script>
<style lang="scss" scoped>
@import "./auth.scss";

.login-type-container {
  margin: 10px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.title-language-dropdown {
  margin-left: auto;
}

.current-language-text {
  margin-left: 4px;
  margin-right: 4px;
  font-size: 12px;
  color: #3d4566;
}

.language-dropdown {
  margin-left: auto;
}

.rotate-down {
  transform: rotate(180deg);
  transition: transform 0.3s ease;
}

.el-icon-arrow-down {
  transition: transform 0.3s ease;
}

:deep(.el-button--primary) {
  background-color: #5778ff;
  border-color: #5778ff;

  &:hover,
  &:focus {
    background-color: #4a6ae8;
    border-color: #4a6ae8;
  }

  &:active {
    background-color: #3d5cd6;
    border-color: #3d5cd6;
  }
}
</style>
