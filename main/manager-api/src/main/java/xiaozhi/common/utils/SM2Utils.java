package xiaozhi.common.utils;

import org.bouncycastle.asn1.x9.X9ECParameters;
import org.bouncycastle.crypto.engines.SM2Engine;
import org.bouncycastle.crypto.params.ECDomainParameters;
import org.bouncycastle.crypto.params.ECPrivateKeyParameters;
import org.bouncycastle.crypto.params.ECPublicKeyParameters;
import org.bouncycastle.crypto.params.ParametersWithRandom;
import org.bouncycastle.jcajce.provider.asymmetric.ec.BCECPrivateKey;
import org.bouncycastle.jcajce.provider.asymmetric.ec.BCECPublicKey;
import org.bouncycastle.jcajce.provider.asymmetric.util.ECUtil;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.bouncycastle.math.ec.ECPoint;
import org.bouncycastle.util.encoders.Base64;

import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.security.spec.ECGenParameterSpec;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.X509EncodedKeySpec;

/**
 * SM2加密工具类
 */
public class SM2Utils {

    static {
        Security.addProvider(new BouncyCastleProvider());
    }

    /**
     * 生成SM2密钥对
     *
     * @return 密钥对
     */
    public static KeyPair generateKeyPair() {
        try {
            KeyPairGenerator keyPairGenerator = KeyPairGenerator.getInstance("EC", "BC");
            ECGenParameterSpec sm2Spec = new ECGenParameterSpec("sm2p256v1");
            keyPairGenerator.initialize(sm2Spec, new SecureRandom());
            return keyPairGenerator.generateKeyPair();
        } catch (Exception e) {
            throw new RuntimeException("生成SM2密钥对失败", e);
        }
    }

    /**
     * 获取公钥字符串
     *
     * @param publicKey 公钥
     * @return Base64编码的公钥字符串
     */
    public static String getPublicKeyStr(PublicKey publicKey) {
        BCECPublicKey bcPublicKey = (BCECPublicKey) publicKey;
        return Base64.toBase64String(bcPublicKey.getEncoded());
    }

    /**
     * 获取私钥字符串
     *
     * @param privateKey 私钥
     * @return Base64编码的私钥字符串
     */
    public static String getPrivateKeyStr(PrivateKey privateKey) {
        BCECPrivateKey bcPrivateKey = (BCECPrivateKey) privateKey;
        return Base64.toBase64String(bcPrivateKey.getEncoded());
    }

    /**
     * 从字符串加载公钥
     *
     * @param publicKeyStr Base64编码的公钥字符串
     * @return 公钥对象
     */
    public static PublicKey loadPublicKey(String publicKeyStr) {
        try {
            byte[] keyBytes = Base64.decode(publicKeyStr);
            X509EncodedKeySpec keySpec = new X509EncodedKeySpec(keyBytes);
            KeyFactory keyFactory = KeyFactory.getInstance("EC", "BC");
            return keyFactory.generatePublic(keySpec);
        } catch (Exception e) {
            throw new RuntimeException("加载公钥失败", e);
        }
    }

    /**
     * 从字符串加载私钥
     *
     * @param privateKeyStr Base64编码的私钥字符串
     * @return 私钥对象
     */
    public static PrivateKey loadPrivateKey(String privateKeyStr) {
        try {
            byte[] keyBytes = Base64.decode(privateKeyStr);
            PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(keyBytes);
            KeyFactory keyFactory = KeyFactory.getInstance("EC", "BC");
            return keyFactory.generatePrivate(keySpec);
        } catch (Exception e) {
            throw new RuntimeException("加载私钥失败", e);
        }
    }

    /**
     * SM2加密
     *
     * @param publicKey  公钥
     * @param plainText  明文
     * @return Base64编码的密文
     */
    public static String encrypt(PublicKey publicKey, String plainText) {
        try {
            BCECPublicKey bcPublicKey = (BCECPublicKey) publicKey;
            ECPoint ecPoint = bcPublicKey.getQ();
            X9ECParameters x9ECParameters = ECUtil.getNamedCurveByName("sm2p256v1");
            ECDomainParameters ecDomainParameters = new ECDomainParameters(
                    x9ECParameters.getCurve(), x9ECParameters.getG(), x9ECParameters.getN());
            ECPublicKeyParameters publicKeyParameters = new ECPublicKeyParameters(ecPoint, ecDomainParameters);

            SM2Engine sm2Engine = new SM2Engine();
            sm2Engine.init(true, new ParametersWithRandom(publicKeyParameters, new SecureRandom()));

            byte[] input = plainText.getBytes(StandardCharsets.UTF_8);
            byte[] encrypted = sm2Engine.processBlock(input, 0, input.length);
            return Base64.toBase64String(encrypted);
        } catch (Exception e) {
            throw new RuntimeException("SM2加密失败", e);
        }
    }

    /**
     * SM2解密
     *
     * @param privateKey 私钥
     * @param cipherText Base64编码的密文
     * @return 明文
     */
    public static String decrypt(PrivateKey privateKey, String cipherText) {
        try {
            BCECPrivateKey bcPrivateKey = (BCECPrivateKey) privateKey;
            BigInteger privateKeyValue = bcPrivateKey.getD();
            X9ECParameters x9ECParameters = ECUtil.getNamedCurveByName("sm2p256v1");
            ECDomainParameters ecDomainParameters = new ECDomainParameters(
                    x9ECParameters.getCurve(), x9ECParameters.getG(), x9ECParameters.getN());
            ECPrivateKeyParameters privateKeyParameters = new ECPrivateKeyParameters(privateKeyValue, ecDomainParameters);

            SM2Engine sm2Engine = new SM2Engine();
            sm2Engine.init(false, privateKeyParameters);

            byte[] encrypted = Base64.decode(cipherText);
            byte[] decrypted = sm2Engine.processBlock(encrypted, 0, encrypted.length);
            return new String(decrypted, StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new RuntimeException("SM2解密失败", e);
        }
    }

    /**
     * 验证SM2密钥对是否匹配
     *
     * @param publicKeyStr  公钥字符串
     * @param privateKeyStr 私钥字符串
     * @return 是否匹配
     */
    public static boolean verifyKeyPair(String publicKeyStr, String privateKeyStr) {
        try {
            String testData = "test_sm2_encryption";
            PublicKey publicKey = loadPublicKey(publicKeyStr);
            PrivateKey privateKey = loadPrivateKey(privateKeyStr);

            String encrypted = encrypt(publicKey, testData);
            String decrypted = decrypt(privateKey, encrypted);

            return testData.equals(decrypted);
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 生成SM2密钥对并返回Base64编码的字符串
     *
     * @return 包含公钥和私钥的字符串数组 [公钥, 私钥]
     */
    public static String[] generateKeyPairStrings() {
        KeyPair keyPair = generateKeyPair();
        String publicKeyStr = getPublicKeyStr(keyPair.getPublic());
        String privateKeyStr = getPrivateKeyStr(keyPair.getPrivate());
        return new String[]{publicKeyStr, privateKeyStr};
    }
}