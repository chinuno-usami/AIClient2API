/**
 * Gemini Enterprise (Business AI Code) 提供商
 *
 * 复用 Antigravity CLI (agy) 的 Workforce Identity Federation 登录凭证：
 *   ~/.gemini/antigravity-cli/antigravity-oauth-token
 *
 * - 生成接口：businessaicode.googleapis.com/v1beta/projects/{project}/locations/{region}:streamGenerateContent
 * - 模型通过 aicode.experience 指定（不能同时设置 model 字段）
 * - 必须携带 entitlement.userTier
 * - 刷新：sts.googleapis.com/v1/oauthtoken，Basic auth = base64("{id_token.aud}:")
 */
import axios from 'axios';
import { promises as fs } from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as readline from 'readline';
import logger from '../../utils/logger.js';
import { atomicWriteFile } from '../../utils/file-lock.js';
import { configureTLSSidecar } from '../../utils/proxy-utils.js';
import {
    MODEL_PROVIDER,
    formatExpiryTime,
    formatExpiryLog,
    getRetryAfterMs,
    isRetryableNetworkError,
    normalizeProviderErrorMessage,
} from '../../utils/common.js';
import { getProviderModels } from '../provider-models.js';
import { getProviderPoolManager } from '../../services/service-manager.js';

// --- Constants ---
export const DEFAULT_ENTERPRISE_TOKEN_PATH = path.join('.gemini', 'antigravity-cli', 'antigravity-oauth-token');
const DEFAULT_BASE_URL = 'https://businessaicode.googleapis.com';
const API_VERSION = 'v1beta';
const STS_TOKEN_URL = 'https://sts.googleapis.com/v1/oauthtoken';
const DEFAULT_USER_AGENT = 'antigravity/cli/1.3.2 (aidev_client; os_type=darwin; arch=arm64; auth_method=gcp)';
const REQUEST_TIMEOUT_MS = 300000;
// 距离过期不足该时长即视为需要刷新
const REFRESH_SKEW_MS = 5 * 60 * 1000;
const NEAR_EXPIRY_MINUTES = 20;

const PROVIDER_LABEL = 'Gemini Enterprise';

function expandHome(p) {
    if (!p || typeof p !== 'string') return p;
    if (p === '~') return os.homedir();
    if (p.startsWith('~/')) return path.join(os.homedir(), p.slice(2));
    return p;
}

// businessaicode v1beta GenerateContentRequest 支持的字段（来自 proto 描述符）
const ALLOWED_REQUEST_FIELDS = [
    'contents', 'systemInstruction', 'tools', 'toolConfig',
    'labels', 'safetySettings', 'generationConfig',
];
const SNAKE_TO_CAMEL_FIELDS = {
    system_instruction: 'systemInstruction',
    tool_config: 'toolConfig',
    safety_settings: 'safetySettings',
    generation_config: 'generationConfig',
};
// businessaicode v1beta HarmCategory 枚举仅支持以下类别（如 CIVIC_INTEGRITY 会被拒绝）
const SUPPORTED_HARM_CATEGORIES = new Set([
    'HARM_CATEGORY_HARASSMENT',
    'HARM_CATEGORY_HATE_SPEECH',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT',
    'HARM_CATEGORY_DANGEROUS_CONTENT',
]);

/**
 * 解析 agy 凭证文件内容
 * @param {object} raw - antigravity-oauth-token 的 JSON 内容
 * @returns {{accessToken: string, refreshToken: string, expiryMs: number|null, projectId: string, region: string, userTier: string, clientId: string|null}}
 */
export function parseEnterpriseToken(raw) {
    if (!raw || typeof raw !== 'object' || !raw.token) {
        throw new Error('Invalid Gemini Enterprise token file: missing "token" object');
    }
    const expiryMs = raw.token.expiry ? Date.parse(raw.token.expiry) : NaN;
    return {
        accessToken: raw.token.access_token || null,
        refreshToken: raw.token.refresh_token || null,
        expiryMs: Number.isFinite(expiryMs) ? expiryMs : null,
        projectId: raw.project_id || null,
        region: raw.region || 'global',
        userTier: raw.user_tier || null,
        clientId: extractClientIdFromIdToken(raw.id_token),
    };
}

/**
 * 从 id_token 的 aud 中提取 Workforce OAuth client id（STS 刷新所需）
 */
export function extractClientIdFromIdToken(idToken) {
    if (!idToken || typeof idToken !== 'string') return null;
    const parts = idToken.split('.');
    if (parts.length < 2) return null;
    try {
        const payload = JSON.parse(Buffer.from(parts[1], 'base64url').toString('utf8'));
        return Array.isArray(payload.aud) ? payload.aud[0] : (payload.aud || null);
    } catch {
        return null;
    }
}

/**
 * 将标准 Gemini 请求体转换为 businessaicode 请求体
 * @param {string} model - 模型 ID（作为 aicode.experience）
 * @param {object} requestBody - Gemini generateContent 请求体
 * @param {string} userTier - entitlement.userTier
 */
export function buildEnterpriseRequest(model, requestBody, userTier) {
    const source = { ...(requestBody || {}) };
    for (const [snake, camel] of Object.entries(SNAKE_TO_CAMEL_FIELDS)) {
        if (source[snake] !== undefined && source[camel] === undefined) {
            source[camel] = source[snake];
        }
    }

    const request = {};
    for (const field of ALLOWED_REQUEST_FIELDS) {
        if (source[field] !== undefined) request[field] = source[field];
    }

    if (Array.isArray(request.contents)) {
        request.contents = request.contents.map(c => (c && !c.role ? { ...c, role: 'user' } : c));
    }
    if (request.systemInstruction && !request.systemInstruction.role) {
        request.systemInstruction = { ...request.systemInstruction, role: 'user' };
    }
    if (Array.isArray(request.safetySettings)) {
        request.safetySettings = request.safetySettings.filter(s => SUPPORTED_HARM_CATEGORIES.has(s?.category));
        if (request.safetySettings.length === 0) delete request.safetySettings;
    }

    // model 与 aicode 属于同一个 oneof，只能使用 aicode.experience
    request.aicode = { experience: model };
    request.entitlement = { userTier };
    return request;
}

/**
 * 只保留标准 Gemini 响应字段
 */
function toGeminiApiResponse(chunk) {
    if (!chunk) return null;
    const out = { candidates: chunk.candidates };
    if (chunk.usageMetadata) out.usageMetadata = chunk.usageMetadata;
    if (chunk.promptFeedback) out.promptFeedback = chunk.promptFeedback;
    if (chunk.modelVersion) out.modelVersion = chunk.modelVersion;
    if (chunk.responseId) out.responseId = chunk.responseId;
    return out;
}

/**
 * 合并流式响应块为单个非流式响应
 * @param {object[]} chunks
 */
export function mergeStreamChunks(chunks) {
    const parts = [];
    let last = null;
    let role = 'model';
    for (const chunk of chunks) {
        const cand = chunk?.candidates?.[0];
        if (!cand) { if (chunk) last = { ...last, ...chunk }; continue; }
        if (cand.content?.role) role = cand.content.role;
        for (const part of cand.content?.parts || []) {
            const prev = parts[parts.length - 1];
            // 相邻的普通文本（或相邻的思考文本）合并为一个 part
            const mergeable = prev && typeof prev.text === 'string' && typeof part.text === 'string' &&
                !!prev.thought === !!part.thought && !prev.functionCall && !part.functionCall;
            if (mergeable) {
                prev.text += part.text;
                if (part.thoughtSignature) prev.thoughtSignature = part.thoughtSignature;
            } else if (part.thoughtSignature && part.text === '' && prev) {
                // 末尾空文本块仅携带签名，附加到上一个 part
                prev.thoughtSignature = part.thoughtSignature;
            } else {
                parts.push({ ...part });
            }
        }
        last = chunk;
    }
    const lastCand = last?.candidates?.[0] || {};
    const merged = {
        candidates: [{ ...lastCand, content: { role, parts }, index: lastCand.index ?? 0 }],
    };
    if (last?.usageMetadata) merged.usageMetadata = last.usageMetadata;
    if (last?.modelVersion) merged.modelVersion = last.modelVersion;
    if (last?.responseId) merged.responseId = last.responseId;
    return merged;
}

export class GeminiEnterpriseApiService {
    constructor(config) {
        this.config = config;
        this.uuid = config.uuid;
        this.providerType = config.MODEL_PROVIDER || MODEL_PROVIDER.GEMINI_ENTERPRISE;
        this.tokenFilePath = expandHome(config.GEMINI_ENTERPRISE_TOKEN_FILE_PATH) ||
            path.join(os.homedir(), DEFAULT_ENTERPRISE_TOKEN_PATH);
        this.baseUrl = (config.GEMINI_ENTERPRISE_BASE_URL || DEFAULT_BASE_URL).replace(/\/$/, '');
        this.userAgent = config.GEMINI_ENTERPRISE_USER_AGENT || DEFAULT_USER_AGENT;
        // 可选覆盖：未配置时从凭证文件读取
        this.projectIdOverride = config.PROJECT_ID || null;
        this.userTierOverride = config.GEMINI_ENTERPRISE_USER_TIER || null;

        this.rawToken = null;
        this.creds = null;
        this.fileMtimeMs = 0;
        this.refreshPromise = null;
        this.availableModels = getProviderModels(MODEL_PROVIDER.GEMINI_ENTERPRISE);
        this.isInitialized = false;
    }

    get projectId() { return this.projectIdOverride || this.creds?.projectId || null; }
    get region() { return this.creds?.region || 'global'; }
    get userTier() { return this.userTierOverride || this.creds?.userTier || null; }

    _applySidecar(axiosConfig) {
        return configureTLSSidecar(axiosConfig, this.config, this.providerType, this.baseUrl);
    }

    async initialize() {
        if (this.isInitialized) return;
        logger.info(`[${PROVIDER_LABEL}] Initializing, token file: ${this.tokenFilePath}`);
        await this.loadCredentials(true);
        if (!this.projectId) {
            throw new Error(`[${PROVIDER_LABEL}] project_id missing in token file; set PROJECT_ID`);
        }
        if (!this.userTier) {
            throw new Error(`[${PROVIDER_LABEL}] user_tier missing in token file; set GEMINI_ENTERPRISE_USER_TIER`);
        }
        this.isInitialized = true;
        logger.info(`[${PROVIDER_LABEL}] Initialized. project=${this.projectId} region=${this.region} tier=${this.userTier}`);
    }

    /**
     * 从文件加载凭证。文件未变化时跳过（agy 可能在外部刷新并写入该文件）
     * @param {boolean} force - 忽略 mtime 缓存强制加载
     */
    async loadCredentials(force = false) {
        let stat;
        try {
            stat = await fs.stat(this.tokenFilePath);
        } catch (error) {
            if (error.code === 'ENOENT') {
                throw new Error(`[${PROVIDER_LABEL}] Token file not found: ${this.tokenFilePath}. Run "agy" and log in first.`);
            }
            throw error;
        }
        if (!force && this.creds && stat.mtimeMs === this.fileMtimeMs) return;

        const raw = JSON.parse(await fs.readFile(this.tokenFilePath, 'utf8'));
        this.rawToken = raw;
        this.creds = parseEnterpriseToken(raw);
        this.fileMtimeMs = stat.mtimeMs;
        logger.debug(`[${PROVIDER_LABEL}] Credentials loaded. ${formatExpiryTime(this.creds.expiryMs)}`);
    }

    _isTokenValid(skewMs = REFRESH_SKEW_MS) {
        return !!(this.creds?.accessToken && this.creds.expiryMs && this.creds.expiryMs - Date.now() > skewMs);
    }

    /**
     * 获取可用 access token，必要时刷新
     */
    async getAccessToken() {
        await this.loadCredentials();
        if (this._isTokenValid()) return this.creds.accessToken;
        await this.refreshAccessToken();
        return this.creds.accessToken;
    }

    /**
     * 通过 STS 刷新 access token 并写回 agy 凭证文件（并发调用共享同一个刷新）
     * @param {boolean} force - 即使 token 仍有效也刷新
     */
    async refreshAccessToken(force = false) {
        if (this.refreshPromise) return this.refreshPromise;
        this.refreshPromise = this._doRefresh(force).finally(() => { this.refreshPromise = null; });
        return this.refreshPromise;
    }

    async _doRefresh(force) {
        // agy 可能已在外部刷新过，先重新读取文件
        await this.loadCredentials(true);
        if (!force && this._isTokenValid()) {
            logger.info(`[${PROVIDER_LABEL}] Token already refreshed externally, skip.`);
            return false;
        }
        const { refreshToken, clientId } = this.creds;
        if (!refreshToken || !clientId) {
            throw new Error(`[${PROVIDER_LABEL}] Cannot refresh: missing refresh_token or id_token. Re-login with "agy".`);
        }

        logger.info(`[${PROVIDER_LABEL}] Refreshing access token via STS...`);
        const axiosConfig = {
            method: 'post',
            url: STS_TOKEN_URL,
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                Authorization: 'Basic ' + Buffer.from(`${clientId}:`).toString('base64'),
            },
            data: new URLSearchParams({ grant_type: 'refresh_token', refresh_token: refreshToken }).toString(),
            timeout: 30000,
        };
        this._applySidecar(axiosConfig);

        let data;
        try {
            ({ data } = await axios.request(axiosConfig));
        } catch (error) {
            const detail = error.response?.data?.error_description || error.response?.data?.error || error.message;
            throw new Error(`[${PROVIDER_LABEL}] STS refresh failed (${error.response?.status || error.code}): ${detail}. Re-login with "agy" if this persists.`);
        }
        if (!data?.access_token) {
            throw new Error(`[${PROVIDER_LABEL}] STS refresh returned no access_token`);
        }

        const expiresInSec = Number(data.expires_in) || 3600;
        const updated = {
            ...this.rawToken,
            token: {
                ...this.rawToken.token,
                access_token: data.access_token,
                token_type: data.token_type || this.rawToken.token.token_type || 'Bearer',
                refresh_token: data.refresh_token || refreshToken,
                expiry: new Date(Date.now() + expiresInSec * 1000).toISOString(),
            },
        };
        await atomicWriteFile(this.tokenFilePath, JSON.stringify(updated), { mode: 0o600 });
        await this.loadCredentials(true);
        logger.info(`[${PROVIDER_LABEL}] Token refreshed and saved. ${formatExpiryTime(this.creds.expiryMs)}`);
        return true;
    }

    isExpiryDateNear() {
        try {
            const { message, isNearExpiry } = formatExpiryLog(PROVIDER_LABEL, this.creds?.expiryMs, NEAR_EXPIRY_MINUTES);
            logger.info(message);
            return isNearExpiry;
        } catch (error) {
            logger.error(`[${PROVIDER_LABEL}] Error checking expiry date: ${error.message}`);
            return false;
        }
    }

    _buildUrl(action) {
        const base = `${this.baseUrl}/${API_VERSION}/projects/${this.projectId}/locations/${this.region}:${action}`;
        return action === 'streamGenerateContent' ? `${base}?alt=sse` : base;
    }

    _resolveModel(model) {
        if (this.availableModels.includes(model)) return model;
        if (this.config.MODEL_FALLBACK_ENABLED === false) {
            throw new Error(`[${PROVIDER_LABEL}] 模型不存在: ${model}`);
        }
        logger.warn(`[${PROVIDER_LABEL}] Model '${model}' not in list, using default '${this.availableModels[0]}'`);
        return this.availableModels[0];
    }

    _stripInternalFields(requestBody) {
        if (requestBody?._monitorRequestId) {
            this.config._monitorRequestId = requestBody._monitorRequestId;
            delete requestBody._monitorRequestId;
        }
        if (requestBody?._requestBaseUrl) delete requestBody._requestBaseUrl;
    }

    _markNeedRefreshIfNear() {
        if (!this.isExpiryDateNear()) return;
        const poolManager = getProviderPoolManager();
        if (poolManager && this.uuid) {
            poolManager.markProviderNeedRefresh(this.providerType, { uuid: this.uuid });
        }
    }

    /**
     * 发起 SSE 请求并逐块产出 JSON；包含 401 刷新重试、429/5xx/网络错误退避重试
     */
    async * streamApi(body, retryCount = 0, isAuthRetry = false) {
        const maxRetries = this.config.REQUEST_MAX_RETRIES || 3;
        const baseDelay = this.config.REQUEST_BASE_DELAY || 1000;
        const accessToken = await this.getAccessToken();

        const axiosConfig = {
            method: 'post',
            url: this._buildUrl('streamGenerateContent'),
            headers: {
                Authorization: `Bearer ${accessToken}`,
                'Content-Type': 'application/json',
                'User-Agent': this.userAgent,
                Accept: 'text/event-stream',
            },
            data: JSON.stringify(body),
            responseType: 'stream',
            timeout: REQUEST_TIMEOUT_MS,
        };
        this._applySidecar(axiosConfig);

        let response;
        try {
            response = await axios.request(axiosConfig);
        } catch (error) {
            const status = error.response?.status;
            if (status) await normalizeProviderErrorMessage(error, { status, context: 'stream' });
            logger.error(`[${PROVIDER_LABEL}] Stream error (Status: ${status}, Code: ${error.code}): ${error.message}`);

            if (status === 401 && !isAuthRetry) {
                logger.info(`[${PROVIDER_LABEL}] 401 received, forcing token refresh and retrying once...`);
                try {
                    await this.refreshAccessToken(true);
                } catch (refreshError) {
                    logger.error(refreshError.message);
                    const poolManager = getProviderPoolManager();
                    if (poolManager && this.uuid) {
                        poolManager.markProviderNeedRefresh(this.providerType, { uuid: this.uuid });
                        error.credentialMarkedUnhealthy = true;
                    }
                    error.shouldSwitchCredential = true;
                    error.skipErrorCount = true;
                    throw error;
                }
                yield* this.streamApi(body, retryCount, true);
                return;
            }

            const retryable = status === 429 || (status >= 500 && status < 600) || (!status && isRetryableNetworkError(error));
            if (status === 429 && getRetryAfterMs(error) !== null) throw error;
            if (retryable && retryCount < maxRetries) {
                const delay = baseDelay * Math.pow(2, retryCount);
                logger.info(`[${PROVIDER_LABEL}] Retrying in ${delay}ms (attempt ${retryCount + 1}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
                yield* this.streamApi(body, retryCount + 1, isAuthRetry);
                return;
            }
            throw error;
        }

        yield* this.parseSSEStream(response.data);
    }

    async * parseSSEStream(stream) {
        const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
        let buffer = [];
        const flush = () => {
            if (buffer.length === 0) return null;
            const text = buffer.join('\n');
            buffer = [];
            try { return JSON.parse(text); } catch {
                logger.error(`[${PROVIDER_LABEL}] Failed to parse SSE chunk: ${text.slice(0, 200)}`);
                return null;
            }
        };
        for await (const line of rl) {
            if (line.startsWith('data: ')) buffer.push(line.slice(6));
            else if (line === '') {
                const obj = flush();
                if (obj) yield obj;
            }
        }
        const obj = flush();
        if (obj) yield obj;
    }

    async * generateContentStream(model, requestBody) {
        if (!this.isInitialized) await this.initialize();
        this._stripInternalFields(requestBody);
        this._markNeedRefreshIfNear();

        const baseModel = this._resolveModel(model);
        if (requestBody && baseModel !== model) requestBody.model = baseModel;
        const body = buildEnterpriseRequest(baseModel, requestBody, this.userTier);

        for await (const chunk of this.streamApi(body)) {
            const response = toGeminiApiResponse(chunk);
            if (response) yield response;
        }
    }

    /**
     * 非流式：后端同样走 SSE，合并后返回（agy 仅使用流式接口）
     */
    async generateContent(model, requestBody) {
        const chunks = [];
        for await (const chunk of this.generateContentStream(model, requestBody)) {
            chunks.push(chunk);
        }
        return mergeStreamChunks(chunks);
    }

    async listModels() {
        if (!this.isInitialized) await this.initialize();
        return {
            models: this.availableModels.map(modelId => ({
                name: `models/${modelId}`,
                version: '1.0.0',
                displayName: modelId,
                description: `Gemini Enterprise model ${modelId}`,
                inputTokenLimit: 1048576,
                outputTokenLimit: 65535,
                supportedGenerationMethods: ['generateContent', 'streamGenerateContent'],
            })),
        };
    }
}
