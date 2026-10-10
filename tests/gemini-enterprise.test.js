jest.mock('open', () => ({
    __esModule: true,
    default: jest.fn()
}));
jest.mock('axios', () => ({
    __esModule: true,
    default: { request: jest.fn() }
}));
jest.mock('../src/utils/proxy-utils.js', () => ({
    configureTLSSidecar: jest.fn(cfg => cfg),
    getProxyConfigForProvider: jest.fn(),
    getGoogleAuthProxyConfig: jest.fn(),
    isTLSSidecarEnabledForProvider: jest.fn(() => false)
}));
jest.mock('../src/services/service-manager.js', () => ({
    getProviderPoolManager: jest.fn(() => null)
}));

import axios from 'axios';
import { Readable } from 'stream';
import { promises as fs } from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
    GeminiEnterpriseApiService,
    buildEnterpriseRequest,
    extractClientIdFromIdToken,
    mergeStreamChunks,
    parseEnterpriseToken
} from '../src/providers/gemini/gemini-enterprise-core.js';
import { getProtocolPrefix, MODEL_PROVIDER } from '../src/utils/common.js';
import { isRegisteredProvider } from '../src/providers/adapter.js';

// 构造测试用 id_token（只关心 payload.aud）
function fakeIdToken(aud) {
    const enc = obj => Buffer.from(JSON.stringify(obj)).toString('base64url');
    return `${enc({ alg: 'none' })}.${enc({ aud, iss: 'https://auth.cloud.google' })}.sig`;
}

function fakeTokenFile({ expiry, accessToken = 'at-old', refreshToken = 'rt-1' } = {}) {
    return {
        token: { access_token: accessToken, token_type: 'Bearer', refresh_token: refreshToken, expiry },
        auth_method: 'gcp',
        id_token: fakeIdToken('client-abc'),
        wif_provider: 'locations/global/workforcePools/p/providers/p',
        project_id: 'proj-1',
        region: 'global',
        user_tier: 'gcp-ge-plus-tier'
    };
}

function sseStream(chunks) {
    return Readable.from(chunks.map(c => `data: ${JSON.stringify(c)}\n\n`));
}

describe('Gemini Enterprise helpers', () => {
    test('registers provider and maps to gemini protocol', () => {
        expect(MODEL_PROVIDER.GEMINI_ENTERPRISE).toBe('gemini-enterprise');
        expect(isRegisteredProvider('gemini-enterprise')).toBe(true);
        expect(getProtocolPrefix('gemini-enterprise')).toBe('gemini');
    });

    test('parses agy token file', () => {
        const parsed = parseEnterpriseToken(fakeTokenFile({ expiry: '2030-01-01T00:00:00+08:00' }));
        expect(parsed).toMatchObject({
            accessToken: 'at-old', refreshToken: 'rt-1', projectId: 'proj-1',
            region: 'global', userTier: 'gcp-ge-plus-tier', clientId: 'client-abc'
        });
        expect(parsed.expiryMs).toBe(Date.parse('2030-01-01T00:00:00+08:00'));
        expect(() => parseEnterpriseToken({})).toThrow();
        expect(extractClientIdFromIdToken('bad')).toBeNull();
    });

    test('builds request with aicode.experience and entitlement, drops unknown fields', () => {
        const req = buildEnterpriseRequest('gemini-3.8-flash-low', {
            model: 'x',
            contents: [{ parts: [{ text: 'hi' }] }],
            system_instruction: { parts: [{ text: 'sys' }] },
            generation_config: { temperature: 0.2 },
            foo: 'bar'
        }, 'gcp-ge-plus-tier');

        expect(req).toEqual({
            contents: [{ role: 'user', parts: [{ text: 'hi' }] }],
            systemInstruction: { role: 'user', parts: [{ text: 'sys' }] },
            generationConfig: { temperature: 0.2 },
            aicode: { experience: 'gemini-3.8-flash-low' },
            entitlement: { userTier: 'gcp-ge-plus-tier' }
        });
        expect(req.model).toBeUndefined();
    });

    test('filters safety categories unsupported by businessaicode', () => {
        const req = buildEnterpriseRequest('m', {
            contents: [],
            safetySettings: [
                { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
                { category: 'HARM_CATEGORY_CIVIC_INTEGRITY', threshold: 'BLOCK_NONE' }
            ]
        }, 't');
        expect(req.safetySettings).toEqual([{ category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' }]);

        const onlyUnsupported = buildEnterpriseRequest('m', {
            contents: [], safetySettings: [{ category: 'HARM_CATEGORY_CIVIC_INTEGRITY', threshold: 'BLOCK_NONE' }]
        }, 't');
        expect(onlyUnsupported.safetySettings).toBeUndefined();
    });

    test('merges stream chunks into one response', () => {
        const merged = mergeStreamChunks([
            { candidates: [{ content: { role: 'model', parts: [{ text: 'Hel' }] } }], modelVersion: 'm' },
            { candidates: [{ content: { role: 'model', parts: [{ text: 'lo' }] } }] },
            {
                candidates: [{ content: { role: 'model', parts: [{ text: '', thoughtSignature: 'sig' }] }, finishReason: 'STOP' }],
                usageMetadata: { totalTokenCount: 3 }, modelVersion: 'm'
            }
        ]);
        expect(merged.candidates[0].content.parts).toEqual([{ text: 'Hello', thoughtSignature: 'sig' }]);
        expect(merged.candidates[0].finishReason).toBe('STOP');
        expect(merged.usageMetadata).toEqual({ totalTokenCount: 3 });
    });
});

describe('GeminiEnterpriseApiService', () => {
    let tmpDir;
    let tokenPath;

    beforeEach(async () => {
        axios.request.mockReset();
        tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ge-test-'));
        tokenPath = path.join(tmpDir, 'antigravity-oauth-token');
    });

    afterEach(async () => {
        await fs.rm(tmpDir, { recursive: true, force: true });
    });

    const newService = () => new GeminiEnterpriseApiService({
        MODEL_PROVIDER: 'gemini-enterprise',
        GEMINI_ENTERPRISE_TOKEN_FILE_PATH: tokenPath,
        REQUEST_BASE_DELAY: 1
    });

    test('streams with valid token and sends expected request', async () => {
        const expiry = new Date(Date.now() + 3600e3).toISOString();
        await fs.writeFile(tokenPath, JSON.stringify(fakeTokenFile({ expiry })));
        axios.request.mockResolvedValueOnce({
            status: 200,
            data: sseStream([{ candidates: [{ content: { role: 'model', parts: [{ text: 'ok' }] } }], responseId: 'r1', extra: 1 }])
        });

        const service = newService();
        const out = [];
        for await (const c of service.generateContentStream('gemini-3.8-flash-low', { contents: [{ role: 'user', parts: [{ text: 'hi' }] }] })) {
            out.push(c);
        }

        expect(out).toEqual([{ candidates: [{ content: { role: 'model', parts: [{ text: 'ok' }] } }], responseId: 'r1' }]);
        const call = axios.request.mock.calls[0][0];
        expect(call.url).toBe('https://businessaicode.googleapis.com/v1beta/projects/proj-1/locations/global:streamGenerateContent?alt=sse');
        expect(call.headers.Authorization).toBe('Bearer at-old');
        expect(JSON.parse(call.data)).toMatchObject({
            aicode: { experience: 'gemini-3.8-flash-low' },
            entitlement: { userTier: 'gcp-ge-plus-tier' }
        });
    });

    test('refreshes expired token via STS and writes back preserving fields', async () => {
        await fs.writeFile(tokenPath, JSON.stringify(fakeTokenFile({ expiry: '2020-01-01T00:00:00Z' })));
        axios.request
            .mockResolvedValueOnce({ data: { access_token: 'at-new', expires_in: 3599, token_type: 'Bearer' } })
            .mockResolvedValueOnce({ status: 200, data: sseStream([{ candidates: [{ content: { parts: [{ text: 'ok' }] } }] }]) });

        const service = newService();
        const res = await service.generateContent('gemini-3.8-flash-low', { contents: [{ parts: [{ text: 'hi' }] }] });

        const stsCall = axios.request.mock.calls[0][0];
        expect(stsCall.url).toBe('https://sts.googleapis.com/v1/oauthtoken');
        expect(stsCall.headers.Authorization).toBe('Basic ' + Buffer.from('client-abc:').toString('base64'));
        expect(stsCall.data).toContain('grant_type=refresh_token');
        expect(axios.request.mock.calls[1][0].headers.Authorization).toBe('Bearer at-new');
        expect(res.candidates[0].content.parts[0].text).toBe('ok');

        const saved = JSON.parse(await fs.readFile(tokenPath, 'utf8'));
        expect(saved.token.access_token).toBe('at-new');
        expect(saved.token.refresh_token).toBe('rt-1');
        expect(Date.parse(saved.token.expiry)).toBeGreaterThan(Date.now());
        expect(saved.wif_provider).toBe('locations/global/workforcePools/p/providers/p');
        expect(saved.user_tier).toBe('gcp-ge-plus-tier');
    });

    test('skips STS when token file was refreshed externally', async () => {
        const expiry = new Date(Date.now() + 3600e3).toISOString();
        await fs.writeFile(tokenPath, JSON.stringify(fakeTokenFile({ expiry, accessToken: 'at-agy' })));
        const service = newService();
        await service.initialize();
        expect(await service.refreshAccessToken()).toBe(false);
        expect(axios.request).not.toHaveBeenCalled();
    });

    test('retries once after 401 with forced refresh', async () => {
        const expiry = new Date(Date.now() + 3600e3).toISOString();
        await fs.writeFile(tokenPath, JSON.stringify(fakeTokenFile({ expiry })));
        const unauthorized = Object.assign(new Error('401'), { response: { status: 401, data: 'unauthorized', headers: {} } });
        axios.request
            .mockRejectedValueOnce(unauthorized)
            .mockResolvedValueOnce({ data: { access_token: 'at-new', expires_in: 3599 } })
            .mockResolvedValueOnce({ status: 200, data: sseStream([{ candidates: [{ content: { parts: [{ text: 'ok' }] } }] }]) });

        const service = newService();
        const res = await service.generateContent('gemini-3.8-flash-low', { contents: [{ parts: [{ text: 'hi' }] }] });
        expect(res.candidates[0].content.parts[0].text).toBe('ok');
        expect(axios.request.mock.calls[2][0].headers.Authorization).toBe('Bearer at-new');
    });

    test('rejects unknown model when fallback disabled', async () => {
        const expiry = new Date(Date.now() + 3600e3).toISOString();
        await fs.writeFile(tokenPath, JSON.stringify(fakeTokenFile({ expiry })));
        const service = new GeminiEnterpriseApiService({
            MODEL_PROVIDER: 'gemini-enterprise',
            GEMINI_ENTERPRISE_TOKEN_FILE_PATH: tokenPath,
            MODEL_FALLBACK_ENABLED: false
        });
        await expect(service.generateContent('nope', { contents: [] })).rejects.toThrow('模型不存在');
    });
});
