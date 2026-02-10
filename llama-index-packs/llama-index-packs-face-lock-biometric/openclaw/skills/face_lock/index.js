const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const https = require('https');
const http = require('http');

// --- CONFIGURATION ---
const WORKSPACE_DIR = '/app/workspace/temp';
const SUBSCRIBERS_DB = '/app/config/subscribers.json';
const PYTHON_SCRIPT = path.join(__dirname, 'run_analysis.py');
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const EXEC_TIMEOUT = 30000; // 30 seconds

// --- HELPER: Download Image ---
const downloadImage = (url, filepath) => {
    return new Promise((resolve, reject) => {
        const protocol = url.startsWith('https') ? https : http;
        const file = fs.createWriteStream(filepath);
        let bytesWritten = 0;

        protocol.get(url, (response) => {
            // Follow redirects
            if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
                file.close();
                fs.unlink(filepath, () => {});
                return downloadImage(response.headers.location, filepath).then(resolve).catch(reject);
            }

            if (response.statusCode !== 200) {
                file.close();
                fs.unlink(filepath, () => {});
                return reject(new Error(`HTTP ${response.statusCode}`));
            }

            response.on('data', (chunk) => {
                bytesWritten += chunk.length;
                if (bytesWritten > MAX_FILE_SIZE) {
                    response.destroy();
                    file.close();
                    fs.unlink(filepath, () => {});
                    reject(new Error('File too large'));
                }
            });

            response.pipe(file);
            file.on('finish', () => file.close(resolve));
        }).on('error', (err) => {
            file.close();
            fs.unlink(filepath, () => {});
            reject(err);
        });
    });
};

// --- HELPER: Check premium status ---
const isPremiumUser = (userId) => {
    try {
        if (!fs.existsSync(SUBSCRIBERS_DB)) return false;
        const db = JSON.parse(fs.readFileSync(SUBSCRIBERS_DB, 'utf8'));
        return (db.premium_users || []).includes(userId.toString());
    } catch (e) {
        console.error('DB read error:', e.message);
        return false;
    }
};

// --- HELPER: Safe cleanup ---
const cleanup = (filepath) => {
    try {
        if (filepath && fs.existsSync(filepath)) {
            fs.unlinkSync(filepath);
        }
    } catch (e) {
        console.error('Cleanup error:', e.message);
    }
};

// --- SKILL DEFINITION ---
module.exports = {
    name: 'face_lock',
    description: 'Extracts biometric face data from an image for AI character consistency.',
    parameters: {
        type: 'object',
        properties: {
            image_url: { type: 'string', description: 'URL of the face image to analyze' },
            user_id: { type: 'string', description: 'Telegram/Discord user ID' },
            platform: { type: 'string', description: 'Target AI platform: reve, flux, midjourney, generic' }
        },
        required: ['image_url', 'user_id']
    },

    execute: async ({ image_url, user_id, platform = 'generic' }) => {
        // 1. SANITIZE INPUTS
        const safeUserId = user_id.toString().replace(/[^a-zA-Z0-9_-]/g, '');
        const safePlatform = ['reve', 'flux', 'midjourney', 'generic'].includes(platform) ? platform : 'generic';
        const filename = `scan_${safeUserId}_${Date.now()}.jpg`;

        // Ensure workspace exists
        if (!fs.existsSync(WORKSPACE_DIR)) {
            fs.mkdirSync(WORKSPACE_DIR, { recursive: true });
        }

        const localPath = path.join(WORKSPACE_DIR, filename);

        try {
            // 2. DOWNLOAD IMAGE
            await downloadImage(image_url, localPath);

            // Verify file exists and is not empty
            const stats = fs.statSync(localPath);
            if (stats.size === 0) {
                cleanup(localPath);
                return '\u26a0\ufe0f **Download Failed.** Empty file received. Try again.';
            }

            // 3. CHECK PREMIUM STATUS
            const premium = isPremiumUser(user_id);

            // 4. EXECUTE PYTHON ENGINE
            // Uses the REAL face_lock_biometric package, not a reimplementation
            const command = `python3 "${PYTHON_SCRIPT}" --image "${localPath}" --platform "${safePlatform}" --json-output`;

            return new Promise((resolve) => {
                exec(command, { timeout: EXEC_TIMEOUT }, (error, stdout, stderr) => {
                    // ALWAYS cleanup the temp image
                    cleanup(localPath);

                    if (error) {
                        if (error.killed) {
                            return resolve('\u26a0\ufe0f **Timeout.** Analysis took too long. Try a smaller image.');
                        }
                        console.error(`Python error: ${stderr}`);
                        return resolve('\u26a0\ufe0f **Scan Failed.** Could not detect a clear face. Send a frontal headshot.');
                    }

                    let data;
                    try {
                        data = JSON.parse(stdout.trim());
                    } catch (e) {
                        return resolve('\u26a0\ufe0f **System Error.** Could not parse biometric data.');
                    }

                    if (data.error) {
                        return resolve(`\u26a0\ufe0f **${data.error}**\nSend a clear portrait with the face fully visible.`);
                    }

                    // 5. FORMAT OUTPUT
                    const geometry =
                        `\u2022 **Gonial Angle:** ${data.gonial_angle}\u00b0\n` +
                        `\u2022 **Canthal Tilt:** ${data.canthal_tilt}\u00b0\n` +
                        `\u2022 **Facial Index:** ${data.facial_index}\n` +
                        `\u2022 **Eye Shape:** ${data.eye_shape}\n` +
                        `\u2022 **Fitzpatrick:** Type ${data.fitzpatrick_type}`;

                    if (!premium) {
                        // --- FREE TIER ---
                        resolve(
                            `\ud83d\udd12 **FACE LOCK: LITE**\n` +
                            `\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n` +
                            `**Geometry:**\n${geometry}\n\n` +
                            `\u26a0\ufe0f **LOCKED:**\n` +
                            `\u2022 Positive Prompt Codes\n` +
                            `\u2022 Anti-Drift Negative Prompts\n` +
                            `\u2022 Zygomatic Prominence\n` +
                            `\u2022 Lip Fullness Ratio\n` +
                            `\u2022 Brow Arch Height\n` +
                            `\u2022 Symmetry Score\n\n` +
                            `\ud83d\udc49 Upgrade to unlock full biometrics + prompt codes.`
                        );
                    } else {
                        // --- PREMIUM TIER ---
                        const fullMetrics =
                            `\u2022 **Nasal Rotation:** ${data.nasal_tip_rotation}\u00b0\n` +
                            `\u2022 **Zygomatic:** ${data.zygomatic_prominence}\n` +
                            `\u2022 **Lip Ratio:** ${data.lip_fullness_ratio}\n` +
                            `\u2022 **Brow Arch:** ${data.brow_arch_height}\n` +
                            `\u2022 **Symmetry:** ${data.face_symmetry_score}`;

                        resolve(
                            `\ud83d\udd13 **FACE LOCK: PRO**\n` +
                            `\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n` +
                            `**Geometry:**\n${geometry}\n\n` +
                            `**Extended Metrics:**\n${fullMetrics}\n\n` +
                            `**\ud83d\udccb POSITIVE PROMPT (${safePlatform}):**\n` +
                            `\`${data.positive_prompt}\`\n\n` +
                            `**\ud83d\udeab NEGATIVE PROMPT:**\n` +
                            `\`${data.negative_prompt}\``
                        );
                    }
                });
            });

        } catch (error) {
            cleanup(localPath);
            if (error.message === 'File too large') {
                return '\u26a0\ufe0f **File too large.** Max 10MB. Resize and resend.';
            }
            return '\u26a0\ufe0f **System Error.** Image download failed.';
        }
    }
};
