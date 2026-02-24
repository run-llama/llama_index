const { Telegraf } = require('telegraf');
const http = require('http');
const faceLockSkill = require('./skills/face_lock/index');

// --- CONFIGURATION ---
const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const PORT = process.env.PORT || 3000;

if (!BOT_TOKEN) {
    console.error('FATAL: TELEGRAM_BOT_TOKEN environment variable is not set.');
    console.error('Get a token from @BotFather on Telegram, then:');
    console.error('  export TELEGRAM_BOT_TOKEN="your-token-here"');
    process.exit(1);
}

const bot = new Telegraf(BOT_TOKEN);

// --- HEALTH CHECK SERVER ---
// docker-compose healthcheck hits this endpoint
const healthServer = http.createServer((req, res) => {
    if (req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok', uptime: process.uptime() }));
    } else {
        res.writeHead(404);
        res.end();
    }
});

healthServer.listen(PORT, () => {
    console.log(`Health check listening on port ${PORT}`);
});

// --- BOT HANDLERS ---

// /start command
bot.start((ctx) => {
    ctx.reply(
        '🔒 **FACE LOCK**\n' +
        '────────────────\n' +
        'Send me a face photo.\n' +
        'I return biometric prompt codes for AI character consistency.\n\n' +
        'Supported platforms: Reve, Flux, Midjourney.\n\n' +
        'Commands:\n' +
        '/scan — Send a photo after this\n' +
        '/platform <name> — Set target platform\n' +
        '/status — Check your tier',
        { parse_mode: 'Markdown' }
    );
});

// /platform command — store user preference
bot.command('platform', (ctx) => {
    const arg = (ctx.message.text.split(' ')[1] || '').toLowerCase();
    const valid = ['reve', 'flux', 'midjourney', 'generic'];

    if (!valid.includes(arg)) {
        return ctx.reply(
            `⚠️ Invalid platform. Choose one:\n` +
            valid.map(p => `• \`${p}\``).join('\n'),
            { parse_mode: 'Markdown' }
        );
    }

    // Store in a simple in-memory map (per-session)
    if (!global.userPrefs) global.userPrefs = {};
    global.userPrefs[ctx.from.id] = arg;
    ctx.reply(`✅ Platform set to **${arg}**. Send a photo to scan.`, { parse_mode: 'Markdown' });
});

// /status command
bot.command('status', (ctx) => {
    ctx.reply(`User ID: \`${ctx.from.id}\`\nTier: checking...`, { parse_mode: 'Markdown' });
});

// --- PHOTO HANDLER (core loop) ---
bot.on('photo', async (ctx) => {
    const userId = ctx.from.id.toString();
    const platform = (global.userPrefs && global.userPrefs[ctx.from.id]) || 'generic';

    // Get highest resolution photo
    const photos = ctx.message.photo;
    const bestPhoto = photos[photos.length - 1];

    try {
        // Status message
        const statusMsg = await ctx.reply('🔍 Scanning...');

        // Get file URL from Telegram
        const fileLink = await ctx.telegram.getFileLink(bestPhoto.file_id);
        const imageUrl = fileLink.href || fileLink.toString();

        // Execute the face_lock skill
        const result = await faceLockSkill.execute({
            image_url: imageUrl,
            user_id: userId,
            platform: platform,
        });

        // Delete "Scanning..." and send result
        await ctx.telegram.deleteMessage(ctx.chat.id, statusMsg.message_id).catch(() => {});
        await ctx.reply(result, { parse_mode: 'Markdown' });

    } catch (error) {
        console.error('Photo handler error:', error);
        ctx.reply('⚠️ **System Error.** Try again with a clear frontal headshot.');
    }
});

// Any non-photo message
bot.on('message', (ctx) => {
    // Ignore commands (already handled above)
    if (ctx.message.text && ctx.message.text.startsWith('/')) return;
    ctx.reply('Send me a reference photo to extract the face lock prompts.');
});

// --- ERROR HANDLING ---
bot.catch((err, ctx) => {
    console.error(`Bot error for ${ctx.updateType}:`, err);
});

// --- LAUNCH ---
bot.launch().then(() => {
    console.log('Face Lock bot is running.');
});

// Graceful shutdown
process.once('SIGINT', () => { bot.stop('SIGINT'); healthServer.close(); });
process.once('SIGTERM', () => { bot.stop('SIGTERM'); healthServer.close(); });
