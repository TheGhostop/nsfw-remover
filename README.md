

# 🛡️ NSFW Content Moderation Telegram Bot

A powerful Telegram group bot for detecting, deleting, and auto-muting users posting NSFW content (images, videos, stickers, GIFs, documents). Brings safe automation, modern interface, and easy admin controls for any community.

***

## ✨ Features

- **Accurate NSFW detection:**  
  Analyzes photos, videos, stickers, GIFs, and more with advanced skin/flesh/curve logic for high precision.

- **Auto-delete & warn ⚠️:**  
  Instantly removes NSFW content and warns the sender. 5 strikes = auto-mute (duration customizable).

- **Smart mute/unmute:**  
  After 5 warnings, user is muted with a 🔊 unmute button for group admins.

- **Easy add-to-group:**  
  `/start` sends a welcome image with a ➕ button—add the bot to any group in a tap.

- **Inline UI and real emojis:**  
  All alerts, actions, and menus use Unicode emojis and beautiful inline buttons for modern interaction.

- **Global ban & whitelisting:**  
  Ban users across groups, whitelist exceptions, and reset warnings.

- **Detailed logs:**  
  Each moderation action is logged for transparency and debugging.

***

## 🚀 Quick Start

### 1. **Clone and run**
```bash
git clone <your-repo-url>
cd nsfw-remover
pip install -r requirements.txt  # install Python, aiogram, pillow, opencv, pymongo, etc.

python3 bot.py
```

### 2. **Configure**

Edit variables in `bot.py`:
- Your bot token (`BOT_TOKEN`)
- MongoDB connection (`MONGODB_URI`), DB name, etc.

### 3. **Add to Group**

- Send `/start` to the bot in DM or group.
- Use the ➕ **Add to Group** button, or add manually.
- Make the bot **admin** in your group with “delete messages” and “restrict members” permissions.

***

## 🖼️ Example Welcome Message

When /start is sent, the bot replies with:

```
🛡️ NSFW Content Moderation Bot

✅ Auto-detects & deletes NSFW
✅ Smart warning system (5 = mute)
✅ Global ban system
✅ Admin controls

[Welcome image + ➕ Add to Group button]
```

***

## 🧑‍💻 Main Commands

| Command         | Description                    |
|-----------------|-------------------------------|
| /start          | Show welcome (image + add)    |
| /help           | Show help and tips            |
| /stats          | View bot/group stats (owner)  |
| /warn_status    | View warning count            |
| /warn_reset     | Reset warning (admin, reply)  |
| /whitelist_add  | Whitelist member (admin)      |
| /gban           | Globally ban (owner)          |
| /ungban         | Remove global ban (owner)     |

***

## ⚙️ Admin Actions

- **Warn & Mute:**  
  Users get 5 warnings (⚠️) before being muted (🔇). Mute message has an admin-unmute button.

- **Unmute:**  
  Admins can tap 🔊 “Unmute User” in the mute message. The bot restores full permissions instantly.

- **Whitelisting:**  
  Don’t want NSFW detection for someone? `/whitelist_add` as admin (reply to their message).

***

## 📝 Example Workflow

1. User posts NSFW image →  
   ⏩ Bot deletes, replies — “⚠️ Warning (1/5)”
2. After 5 warnings:  
   ⏩ Bot mutes user, sends mute with 🔊 Unmute button
3. Admin taps 🔊:  
   ⏩ User is unmuted, admin noted

***

## 🎨 Customization

- **Welcome image:**  
  Edit `WELCOME_IMAGE_URL` at top of your script.

- **Warning limit:**  
  Change `WARN_LIMIT = 5` to any value.

- **Mute duration:**  
  Edit `MUTE_DURATION = 1800` (seconds).

***

## 💡 Tips

- All inline and alert emojis are real Unicode.
- Make sure bot’s environment (Python terminal, logs) uses UTF-8 encoding, else emojis may be garbled.
- For best results, connect to a MongoDB Atlas (free tier is enough).

***

## ☑️ Requirements

- Python 3.8+
- `aiogram` (Telegram Bot API framework)
- `opencv-python` (CV for NSFW analysis)
- `pillow` (for format conversion)
- `pymongo` (MongoDB database)
- `ffmpeg` installed in your OS for video frame extraction

***

## 🔗 License & Credits

- Built by [your name/handle].
- Open source, MIT license.
- Inspired by Telegram community moderation needs and open-source NSFW research.

***

### 🙏 PRs, issues, and feature requests welcome!  
For support, DM [Your Telegram] or open an issue.

***

**Safe communities, one group at a time!** 🛡️🚀

Citations:
[1] 1000347921.jpg https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/82461510/533927b0-4d2d-4fb6-ad82-cbd5fd70be15/1000347921.jpg
