# moderate_bot.py
import asyncio
import os
import re
import subprocess
import tempfile
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from nudenet import NudeClassifier
from PIL import Image
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

# Configuration
BOT_TOKEN = "8498609067:AAFumiioVKuUP-HM9PEC7FveoE9j-2H8rDo"
OWNER_IDS = [7641743441, 6361404699]
LOGGER_GROUP_ID = -1002529491709
MONGODB_URI = "mongodb+srv://deathhide08:UZYSj9T0VuAgIFAB@cluster0.elg19jx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "nsfw_db"

# Bot Settings
WARN_LIMIT = 3
NSFW_THRESHOLD = 0.70
MAX_VIDEO_FRAMES = 8
VIDEO_FPS = 1
MUTE_DURATION = 1800  # 30 minutes
ALLOWED_PRIVATE = False

# Initialize bot and components
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('moderation_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Initialize MongoDB with motor for async operations
mongo = AsyncIOMotorClient(MONGODB_URI)
db = mongo[DATABASE_NAME]
warns_col = db.warns
log_col = db.logs
whitelist_col = db.whitelist
gban_col = db.gban
chats_col = db.chats
users_col = db.users

# Initialize classifier (lazy loading)
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        logger.info("Initializing NudeClassifier...")
        classifier = NudeClassifier()
    return classifier

# Utility functions
def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks"""
    return re.sub(r'[^\w\.-]', '_', filename)

@asynccontextmanager
async def temporary_download(file_obj, file_extension: str = ""):
    """Context manager for safe temporary file handling"""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            temp_path = tmp.name
            await file_obj.download(destination=temp_path)
            temp_file = temp_path
            yield temp_file
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

async def extract_frames(video_path: str, out_dir: str, fps: int = 1, max_frames: int = 10) -> List[str]:
    """Extract frames from video using ffmpeg"""
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "3",
        "-frames:v", str(max_frames),
        os.path.join(out_dir, "frame_%03d.jpg")
    ]
    
    try:
        subprocess.run(cmd, check=True, timeout=30)
        frames = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg')])
        return frames[:max_frames]
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timeout while extracting frames")
        return []
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"FFmpeg error: {e}")
        return []

def convert_webp_to_png(webp_path: str) -> str:
    """Convert WEBP to PNG and return new path"""
    png_path = webp_path + ".png"
    try:
        with Image.open(webp_path) as im:
            im.convert("RGB").save(png_path, "PNG")
        return png_path
    except Exception as e:
        logger.error(f"WEBP conversion failed: {e}")
        return webp_path

async def analyze_images(paths: List[str]) -> Tuple[bool, Optional[dict]]:
    """Analyze images for NSFW content"""
    if not paths:
        return False, None
        
    classifier = get_classifier()
    try:
        results = classifier.classify(paths)
        for path, result in results.items():
            for label, score in result.items():
                if (score >= NSFW_THRESHOLD and 
                    label.lower() in ("porn", "unsafe", "sexy", "hentai")):
                    return True, {"path": path, "label": label, "score": score}
        return False, None
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return False, None

async def is_user_whitelisted(chat_id: int, user_id: int) -> bool:
    """Check if user is whitelisted"""
    try:
        doc = await whitelist_col.find_one({"chat_id": chat_id, "user_id": user_id})
        return doc is not None
    except PyMongoError as e:
        logger.error(f"Database error checking whitelist: {e}")
        return False

async def is_user_admin(chat_id: int, user_id: int) -> bool:
    """Check if user is admin in the chat"""
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        return member.is_chat_admin()
    except Exception as e:
        logger.error(f"Failed to check admin status: {e}")
        return False

async def is_owner(user_id: int) -> bool:
    """Check if user is bot owner"""
    return user_id in OWNER_IDS

async def log_to_channel(message: str):
    """Send log message to logger group"""
    try:
        await bot.send_message(LOGGER_GROUP_ID, message)
    except Exception as e:
        logger.error(f"Failed to send log to channel: {e}")

async def update_chat_data(chat_id: int, chat_title: str = ""):
    """Update chat information in database"""
    try:
        await chats_col.update_one(
            {"chat_id": chat_id},
            {
                "$set": {
                    "chat_title": chat_title,
                    "last_seen": time.time(),
                    "is_group": chat_id < 0
                }
            },
            upsert=True
        )
    except PyMongoError as e:
        logger.error(f"Failed to update chat data: {e}")

async def update_user_data(user_id: int, username: str = "", first_name: str = ""):
    """Update user information in database"""
    try:
        await users_col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "username": username,
                    "first_name": first_name,
                    "last_seen": time.time()
                }
            },
            upsert=True
        )
    except PyMongoError as e:
        logger.error(f"Failed to update user data: {e}")

async def is_user_gbanned(user_id: int) -> bool:
    """Check if user is globally banned"""
    try:
        doc = await gban_col.find_one({"user_id": user_id, "banned": True})
        return doc is not None
    except PyMongoError as e:
        logger.error(f"Database error checking gban: {e}")
        return False

async def gban_user(user_id: int, reason: str = "No reason provided", banned_by: int = 0):
    """Globally ban a user"""
    try:
        user_info = await users_col.find_one({"user_id": user_id})
        username = user_info.get("username", "") if user_info else ""
        first_name = user_info.get("first_name", "") if user_info else ""
        
        await gban_col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "username": username,
                    "first_name": first_name,
                    "reason": reason,
                    "banned_by": banned_by,
                    "banned_at": time.time(),
                    "banned": True
                }
            },
            upsert=True
        )
        
        # Ban user from all groups where bot is admin
        banned_count = 0
        async for chat in chats_col.find({"is_group": True}):
            try:
                chat_id = chat["chat_id"]
                await bot.ban_chat_member(chat_id, user_id)
                banned_count += 1
            except Exception as e:
                logger.error(f"Failed to ban from chat {chat_id}: {e}")
                
        return banned_count
    except PyMongoError as e:
        logger.error(f"Failed to gban user: {e}")
        return 0

async def ungban_user(user_id: int) -> bool:
    """Remove global ban from user"""
    try:
        result = await gban_col.update_one(
            {"user_id": user_id},
            {"$set": {"banned": False}}
        )
        return result.modified_count > 0
    except PyMongoError as e:
        logger.error(f"Failed to ungban user: {e}")
        return False

async def get_bot_stats() -> Dict[str, Any]:
    """Get bot statistics"""
    try:
        total_chats = await chats_col.count_documents({})
        total_groups = await chats_col.count_documents({"is_group": True})
        total_users = await users_col.count_documents({})
        total_gbanned = await gban_col.count_documents({"banned": True})
        total_warns = await warns_col.count_documents({})
        total_logs = await log_col.count_documents({})
        
        return {
            "total_chats": total_chats,
            "total_groups": total_groups,
            "total_users": total_users,
            "total_gbanned": total_gbanned,
            "total_warns": total_warns,
            "total_logs": total_logs
        }
    except PyMongoError as e:
        logger.error(f"Failed to get stats: {e}")
        return {}

async def broadcast_message(message: Message, forward: bool = False) -> Dict[str, int]:
    """Broadcast message to all chats"""
    stats = {"success": 0, "failed": 0, "pinned": 0}
    
    async for chat in chats_col.find({}):
        try:
            chat_id = chat["chat_id"]
            
            if forward:
                # Forward the message
                sent_msg = await bot.forward_message(
                    chat_id=chat_id,
                    from_chat_id=message.chat.id,
                    message_id=message.message_id
                )
            else:
                # Copy the message
                sent_msg = await message.copy_to(chat_id)
            
            stats["success"] += 1
            
            # Pin message in groups
            if chat_id < 0:  # Group chat
                try:
                    await bot.pin_chat_message(chat_id, sent_msg.message_id)
                    stats["pinned"] += 1
                except Exception:
                    pass  # Can't pin in some groups
                    
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"Broadcast failed for chat {chat_id}: {e}")
    
    return stats

async def handle_detect_and_action(message: Message, file_path: str, origin: str = "media"):
    """Handle NSFW detection and take appropriate action"""
    user_id = message.from_user.id if message.from_user else None
    chat_id = message.chat.id
    
    # Check if user is globally banned
    if user_id and await is_user_gbanned(user_id):
        try:
            await message.delete()
            await message.answer(f"🚫 User is globally banned and cannot send messages.")
            return
        except Exception as e:
            logger.error(f"Failed to handle gbanned user: {e}")
    
    # Skip if user is admin or whitelisted
    if user_id:
        if await is_user_admin(chat_id, user_id) or await is_user_whitelisted(chat_id, user_id):
            logger.info(f"Skipping check for admin/whitelisted user {user_id}")
            return

    is_nsfw = False
    reason = None
    temp_files_to_cleanup = []

    try:
        # Handle different media types
        if file_path.lower().endswith((".mp4", ".mkv", ".webm", ".mov", ".avi")):
            # Video processing
            with tempfile.TemporaryDirectory() as tmpdir:
                frames = await extract_frames(file_path, tmpdir, fps=VIDEO_FPS, max_frames=MAX_VIDEO_FRAMES)
                if frames:
                    is_nsfw, reason = await analyze_images(frames)
        elif file_path.lower().endswith(".webp"):
            # WEBP processing
            png_path = convert_webp_to_png(file_path)
            if png_path != file_path:
                temp_files_to_cleanup.append(png_path)
            is_nsfw, reason = await analyze_images([png_path])
        else:
            # Image processing
            is_nsfw, reason = await analyze_images([file_path])

        if is_nsfw:
            await take_moderation_action(message, reason, user_id, chat_id)
            
    except Exception as e:
        logger.error(f"Error in handle_detect_and_action: {e}")
    finally:
        # Cleanup temporary files
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Cleanup error for {temp_file}: {e}")

async def take_moderation_action(message: Message, reason: dict, user_id: int, chat_id: int):
    """Take moderation actions for NSFW content"""
    try:
        # Delete the offensive message
        await message.delete()
        logger.info(f"Deleted NSFW message from user {user_id} in chat {chat_id}")
        
        # Log to channel
        user_name = message.from_user.full_name if message.from_user else "Unknown"
        await log_to_channel(f"🚫 NSFW Detected\nUser: {user_name} ({user_id})\nChat: {message.chat.title}\nReason: {reason}")
        
    except Exception as e:
        logger.error(f"Failed to delete message: {e}")

    # Update warning count
    try:
        result = await warns_col.update_one(
            {"chat_id": chat_id, "user_id": user_id},
            {
                "$inc": {"warns": 1},
                "$set": {
                    "last_reason": reason,
                    "last_warn": time.time(),
                    "username": message.from_user.username if message.from_user else None
                }
            },
            upsert=True
        )
        
        # Get current warning count
        doc = await warns_col.find_one({"chat_id": chat_id, "user_id": user_id})
        current_warns = doc.get("warns", 1) if doc else 1
        
        # Log the action
        await log_col.insert_one({
            "chat_id": chat_id,
            "user_id": user_id,
            "message_id": message.message_id,
            "reason": reason,
            "timestamp": time.time(),
            "media_type": message.content_type,
            "warn_count": current_warns
        })
        
        # Notify chat about warning
        try:
            user_name = message.from_user.full_name if message.from_user else "User"
            warning_msg = await message.answer(
                f"⚠️ {user_name} has been warned ({current_warns}/{WARN_LIMIT}) - NSFW content detected."
            )
            # Delete warning message after 10 seconds
            await asyncio.sleep(10)
            await warning_msg.delete()
        except Exception:
            pass

        # Apply mute if over limit
        if current_warns >= WARN_LIMIT:
            await mute_user(chat_id, user_id, message.from_user.full_name if message.from_user else "User")
            
    except PyMongoError as e:
        logger.error(f"Database error during moderation: {e}")

async def mute_user(chat_id: int, user_id: int, user_name: str):
    """Mute user in the chat"""
    try:
        until_date = int(time.time()) + MUTE_DURATION
        
        await bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=types.ChatPermissions(
                can_send_messages=False,
                can_send_media_messages=False,
                can_send_other_messages=False,
                can_add_web_page_previews=False
            ),
            until_date=until_date
        )
        
        mute_msg = await bot.send_message(
            chat_id, 
            f"🔇 {user_name} has been muted for {MUTE_DURATION // 60} minutes due to repeated NSFW violations."
        )
        
        logger.info(f"Muted user {user_id} in chat {chat_id}")
        
        # Delete mute message after 15 seconds
        await asyncio.sleep(15)
        await mute_msg.delete()
        
    except Exception as e:
        logger.error(f"Failed to mute user {user_id}: {e}")

# Message handlers
@dp.message()
async def all_messages(msg: Message):
    """Handle all incoming messages"""
    # Update chat and user data
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)

    # Skip private chats unless allowed
    if msg.chat.type == "private" and not ALLOWED_PRIVATE:
        return

    # Skip if message is from anonymous user
    if not msg.from_user:
        return

    # Handle different media types
    try:
        if msg.photo:
            async with temporary_download(await msg.photo[-1].get_file(), ".jpg") as file_path:
                await handle_detect_and_action(msg, file_path, "photo")
                
        elif msg.sticker:
            async with temporary_download(await msg.sticker.get_file(), ".webp") as file_path:
                await handle_detect_and_action(msg, file_path, "sticker")
                
        elif msg.video:
            async with temporary_download(await msg.video.get_file(), ".mp4") as file_path:
                await handle_detect_and_action(msg, file_path, "video")
                
        elif msg.document:
            # Check if document is likely an image or video
            mime_type = msg.document.mime_type or ""
            if mime_type.startswith(('image/', 'video/')):
                async with temporary_download(await msg.document.get_file()) as file_path:
                    await handle_detect_and_action(msg, file_path, "document")
                    
        elif msg.animation:  # GIFs
            async with temporary_download(await msg.animation.get_file(), ".mp4") as file_path:
                await handle_detect_and_action(msg, file_path, "animation")
                
    except Exception as e:
        logger.error(f"Error processing message {msg.message_id}: {e}")

# Owner Commands
@dp.message(Command("broadcast"))
async def broadcast_cmd(msg: Message):
    """Broadcast message to all chats (owner only)"""
    if not await is_owner(msg.from_user.id):
        return
        
    if not msg.reply_to_message:
        await msg.reply("Please reply to a message to broadcast it.")
        return
        
    processing_msg = await msg.reply("🔄 Starting broadcast...")
    
    try:
        stats = await broadcast_message(msg.reply_to_message, forward=True)
        
        result_text = (
            f"📢 Broadcast Completed!\n\n"
            f"✅ Success: {stats['success']}\n"
            f"❌ Failed: {stats['failed']}\n"
            f"📌 Pinned: {stats['pinned']}"
        )
        
        await processing_msg.edit_text(result_text)
        await log_to_channel(f"📢 Broadcast by {msg.from_user.full_name}\n{result_text}")
        
    except Exception as e:
        await processing_msg.edit_text(f"❌ Broadcast failed: {str(e)}")

@dp.message(Command("gban"))
async def gban_cmd(msg: Message):
    """Globally ban a user (owner only)"""
    if not await is_owner(msg.from_user.id):
        return
        
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("Please reply to a user's message to gban them.")
        return
        
    target_user = msg.reply_to_message.from_user
    reason = " ".join(msg.text.split()[1:]) or "No reason provided"
    
    processing_msg = await msg.reply(f"🔄 GBanning user {target_user.full_name}...")
    
    try:
        banned_count = await gban_user(target_user.id, reason, msg.from_user.id)
        
        result_text = (
            f"🚫 User {target_user.full_name} has been globally banned!\n"
            f"📝 Reason: {reason}\n"
            f"👥 Banned from: {banned_count} groups"
        )
        
        await processing_msg.edit_text(result_text)
        await log_to_channel(f"🚫 GBan by {msg.from_user.full_name}\n{result_text}")
        
    except Exception as e:
        await processing_msg.edit_text(f"❌ GBan failed: {str(e)}")

@dp.message(Command("ungban"))
async def ungban_cmd(msg: Message):
    """Remove global ban from user (owner only)"""
    if not await is_owner(msg.from_user.id):
        return
        
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("Please reply to a user's message to ungban them.")
        return
        
    target_user = msg.reply_to_message.from_user
    
    processing_msg = await msg.reply(f"🔄 Removing gban from {target_user.full_name}...")
    
    try:
        success = await ungban_user(target_user.id)
        
        if success:
            result_text = f"✅ Global ban removed from {target_user.full_name}"
            await processing_msg.edit_text(result_text)
            await log_to_channel(f"✅ Ungban by {msg.from_user.full_name}\n{result_text}")
        else:
            await processing_msg.edit_text("❌ User was not gbanned or operation failed")
            
    except Exception as e:
        await processing_msg.edit_text(f"❌ Ungban failed: {str(e)}")

@dp.message(Command("stats"))
async def stats_cmd(msg: Message):
    """Show bot statistics (owner only)"""
    if not await is_owner(msg.from_user.id):
        return
        
    processing_msg = await msg.reply("🔄 Gathering statistics...")
    
    try:
        stats = await get_bot_stats()
        
        stats_text = (
            "🤖 Bot Statistics\n\n"
            f"💬 Total Chats: {stats.get('total_chats', 0)}\n"
            f"👥 Groups: {stats.get('total_groups', 0)}\n"
            f"👤 Users: {stats.get('total_users', 0)}\n"
            f"🚫 GBanned Users: {stats.get('total_gbanned', 0)}\n"
            f"⚠️ Total Warnings: {stats.get('total_warns', 0)}\n"
            f"📊 Total Logs: {stats.get('total_logs', 0)}"
        )
        
        await processing_msg.edit_text(stats_text)
        
    except Exception as e:
        await processing_msg.edit_text(f"❌ Failed to get stats: {str(e)}")

@dp.message(Command("gbanlist"))
async def gbanlist_cmd(msg: Message):
    """Show gbanned users list (owner only)"""
    if not await is_owner(msg.from_user.id):
        return
        
    try:
        gbanned_users = []
        async for user in gban_col.find({"banned": True}).limit(50):
            user_info = f"• {user.get('first_name', 'Unknown')} (@{user.get('username', 'N/A')}) - {user.get('reason', 'No reason')}"
            gbanned_users.append(user_info)
        
        if gbanned_users:
            text = "🚫 GBanned Users:\n\n" + "\n".join(gbanned_users)
            if len(gbanned_users) == 50:
                text += "\n\n... and more"
        else:
            text = "✅ No users are currently gbanned."
            
        await msg.reply(text)
        
    except Exception as e:
        await msg.reply(f"❌ Failed to get gbanlist: {str(e)}")

# Admin Commands
@dp.message(Command("warn_reset"))
async def reset_warns_cmd(msg: Message):
    """Reset warnings for a user (admin only)"""
    if not await is_user_admin(msg.chat.id, msg.from_user.id):
        return
        
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("Please reply to a user's message to reset their warnings.")
        return
        
    target_user = msg.reply_to_message.from_user
    try:
        await warns_col.delete_one({"chat_id": msg.chat.id, "user_id": target_user.id})
        response = await msg.reply(f"✅ Warnings reset for {target_user.full_name}")
        await asyncio.sleep(10)
        await response.delete()
        await msg.delete()
    except PyMongoError as e:
        await msg.reply("❌ Failed to reset warnings.")

@dp.message(Command("warn_status"))
async def warn_status_cmd(msg: Message):
    """Check warning status for a user"""
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("Please reply to a user's message to check their warnings.")
        return
        
    target_user = msg.reply_to_message.from_user
    try:
        doc = await warns_col.find_one({"chat_id": msg.chat.id, "user_id": target_user.id})
        warns = doc.get("warns", 0) if doc else 0
        await msg.reply(f"⚠️ {target_user.full_name} has {warns}/{WARN_LIMIT} warnings")
    except PyMongoError as e:
        await msg.reply("❌ Failed to check warning status.")

@dp.message(Command("whitelist_add"))
async def whitelist_add_cmd(msg: Message):
    """Add user to whitelist (admin only)"""
    if not await is_user_admin(msg.chat.id, msg.from_user.id):
        return
        
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("Please reply to a user's message to whitelist them.")
        return
        
    target_user = msg.reply_to_message.from_user
    try:
        await whitelist_col.update_one(
            {"chat_id": msg.chat.id, "user_id": target_user.id},
            {"$set": {"username": target_user.username, "full_name": target_user.full_name}},
            upsert=True
        )
        response = await msg.reply(f"✅ {target_user.full_name} added to whitelist")
        await asyncio.sleep(10)
        await response.delete()
        await msg.delete()
    except PyMongoError as e:
        await msg.reply("❌ Failed to add user to whitelist.")

async def setup_database():
    """Setup database indexes"""
    try:
        await warns_col.create_index([("chat_id", 1), ("user_id", 1)], unique=True)
        await warns_col.create_index([("last_warn", 1)], expireAfterSeconds=30*24*3600)  # 30 days TTL
        await log_col.create_index([("chat_id", 1), ("timestamp", -1)])
        await whitelist_col.create_index([("chat_id", 1), ("user_id", 1)], unique=True)
        await gban_col.create_index([("user_id", 1)], unique=True)
        await chats_col.create_index([("chat_id", 1)], unique=True)
        await users_col.create_index([("user_id", 1)], unique=True)
        logger.info("Database indexes created successfully")
    except PyMongoError as e:
        logger.error(f"Failed to create database indexes: {e}")

async def main():
    """Main function to start the bot"""
    await setup_database()
    await log_to_channel("🤖 Bot Started Successfully!")
    logger.info("Bot started successfully")
    
    # Start polling
    await dp.start_polling(bot)

if __name__ == "__main__":
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    
    # Start the bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
