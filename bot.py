import asyncio
import os
import re
import subprocess
import tempfile
import logging
import time
import cv2
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Tuple, List, Dict, Any

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters.command import Command
from aiogram.types import ChatPermissions
from aiogram.exceptions import TelegramAPIError
from PIL import Image
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

# ==================== CONFIGURATION ====================
BOT_TOKEN = "8395371421:AAHeXyUrhwFf-4WdgLe3eU5xCymdCH1snyA"
OWNER_IDS = [7641743441, 6361404699]
LOGGER_GROUP_ID = -1002529491709
MONGODB_URI = "mongodb+srv://deathhide08:UZYSj9T0VuAgIFAB@cluster0.elg19jx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "nsfw_db"

# Bot Settings
WARN_LIMIT = 3
NSFW_THRESHOLD = 0.5
MAX_VIDEO_FRAMES = 6
VIDEO_FPS = 1
MUTE_DURATION = 1800
ALLOWED_PRIVATE = True

# Initialize bot and dispatcher
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

# MongoDB setup
mongo = AsyncIOMotorClient(MONGODB_URI)
db = mongo[DATABASE_NAME]
warns_col = db.warns
log_col = db.logs
whitelist_col = db.whitelist
gban_col = db.gban
chats_col = db.chats
users_col = db.users

def detect_nsfw_opencv(image_path: str) -> Tuple[bool, Optional[dict]]:
    """Detect NSFW using OpenCV skin detection"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return False, None
        
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        total_pixels = mask.size
        skin_pixels = cv2.countNonZero(mask)
        skin_percentage = skin_pixels / total_pixels
        
        logger.info(f"Skin detection: {skin_percentage:.2%} of image is skin-colored")
        
        if skin_percentage >= NSFW_THRESHOLD:
            return True, {"method": "skin_detection", "skin_percentage": skin_percentage}
        
        return False, None
        
    except Exception as e:
        logger.error(f"OpenCV NSFW detection failed: {e}")
        return False, None

@asynccontextmanager
async def temporary_download(downloadable, file_extension: str = ""):
    """Download file from Telegram"""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            temp_path = tmp.name
            await bot.download(downloadable, destination=temp_path)
            temp_file = temp_path
            logger.info(f"Downloaded file to {temp_path}")
            yield temp_file
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        yield None
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.info(f"Cleaned up temp file {temp_file}")
            except Exception as e:
                logger.error(f"Failed to delete temp file {temp_file}: {e}")

def convert_webp_to_jpg_pil(webp_path: str) -> str:
    """Convert WEBP to JPG using PIL directly"""
    jpg_path = webp_path.replace('.webp', '.jpg')
    try:
        # Open WEBP and convert to RGB
        with Image.open(webp_path) as im:
            # Convert RGBA to RGB if needed
            if im.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[-1] if im.mode in ('RGBA', 'LA') else None)
                background.save(jpg_path, "JPEG", quality=95)
            else:
                im.convert("RGB").save(jpg_path, "JPEG", quality=95)
        
        logger.info(f"Successfully converted WEBP to JPG: {jpg_path}")
        return jpg_path
    except Exception as e:
        logger.error(f"PIL WEBP conversion failed: {e}")
        return webp_path

async def extract_frames(video_path: str, out_dir: str, fps: int = 1, max_frames: int = 6) -> List[str]:
    """Extract frames from video/animation"""
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
        subprocess.run(cmd, check=True, timeout=60)
        frames = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg')])
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames[:max_frames]
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timeout while extracting frames")
        return []
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"FFmpeg error: {e}")
        return []

async def analyze_image(image_path: str) -> Tuple[bool, Optional[dict]]:
    """Analyze single image"""
    if not image_path or not os.path.exists(image_path):
        return False, None
    
    try:
        is_nsfw, result = detect_nsfw_opencv(image_path)
        return is_nsfw, result
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return False, None

async def analyze_images(paths: List[str]) -> Tuple[bool, Optional[dict]]:
    """Analyze multiple images"""
    if not paths:
        return False, None
    
    try:
        logger.info(f"Analyzing {len(paths)} image(s) for NSFW content...")
        
        for path in paths:
            is_nsfw, result = await analyze_image(path)
            if is_nsfw:
                logger.warning(f"NSFW detected in {path}: {result}")
                return True, result
        
        return False, None
    except Exception as e:
        logger.error(f"Batch image analysis failed: {e}")
        return False, None

async def is_user_whitelisted(chat_id: int, user_id: int) -> bool:
    try:
        doc = await whitelist_col.find_one({"chat_id": chat_id, "user_id": user_id})
        return doc is not None
    except PyMongoError as e:
        logger.error(f"Database error checking whitelist: {e}")
        return False

async def is_user_admin(chat_id: int, user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        return member.status in ["creator", "administrator"]
    except TelegramAPIError as e:
        logger.error(f"Failed to check admin status: {e}")
        return False

async def is_owner(user_id: int) -> bool:
    return user_id in OWNER_IDS

async def log_to_channel(message: str):
    try:
        await bot.send_message(LOGGER_GROUP_ID, message, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Failed to send log to channel: {e}")

async def update_chat_data(chat_id: int, chat_title: str = ""):
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
    try:
        doc = await gban_col.find_one({"user_id": user_id, "banned": True})
        return doc is not None
    except PyMongoError as e:
        logger.error(f"Database error checking gban: {e}")
        return False

async def gban_user(user_id: int, reason: str = "No reason provided", banned_by: int = 0):
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

async def broadcast_message(message: types.Message, forward: bool = False) -> Dict[str, int]:
    stats = {"success": 0, "failed": 0, "pinned": 0}
    
    async for chat in chats_col.find({}):
        try:
            chat_id = chat["chat_id"]
            
            if forward and message.forward_from_chat:
                sent_msg = await bot.forward_message(
                    chat_id=chat_id,
                    from_chat_id=message.chat.id,
                    message_id=message.message_id
                )
            else:
                if message.text:
                    sent_msg = await bot.send_message(chat_id, message.text)
                elif message.caption:
                    if message.photo:
                        sent_msg = await bot.send_photo(
                            chat_id, 
                            message.photo[-1].file_id, 
                            caption=message.caption
                        )
                    elif message.video:
                        sent_msg = await bot.send_video(
                            chat_id, 
                            message.video.file_id, 
                            caption=message.caption
                        )
                    elif message.document:
                        sent_msg = await bot.send_document(
                            chat_id, 
                            message.document.file_id, 
                            caption=message.caption
                        )
                    else:
                        sent_msg = await bot.copy_message(chat_id, message.chat.id, message.message_id)
                else:
                    sent_msg = await bot.copy_message(chat_id, message.chat.id, message.message_id)
            
            stats["success"] += 1
            
            if chat_id < 0:
                try:
                    await bot.pin_chat_message(chat_id, sent_msg.message_id)
                    stats["pinned"] += 1
                except Exception:
                    pass
                    
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"Broadcast failed for chat {chat_id}: {e}")
    
    return stats

async def handle_detect_and_action(message: types.Message, file_path: str, media_type: str = "media"):
    """Analyze media and take action if NSFW"""
    user_id = message.from_user.id if message.from_user else None
    chat_id = message.chat.id
    
    if not file_path:
        logger.error("File path is None, skipping analysis")
        return
    
    if user_id and await is_user_gbanned(user_id):
        try:
            await message.delete()
            await message.answer("User is globally banned")
            return
        except Exception as e:
            logger.error(f"Failed to handle gbanned user: {e}")
    
    if user_id:
        if await is_user_admin(chat_id, user_id) or await is_user_whitelisted(chat_id, user_id):
            logger.info(f"Skipping check for admin/whitelisted user {user_id}")
            return

    is_nsfw = False
    reason = None
    temp_files_to_cleanup = []

    try:
        file_lower = file_path.lower()
        
        if file_lower.endswith((".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".3gp")):
            logger.info(f"Processing video ({media_type}): {file_path}")
            with tempfile.TemporaryDirectory() as tmpdir:
                frames = await extract_frames(file_path, tmpdir, fps=VIDEO_FPS, max_frames=MAX_VIDEO_FRAMES)
                if frames:
                    is_nsfw, reason = await analyze_images(frames)
                else:
                    logger.warning("No frames extracted from video")
                    
        elif file_lower.endswith(".webp"):
            logger.info(f"Processing WEBP ({media_type}): {file_path}")
            # Try PIL conversion for static stickers
            jpg_path = convert_webp_to_jpg_pil(file_path)
            if jpg_path != file_path and os.path.exists(jpg_path):
                temp_files_to_cleanup.append(jpg_path)
                is_nsfw, reason = await analyze_images([jpg_path])
            else:
                # Fallback: analyze WEBP directly with PIL
                try:
                    with Image.open(file_path) as im:
                        if im.mode in ('RGBA', 'LA', 'P'):
                            background = Image.new('RGB', im.size, (255, 255, 255))
                            background.paste(im, mask=im.split()[-1] if im.mode in ('RGBA', 'LA') else None)
                        else:
                            background = im.convert("RGB")
                        jpg_path = file_path + "_fallback.jpg"
                        background.save(jpg_path, "JPEG", quality=95)
                        temp_files_to_cleanup.append(jpg_path)
                        is_nsfw, reason = await analyze_images([jpg_path])
                except Exception as e:
                    logger.error(f"WEBP fallback failed: {e}")
            
        elif file_lower.endswith((".gif", ".gifv")):
            logger.info(f"Processing GIF ({media_type}): {file_path}")
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-i", file_path,
                    "-vf", f"fps={VIDEO_FPS}",
                    "-q:v", "3",
                    "-frames:v", str(MAX_VIDEO_FRAMES),
                    os.path.join(tmpdir, "frame_%03d.jpg")
                ]
                try:
                    subprocess.run(cmd, check=True, timeout=30)
                    frames = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.jpg')])
                    if frames:
                        is_nsfw, reason = await analyze_images(frames)
                except:
                    is_nsfw, reason = await analyze_images([file_path])
                    
        else:  # Image formats
            logger.info(f"Processing image ({media_type}): {file_path}")
            is_nsfw, reason = await analyze_images([file_path])

        if is_nsfw:
            logger.warning(f"NSFW content detected! Reason: {reason}")
            await take_moderation_action(message, reason, user_id, chat_id)
        else:
            logger.info(f"Content is safe")
            
    except Exception as e:
        logger.error(f"Error in handle_detect_and_action: {e}")
    finally:
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Cleanup error for {temp_file}: {e}")

async def take_moderation_action(message: types.Message, reason: dict, user_id: int, chat_id: int):
    """Take moderation action"""
    try:
        await message.delete()
        logger.info(f"Deleted NSFW message from user {user_id} in chat {chat_id}")
        
        user_name = message.from_user.full_name if message.from_user else "Unknown"
        log_msg = f"NSFW Detected\nUser: {user_name} ({user_id})\nChat: {message.chat.title or 'Private'}\nReason: {reason}"
        await log_to_channel(log_msg)
        
    except TelegramAPIError as e:
        logger.error(f"Failed to delete message: {e}")

    try:
        await warns_col.update_one(
            {"chat_id": chat_id, "user_id": user_id},
            {
                "$inc": {"warns": 1},
                "$set": {
                    "last_reason": str(reason),
                    "last_warn": time.time(),
                    "username": message.from_user.username if message.from_user else None
                }
            },
            upsert=True
        )
        
        doc = await warns_col.find_one({"chat_id": chat_id, "user_id": user_id})
        current_warns = doc.get("warns", 1) if doc else 1
        
        await log_col.insert_one({
            "chat_id": chat_id,
            "user_id": user_id,
            "message_id": message.message_id,
            "reason": str(reason),
            "timestamp": time.time(),
            "media_type": message.content_type,
            "warn_count": current_warns
        })
        
        try:
            user_name = message.from_user.full_name if message.from_user else "User"
            warning_msg = await message.answer(
                f"Warning: {user_name} has been warned ({current_warns}/{WARN_LIMIT}) for posting NSFW content."
            )
            await asyncio.sleep(10)
            await warning_msg.delete()
        except TelegramAPIError:
            pass

        if current_warns >= WARN_LIMIT:
            await mute_user(chat_id, user_id, message.from_user.full_name if message.from_user else "User")
            
    except PyMongoError as e:
        logger.error(f"Database error during moderation: {e}")

async def mute_user(chat_id: int, user_id: int, user_name: str):
    """Mute user"""
    try:
        until_date = int(time.time()) + MUTE_DURATION
        
        await bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=ChatPermissions(
                can_send_messages=False,
                can_send_media_messages=False,
                can_send_other_messages=False,
                can_add_web_page_previews=False
            ),
            until_date=until_date
        )
        
        mute_msg = await bot.send_message(
            chat_id, 
            f"{user_name} has been muted for {MUTE_DURATION // 60} minutes due to repeated NSFW violations."
        )
        
        logger.info(f"Muted user {user_id} in chat {chat_id}")
        
        await asyncio.sleep(15)
        await mute_msg.delete()
        
    except TelegramAPIError as e:
        logger.error(f"Failed to mute user {user_id}: {e}")

# ==================== COMMAND HANDLERS ====================

@dp.message(Command("start"))
async def start_cmd(msg: types.Message):
    welcome_text = (
        "Hello! I am an NSFW Content Moderation Bot.\n\n"
        "Features:\n"
        "- Auto-detect NSFW in photos, videos, stickers, GIFs\n"
        "- Warn users (3 warnings = mute)\n"
        "- Auto-mute repeat offenders\n"
        "- Global ban system\n\n"
        "Admin Commands:\n"
        "/warn_status - Check user warnings\n"
        "/warn_reset - Reset warnings\n"
        "/whitelist_add - Whitelist user\n\n"
        "Owner Commands:\n"
        "/stats - Bot statistics\n"
        "/gban - Global ban user\n"
        "/ungban - Remove global ban\n"
        "/gbanlist - List banned users\n"
        "/broadcast - Broadcast message\n\n"
        "Add me to your group to start moderating!"
    )
    
    await msg.answer(welcome_text)
    logger.info(f"/start command from {msg.from_user.id}")

@dp.message(Command("help"))
async def help_cmd(msg: types.Message):
    await start_cmd(msg)

@dp.message(Command("broadcast"))
async def broadcast_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        return
    
    if not msg.reply_to_message:
        await msg.reply("Please reply to a message to broadcast it.")
        return
    
    processing_msg = await msg.reply("Starting broadcast...")
    
    try:
        stats = await broadcast_message(msg.reply_to_message, forward=True)
        
        result_text = (
            f"Broadcast Completed!\n\n"
            f"Success: {stats['success']}\n"
            f"Failed: {stats['failed']}\n"
            f"Pinned: {stats['pinned']}"
        )
        
        await processing_msg.edit_text(result_text)
        await log_to_channel(f"Broadcast by {msg.from_user.full_name}\n{result_text}")
        
    except Exception as e:
        await processing_msg.edit_text(f"Broadcast failed: {str(e)}")

@dp.message(Command("gban"))
async def gban_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        return
    
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("Please reply to a user message to gban them.")
        return
    
    target_user = msg.reply_to_message.from_user
    reason = " ".join(msg.text.split()[1:]) or "No reason provided"
    
    processing_msg = await msg.reply(f"GBanning user {target_user.full_name}...")
    
    try:
        banned_count = await gban_user(target_user.id, reason, msg.from_user.id)
        
        result_text = (
            f"User {target_user.full_name} has been globally banned!\n"
            f"Reason: {reason}\n"
            f"Banned from: {banned_count} groups"
        )
        
        await processing_msg.edit_text(result_text)
        await log_to_channel(f"GBan by {msg.from_user.full_name}\n{result_text}")
        
    except Exception as e:
        await processing_msg.edit_text(f"GBan failed: {str(e)}")

@dp.message(Command("ungban"))
async def ungban_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        return
    
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("Please reply to a user message to ungban them.")
        return
    
    target_user = msg.reply_to_message.from_user
    
    processing_msg = await msg.reply(f"Removing gban from {target_user.full_name}...")
    
    try:
        success = await ungban_user(target_user.id)
        
        if success:
            result_text = f"Global ban removed from {target_user.full_name}"
            await processing_msg.edit_text(result_text)
            await log_to_channel(f"Ungban by {msg.from_user.full_name}\n{result_text}")
        else:
            await processing_msg.edit_text("User was not gbanned or operation failed")
            
    except Exception as e:
        await processing_msg.edit_text(f"Ungban failed: {str(e)}")

@dp.message(Command("stats"))
async def stats_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        return
    
    processing_msg = await msg.reply("Gathering statistics...")
    
    try:
        stats = await get_bot_stats()
        
        stats_text = (
            "Bot Statistics\n\n"
            f"Total Chats: {stats.get('total_chats', 0)}\n"
            f"Groups: {stats.get('total_groups', 0)}\n"
            f"Users: {stats.get('total_users', 0)}\n"
            f"GBanned Users: {stats.get('total_gbanned', 0)}\n"
            f"Total Warnings: {stats.get('total_warns', 0)}\n"
            f"Total Logs: {stats.get('total_logs', 0)}"
        )
        
        await processing_msg.edit_text(stats_text)
        
    except Exception as e:
        await processing_msg.edit_text(f"Failed to get stats: {str(e)}")

@dp.message(Command("gbanlist"))
async def gbanlist_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        return
    
    try:
        gbanned_users = []
        async for user in gban_col.find({"banned": True}).limit(50):
            user_info = f"- {user.get('first_name', 'Unknown')} (@{user.get('username', 'N/A')}) - {user.get('reason', 'No reason')}"
            gbanned_users.append(user_info)
        
        if gbanned_users:
            text = "GBanned Users:\n\n" + "\n".join(gbanned_users)
            if len(gbanned_users) == 50:
                text += "\n\nand more"
        else:
            text = "No users are currently gbanned."
            
        await msg.reply(text)
        
    except Exception as e:
        await msg.reply(f"Failed to get gbanlist: {str(e)}")

@dp.message(Command("warn_reset"))
async def reset_warns_cmd(msg: types.Message):
    if not await is_user_admin(msg.chat.id, msg.from_user.id):
        return
    
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("Please reply to a user message to reset their warnings.")
        return
    
    target_user = msg.reply_to_message.from_user
    try:
        await warns_col.delete_one({"chat_id": msg.chat.id, "user_id": target_user.id})
        response = await msg.reply(f"Warnings reset for {target_user.full_name}")
        await asyncio.sleep(10)
        await response.delete()
        await msg.delete()
    except PyMongoError as e:
        await msg.reply("Failed to reset warnings.")

@dp.message(Command("warn_status"))
async def warn_status_cmd(msg: types.Message):
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("Please reply to a user message to check their warnings.")
        return
    
    target_user = msg.reply_to_message.from_user
    try:
        doc = await warns_col.find_one({"chat_id": msg.chat.id, "user_id": target_user.id})
        warns = doc.get("warns", 0) if doc else 0
        await msg.reply(f"{target_user.full_name} has {warns}/{WARN_LIMIT} warnings")
    except PyMongoError as e:
        await msg.reply("Failed to check warning status.")

@dp.message(Command("whitelist_add"))
async def whitelist_add_cmd(msg: types.Message):
    if not await is_user_admin(msg.chat.id, msg.from_user.id):
        return
    
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("Please reply to a user message to whitelist them.")
        return
    
    target_user = msg.reply_to_message.from_user
    try:
        await whitelist_col.update_one(
            {"chat_id": msg.chat.id, "user_id": target_user.id},
            {"$set": {"username": target_user.username, "full_name": target_user.full_name}},
            upsert=True
        )
        response = await msg.reply(f"{target_user.full_name} added to whitelist")
        await asyncio.sleep(10)
        await response.delete()
        await msg.delete()
    except PyMongoError as e:
        await msg.reply("Failed to add user to whitelist.")

# ==================== MEDIA HANDLERS ====================

@dp.message(F.photo)
async def handle_photo(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    logger.info(f"Photo received from {msg.from_user.id if msg.from_user else 'unknown'} in chat {msg.chat.id}")
    
    try:
        async with temporary_download(msg.photo[-1], ".jpg") as file_path:
            if file_path:
                await handle_detect_and_action(msg, file_path, "photo")
    except Exception as e:
        logger.error(f"Error processing photo: {e}")

@dp.message(F.sticker)
async def handle_sticker(msg: types.Message):
    """Handle all sticker types - static, animated, video"""
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    logger.info(f"Sticker received from {msg.from_user.id if msg.from_user else 'unknown'} in chat {msg.chat.id}")
    logger.info(f"Sticker type: {msg.sticker.type}, is_animated: {msg.sticker.is_animated}, is_video: {msg.sticker.is_video}")
    
    try:
        # Determine correct extension based on sticker type
        if msg.sticker.is_video:
            # Video stickers are WEBM format
            ext = ".webm"
            logger.info("Video sticker detected - using .webm extension")
        elif msg.sticker.is_animated:
            # Animated stickers are TGS format (but Telegram sends as WEBP with animation)
            ext = ".webp"
            logger.info("Animated sticker detected - using .webp extension")
        else:
            # Static stickers are WEBP format
            ext = ".webp"
            logger.info("Static sticker detected - using .webp extension")
        
        async with temporary_download(msg.sticker, ext) as file_path:
            if file_path:
                await handle_detect_and_action(msg, file_path, f"sticker({msg.sticker.type})")
    except Exception as e:
        logger.error(f"Error processing sticker: {e}")

@dp.message(F.video)
async def handle_video(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    logger.info(f"Video received from {msg.from_user.id if msg.from_user else 'unknown'} in chat {msg.chat.id}")
    
    try:
        async with temporary_download(msg.video, ".mp4") as file_path:
            if file_path:
                await handle_detect_and_action(msg, file_path, "video")
    except Exception as e:
        logger.error(f"Error processing video: {e}")

@dp.message(F.animation)
async def handle_animation(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    logger.info(f"Animation/GIF received from {msg.from_user.id if msg.from_user else 'unknown'} in chat {msg.chat.id}")
    
    try:
        async with temporary_download(msg.animation, ".mp4") as file_path:
            if file_path:
                await handle_detect_and_action(msg, file_path, "animation")
    except Exception as e:
        logger.error(f"Error processing animation: {e}")

@dp.message(F.document)
async def handle_document(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    mime_type = msg.document.mime_type or ""
    file_name = msg.document.file_name or ""
    
    logger.info(f"Document received: {file_name} (MIME: {mime_type})")
    
    if mime_type.startswith(('image/', 'video/')) or any(file_name.lower().endswith(ext) for ext in 
        ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.mp4', '.mkv', '.webm', '.mov', '.avi', '.flv', '.m4v', '.3gp']):
        
        logger.info(f"Processing media document: {file_name}")
        
        try:
            if mime_type.startswith('image/'):
                ext = '.jpg'
            elif mime_type.startswith('video/'):
                ext = '.mp4'
            else:
                ext = ''
                
            async with temporary_download(msg.document, ext) as file_path:
                if file_path:
                    await handle_detect_and_action(msg, file_path, f"document({file_name})")
        except Exception as e:
            logger.error(f"Error processing document: {e}")

@dp.message(F.video_note)
async def handle_video_note(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    logger.info(f"Video note received from {msg.from_user.id if msg.from_user else 'unknown'}")
    
    try:
        async with temporary_download(msg.video_note, ".mp4") as file_path:
            if file_path:
                await handle_detect_and_action(msg, file_path, "video_note")
    except Exception as e:
        logger.error(f"Error processing video note: {e}")

@dp.message(F.text)
async def handle_text(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)

# ==================== DATABASE SETUP ====================

async def setup_database():
    try:
        await warns_col.create_index([("chat_id", 1), ("user_id", 1)], unique=True)
        await warns_col.create_index([("last_warn", 1)], expireAfterSeconds=30*24*3600)
        await log_col.create_index([("chat_id", 1), ("timestamp", -1)])
        await whitelist_col.create_index([("chat_id", 1), ("user_id", 1)], unique=True)
        await gban_col.create_index([("user_id", 1)], unique=True)
        await chats_col.create_index([("chat_id", 1)], unique=True)
        await users_col.create_index([("user_id", 1)], unique=True)
        logger.info("Database indexes created successfully")
    except PyMongoError as e:
        logger.error(f"Failed to create database indexes: {e}")

async def on_startup(bot):
    logger.info("Starting moderation bot...")
    await setup_database()
    await log_to_channel("Bot Started Successfully!")
    logger.info("Bot started successfully")

async def on_shutdown(bot):
    logger.info("Shutting down moderation bot...")
    await log_to_channel("Bot Shutting Down...")
    mongo.close()
    logger.info("Bot shutdown complete")

async def main():
    Path("logs").mkdir(exist_ok=True)
    
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
