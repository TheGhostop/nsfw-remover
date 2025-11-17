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
from aiogram.types import ChatPermissions, InlineKeyboardMarkup, InlineKeyboardButton
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
WARN_LIMIT = 5  # CHANGED: 3 -> 5
MAX_VIDEO_FRAMES = 6
VIDEO_FPS = 1
MUTE_DURATION = 1800

# Thresholds
NSFW_MIN_SCORE = 4.0
NSFW_SKIN_THRESHOLD = 0.35
NSFW_FLESH_THRESHOLD = 0.35

# Image URL for welcome message
WELCOME_IMAGE_URL = "https://i.ibb.co/KzK6R4zW/IMG-20251117-212221-098.jpg"

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

def detect_nsfw_improved(image_path: str) -> Tuple[bool, Optional[dict]]:
    """Improved NSFW detection"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return False, None
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 15, 60], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        
        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = cv2.countNonZero(mask_skin)
        skin_percentage = skin_pixels / mask_skin.size
        
        logger.info(f"Skin pixels: {skin_percentage:.2%}")
        
        b, g, r = cv2.split(img)
        
        flesh_mask = cv2.inRange(r, 95, 220) & \
                    cv2.inRange(g, 40, 220) & \
                    cv2.inRange(b, 20, 220)
        
        flesh_mask2 = cv2.inRange(r, 95, 220) & \
                     cv2.inRange(g, 40, 200) & \
                     cv2.inRange(b, 20, 100)
        
        combined_flesh = cv2.bitwise_or(flesh_mask, flesh_mask2)
        flesh_pixels = cv2.countNonZero(combined_flesh)
        flesh_percentage = flesh_pixels / combined_flesh.size
        
        logger.info(f"Flesh tone pixels: {flesh_percentage:.2%}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_pixels = cv2.countNonZero(edges)
        edge_density = edge_pixels / edges.size
        
        logger.info(f"Edge density: {edge_density:.2%}")
        
        saturation = hsv[:, :, 1]
        low_sat_pixels = np.sum(saturation < 100)
        low_sat_percentage = low_sat_pixels / saturation.size
        
        logger.info(f"Low saturation (skin-like): {low_sat_percentage:.2%}")
        
        nsfw_score = 0
        reasons = []
        
        if skin_percentage >= NSFW_SKIN_THRESHOLD and flesh_percentage >= NSFW_FLESH_THRESHOLD:
            nsfw_score += 3
            reasons.append(f"High skin ({skin_percentage:.1%}) AND flesh ({flesh_percentage:.1%})")
        
        if skin_percentage >= 0.85:
            nsfw_score += 2.5
            reasons.append(f"Extreme skin tone: {skin_percentage:.1%}")
        
        elif skin_percentage >= 0.50 and flesh_percentage >= 0.50:
            nsfw_score += 2
            reasons.append(f"High skin ({skin_percentage:.1%}) + flesh ({flesh_percentage:.1%})")
        
        if skin_percentage >= 0.70:
            nsfw_score += 1.5
            reasons.append(f"Very high skin coverage: {skin_percentage:.1%}")
        
        if flesh_percentage > 0.50:
            nsfw_score += 1
            reasons.append(f"Flesh-heavy image: {flesh_percentage:.1%}")
        
        if skin_percentage > 0.30 and edge_density > 0.15:
            nsfw_score += 1.5
            reasons.append(f"Skin with defined edges (curves): {edge_density:.1%}")
        
        if skin_percentage > 0.40 and low_sat_percentage > 0.6:
            nsfw_score += 1.5
            reasons.append(f"Skin + low saturation (natural skin): {low_sat_percentage:.1%}")
        
        logger.info(f"NSFW Score: {nsfw_score:.1f}")
        logger.info(f"Reasons: {reasons}")
        
        if nsfw_score >= NSFW_MIN_SCORE:
            return True, {
                "method": "improved_detection",
                "score": nsfw_score,
                "skin_percentage": skin_percentage,
                "flesh_percentage": flesh_percentage,
                "reasons": reasons
            }
        
        return False, None
        
    except Exception as e:
        logger.error(f"NSFW detection failed: {e}")
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
            except:
                pass

def convert_webp_to_jpg_pil(webp_path: str) -> str:
    """Convert WEBP to JPG"""
    jpg_path = webp_path.replace('.webp', '.jpg')
    try:
        with Image.open(webp_path) as im:
            if im.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[-1] if im.mode in ('RGBA', 'LA') else None)
                background.save(jpg_path, "JPEG", quality=95)
            else:
                im.convert("RGB").save(jpg_path, "JPEG", quality=95)
        return jpg_path
    except:
        return webp_path

async def extract_frames(video_path: str, out_dir: str, fps: int = 1, max_frames: int = 6) -> List[str]:
    """Extract frames from video"""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-vf", f"fps={fps}",
        "-q:v", "3", "-frames:v", str(max_frames),
        os.path.join(out_dir, "frame_%03d.jpg")
    ]
    try:
        subprocess.run(cmd, check=True, timeout=60)
        frames = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg')])
        return frames[:max_frames]
    except:
        return []

async def analyze_image(image_path: str) -> Tuple[bool, Optional[dict]]:
    """Analyze single image"""
    if not image_path or not os.path.exists(image_path):
        return False, None
    
    try:
        return detect_nsfw_improved(image_path)
    except:
        return False, None

async def analyze_images(paths: List[str]) -> Tuple[bool, Optional[dict]]:
    """Analyze multiple images"""
    if not paths:
        return False, None
    
    try:
        logger.info(f"Analyzing {len(paths)} image(s)...")
        
        for idx, path in enumerate(paths, 1):
            is_nsfw, result = await analyze_image(path)
            if is_nsfw:
                logger.warning(f"NSFW DETECTED in frame {idx}!")
                return True, result
        
        logger.info(f"All {len(paths)} frame(s) analyzed - SAFE")
        return False, None
    except:
        return False, None

async def is_user_whitelisted(chat_id: int, user_id: int) -> bool:
    try:
        doc = await whitelist_col.find_one({"chat_id": chat_id, "user_id": user_id})
        return doc is not None
    except:
        return False

async def is_user_admin(chat_id: int, user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        return member.status in ["creator", "administrator"]
    except:
        return False

async def is_owner(user_id: int) -> bool:
    return user_id in OWNER_IDS

async def log_to_channel(message: str):
    try:
        await bot.send_message(LOGGER_GROUP_ID, message, parse_mode="HTML")
    except:
        pass

async def update_chat_data(chat_id: int, chat_title: str = ""):
    try:
        await chats_col.update_one(
            {"chat_id": chat_id},
            {"$set": {"chat_title": chat_title, "last_seen": time.time(), "is_group": chat_id < 0}},
            upsert=True
        )
    except:
        pass

async def update_user_data(user_id: int, username: str = "", first_name: str = ""):
    try:
        await users_col.update_one(
            {"user_id": user_id},
            {"$set": {"username": username, "first_name": first_name, "last_seen": time.time()}},
            upsert=True
        )
    except:
        pass

async def is_user_gbanned(user_id: int) -> bool:
    try:
        doc = await gban_col.find_one({"user_id": user_id, "banned": True})
        return doc is not None
    except:
        return False

async def gban_user(user_id: int, reason: str = "No reason", banned_by: int = 0):
    try:
        user_info = await users_col.find_one({"user_id": user_id})
        username = user_info.get("username", "") if user_info else ""
        first_name = user_info.get("first_name", "") if user_info else ""
        
        await gban_col.update_one(
            {"user_id": user_id},
            {"$set": {"username": username, "first_name": first_name, "reason": reason, 
                     "banned_by": banned_by, "banned_at": time.time(), "banned": True}},
            upsert=True
        )
        
        banned_count = 0
        async for chat in chats_col.find({"is_group": True}):
            try:
                await bot.ban_chat_member(chat["chat_id"], user_id)
                banned_count += 1
            except:
                pass
        
        return banned_count
    except:
        return 0

async def ungban_user(user_id: int) -> bool:
    try:
        result = await gban_col.update_one(
            {"user_id": user_id},
            {"$set": {"banned": False}}
        )
        return result.modified_count > 0
    except:
        return False

async def get_bot_stats() -> Dict[str, Any]:
    try:
        return {
            "total_chats": await chats_col.count_documents({}),
            "total_groups": await chats_col.count_documents({"is_group": True}),
            "total_users": await users_col.count_documents({}),
            "total_gbanned": await gban_col.count_documents({"banned": True}),
            "total_warns": await warns_col.count_documents({}),
            "total_logs": await log_col.count_documents({})
        }
    except:
        return {}

async def handle_detect_and_action(message: types.Message, file_path: str, media_type: str = "media"):
    """Analyze media and delete if NSFW"""
    user_id = message.from_user.id if message.from_user else None
    chat_id = message.chat.id
    
    if not file_path:
        return
    
    logger.info(f"Starting analysis: msg_id={message.message_id}, user_id={user_id}")
    
    if user_id and await is_user_gbanned(user_id):
        try:
            await message.delete()
        except:
            pass
        return
    
    if user_id:
        if await is_user_admin(chat_id, user_id) or await is_user_whitelisted(chat_id, user_id):
            logger.info(f"Skipping admin/whitelisted: {user_id}")
            return

    is_nsfw = False
    reason = None
    temp_files_to_cleanup = []

    try:
        file_lower = file_path.lower()
        
        if file_lower.endswith((".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".3gp")):
            with tempfile.TemporaryDirectory() as tmpdir:
                frames = await extract_frames(file_path, tmpdir, fps=VIDEO_FPS, max_frames=MAX_VIDEO_FRAMES)
                if frames:
                    is_nsfw, reason = await analyze_images(frames)
                    
        elif file_lower.endswith(".webp"):
            jpg_path = convert_webp_to_jpg_pil(file_path)
            if jpg_path != file_path and os.path.exists(jpg_path):
                temp_files_to_cleanup.append(jpg_path)
                is_nsfw, reason = await analyze_images([jpg_path])
            else:
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
                except:
                    pass
            
        elif file_lower.endswith((".gif", ".gifv")):
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-i", file_path, "-vf", f"fps={VIDEO_FPS}",
                    "-q:v", "3", "-frames:v", str(MAX_VIDEO_FRAMES),
                    os.path.join(tmpdir, "frame_%03d.jpg")
                ]
                try:
                    subprocess.run(cmd, check=True, timeout=30)
                    frames = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith('.jpg')])
                    if frames:
                        is_nsfw, reason = await analyze_images(frames)
                except:
                    is_nsfw, reason = await analyze_images([file_path])
                    
        else:
            is_nsfw, reason = await analyze_images([file_path])

        if is_nsfw:
            logger.warning(f"ðŸš¨ NSFW DETECTED! msg_id={message.message_id}, user_id={user_id}, score={reason.get('score')}")
            await take_moderation_action(message, reason, user_id, chat_id)
        else:
            logger.info(f"âœ… Content is safe")
            
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
    finally:
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

async def take_moderation_action(message: types.Message, reason: dict, user_id: int, chat_id: int):
    """Delete NSFW message and warn user"""
    
    # STEP 1: DELETE MESSAGE
    try:
        logger.info(f"ðŸ—‘ï¸ ATTEMPTING DELETE: msg_id={message.message_id}")
        await message.delete()
        logger.info(f"âœ… SUCCESS: Deleted message {message.message_id}")
    except Exception as e:
        logger.error(f"âŒ Delete failed: {e}")

    # STEP 2: WARN USER
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
        
        logger.info(f"âš ï¸ Warning #{current_warns} given to {user_id}")
        
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
            warning_msg = await bot.send_message(
                chat_id,
                f"âš ï¸ Warning: {user_name} warned ({current_warns}/{WARN_LIMIT}) for NSFW!"
            )
            
            await asyncio.sleep(10)
            try:
                await warning_msg.delete()
            except:
                pass
        except:
            pass

        if current_warns >= WARN_LIMIT:
            await mute_user(chat_id, user_id, message.from_user.full_name if message.from_user else "User")
            
    except Exception as e:
        logger.error(f"âš ï¸ Warning failed: {e}")

async def mute_user(chat_id: int, user_id: int, user_name: str):
    """Mute user with unmute button"""
    try:
        logger.info(f"ðŸ”‡ MUTING {user_id} in {chat_id}")
        
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
        
        # Create inline button for unmute
        unmute_button = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="ðŸ”Š Unmute User", callback_data=f"unmute_{user_id}_{chat_id}")]
        ])
        
        mute_msg = await bot.send_message(
            chat_id, 
            f"ðŸ”‡ {user_name} muted for {MUTE_DURATION // 60} min (NSFW violations)\n\nâ±ï¸ Will auto-unmute after timeout.",
            reply_markup=unmute_button
        )
        
        logger.info(f"âœ… User {user_id} muted with unmute button")
        
    except Exception as e:
        logger.error(f"ðŸ”‡ Mute failed: {e}")

# ==================== CALLBACK HANDLERS ====================

@dp.callback_query(lambda query: query.data.startswith("unmute_"))
async def unmute_callback(query: types.CallbackQuery):
    """Handle unmute button"""
    try:
        data_parts = query.data.split("_")
        user_id = int(data_parts[1])
        chat_id = int(data_parts[2])
        
        # Check if callback user is admin
        if not await is_user_admin(chat_id, query.from_user.id):
            await query.answer("âŒ Only admins can unmute!", show_alert=True)
            return
        
        # Unmute user
        await bot.restrict_chat_member(
            chat_id=chat_id,
            user_id=user_id,
            permissions=ChatPermissions(
                can_send_messages=True,
                can_send_media_messages=True,
                can_send_other_messages=True,
                can_add_web_page_previews=True
            )
        )
        
        await query.answer("âœ… User unmuted!", show_alert=False)
        
        # Edit message
        user = await bot.get_chat_member(chat_id, user_id)
        await query.message.edit_text(
            f"âœ… {user.user.full_name} unmuted by {query.from_user.full_name}"
        )
        
        logger.info(f"ðŸ”Š User {user_id} unmuted by {query.from_user.id}")
        
    except Exception as e:
        logger.error(f"Unmute error: {e}")
        await query.answer(f"âŒ Error: {str(e)}", show_alert=True)

# ==================== COMMANDS ====================

@dp.message(Command("start"))
async def start_cmd(msg: types.Message):
    """Start command with image and add button"""
    
    # Create add button
    add_button = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="âž• Add to Group", url="https://t.me/NFSW_Protectionbot?startgroup=true")]
    ])
    
    # Send with image
    try:
        await bot.send_photo(
            msg.chat.id,
            photo=WELCOME_IMAGE_URL,
            caption="ðŸ›¡ï¸ **NSFW Content Moderation Bot**\n\n"
                   "âœ… Auto-detects & deletes NSFW\n"
                   "âœ… Smart warning system (5 = mute)\n"
                   "âœ… Global ban system\n"
                   "âœ… Admin controls\n\n"
                   "ðŸš€ Click below to add me to your group!",
            parse_mode="Markdown",
            reply_markup=add_button
        )
    except Exception as e:
        logger.error(f"Photo send failed: {e}")
        # Fallback to text
        await msg.answer(
            "ðŸ›¡ï¸ **NSFW Content Moderation Bot**\n\n"
            "âœ… Auto-detects & deletes NSFW\n"
            "âœ… Smart warning system (5 = mute)\n"
            "âœ… Global ban system\n"
            "âœ… Admin controls\n\n"
            "ðŸš€ Click below to add me to your group!",
            parse_mode="Markdown",
            reply_markup=add_button
        )
    
    logger.info(f"ðŸš€ /start from {msg.from_user.id}")

@dp.message(Command("help"))
async def help_cmd(msg: types.Message):
    await msg.answer(
        "ðŸ“‹ **Available Commands:**\n\n"
        "/start - Show welcome message\n"
        "/stats - Bot statistics (owner)\n"
        "/warn_status - Check warnings\n"
        "/warn_reset - Reset warnings (admin)\n"
        "/whitelist_add - Whitelist user (admin)\n"
        "/gban - Global ban (owner)\n"
        "/ungban - Remove ban (owner)\n\n"
        "ðŸ›¡ï¸ Protect your group now!",
        parse_mode="Markdown"
    )

@dp.message(Command("stats"))
async def stats_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        await msg.reply("âŒ Owner only!")
        return
    
    try:
        stats = await get_bot_stats()
        text = (
            "ðŸ“Š **Bot Statistics**\n\n"
            f"ðŸ’¬ Total Chats: {stats.get('total_chats', 0)}\n"
            f"ðŸ‘¥ Groups: {stats.get('total_groups', 0)}\n"
            f"ðŸ‘¤ Users: {stats.get('total_users', 0)}\n"
            f"ðŸš« GBanned: {stats.get('total_gbanned', 0)}\n"
            f"âš ï¸ Total Warns: {stats.get('total_warns', 0)}\n"
            f"ðŸ“ Total Logs: {stats.get('total_logs', 0)}"
        )
        await msg.reply(text, parse_mode="Markdown")
    except:
        await msg.reply("âŒ Error fetching stats")

@dp.message(Command("gban"))
async def gban_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        return
    
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("âŒ Reply to a user message to gban them")
        return
    
    try:
        target = msg.reply_to_message.from_user
        await gban_user(target.id, "NSFW ban", msg.from_user.id)
        await msg.reply(f"âœ… {target.full_name} globally banned")
    except:
        await msg.reply("âŒ Error")

@dp.message(Command("ungban"))
async def ungban_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        return
    
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("âŒ Reply to a user message")
        return
    
    try:
        target = msg.reply_to_message.from_user
        if await ungban_user(target.id):
            await msg.reply(f"âœ… {target.full_name} unbanned")
        else:
            await msg.reply("âŒ User not gbanned")
    except:
        await msg.reply("âŒ Error")

@dp.message(Command("warn_status"))
async def warn_status_cmd(msg: types.Message):
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("âŒ Reply to a user")
        return
    
    try:
        target = msg.reply_to_message.from_user
        doc = await warns_col.find_one({"chat_id": msg.chat.id, "user_id": target.id})
        warns = doc.get("warns", 0) if doc else 0
        await msg.reply(f"âš ï¸ {target.full_name}: {warns}/{WARN_LIMIT} warnings")
    except:
        pass

@dp.message(Command("warn_reset"))
async def warn_reset_cmd(msg: types.Message):
    if not await is_user_admin(msg.chat.id, msg.from_user.id):
        await msg.reply("âŒ Admin only!")
        return
    
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("âŒ Reply to a user")
        return
    
    try:
        target = msg.reply_to_message.from_user
        await warns_col.delete_one({"chat_id": msg.chat.id, "user_id": target.id})
        await msg.reply(f"âœ… Warnings reset for {target.full_name}")
    except:
        pass

@dp.message(Command("whitelist_add"))
async def whitelist_add_cmd(msg: types.Message):
    if not await is_user_admin(msg.chat.id, msg.from_user.id):
        await msg.reply("âŒ Admin only!")
        return
    
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("âŒ Reply to a user")
        return
    
    try:
        target = msg.reply_to_message.from_user
        await whitelist_col.update_one(
            {"chat_id": msg.chat.id, "user_id": target.id},
            {"$set": {"username": target.username, "full_name": target.full_name}},
            upsert=True
        )
        await msg.reply(f"âœ… {target.full_name} whitelisted")
    except:
        pass

# ==================== MEDIA HANDLERS ====================

@dp.message(F.photo)
async def handle_photo(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    logger.info(f"ðŸ“· Photo from {msg.from_user.id}")
    try:
        async with temporary_download(msg.photo[-1], ".jpg") as file_path:
            if file_path:
                await handle_detect_and_action(msg, file_path, "photo")
    except Exception as e:
        logger.error(f"Photo error: {e}")

@dp.message(F.sticker)
async def handle_sticker(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    logger.info(f"ðŸŽ¨ Sticker")
    try:
        ext = ".webm" if msg.sticker.is_video else ".webp"
        async with temporary_download(msg.sticker, ext) as file_path:
            if file_path:
                await handle_detect_and_action(msg, file_path, "sticker")
    except Exception as e:
        logger.error(f"Sticker error: {e}")

@dp.message(F.video)
async def handle_video(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    logger.info(f"ðŸŽ¬ Video")
    try:
        async with temporary_download(msg.video, ".mp4") as file_path:
            if file_path:
                await handle_detect_and_action(msg, file_path, "video")
    except Exception as e:
        logger.error(f"Video error: {e}")

@dp.message(F.animation)
async def handle_animation(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    logger.info(f"ðŸŽžï¸ GIF")
    try:
        async with temporary_download(msg.animation, ".mp4") as file_path:
            if file_path:
                await handle_detect_and_action(msg, file_path, "animation")
    except Exception as e:
        logger.error(f"GIF error: {e}")

@dp.message(F.document)
async def handle_document(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    mime = msg.document.mime_type or ""
    name = msg.document.file_name or ""
    
    if mime.startswith(('image/', 'video/')) or any(name.lower().endswith(x) for x in 
        ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.mp4', '.mkv', '.webm']):
        
        logger.info(f"ðŸ“„ Document: {name}")
        try:
            ext = '.jpg' if mime.startswith('image/') else '.mp4'
            async with temporary_download(msg.document, ext) as file_path:
                if file_path:
                    await handle_detect_and_action(msg, file_path, "document")
        except Exception as e:
            logger.error(f"Document error: {e}")

@dp.message(F.video_note)
async def handle_video_note(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)
    
    logger.info(f"ðŸŽ¥ Video note")
    try:
        async with temporary_download(msg.video_note, ".mp4") as file_path:
            if file_path:
                await handle_detect_and_action(msg, file_path, "video_note")
    except Exception as e:
        logger.error(f"Video note error: {e}")

@dp.message(F.text)
async def handle_text(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)

# ==================== DATABASE ====================

async def setup_database():
    try:
        await warns_col.create_index([("chat_id", 1), ("user_id", 1)], unique=True)
        await log_col.create_index([("chat_id", 1), ("timestamp", -1)])
        await whitelist_col.create_index([("chat_id", 1), ("user_id", 1)], unique=True)
        await gban_col.create_index([("user_id", 1)], unique=True)
        await chats_col.create_index([("chat_id", 1)], unique=True)
        await users_col.create_index([("user_id", 1)], unique=True)
        logger.info("âœ… Database ready")
    except:
        pass

async def on_startup(bot_instance):
    logger.info("ðŸš€ Bot starting...")
    await setup_database()
    await log_to_channel("âœ… NSFW Bot Started!")
    logger.info("âœ… Ready!")

async def on_shutdown(bot_instance):
    logger.info("ðŸ›‘ Shutting down...")
    await log_to_channel("â›” NSFW Bot Offline")
    mongo.close()

async def main():
    Path("logs").mkdir(exist_ok=True)
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
