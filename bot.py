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
WARN_LIMIT = 5
MAX_VIDEO_FRAMES = 6
VIDEO_FPS = 1
MUTE_DURATION = 1800

# Thresholds - INCREASED TO REDUCE FALSE POSITIVES
NSFW_MIN_SCORE = 6.0  # Increased from 4.0 to 6.0
NSFW_SKIN_THRESHOLD = 0.45  # Increased from 0.35
NSFW_FLESH_THRESHOLD = 0.45  # Increased from 0.35

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
    """Improved NSFW detection with better false positive prevention"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return False, None

        # Get image dimensions
        height, width = img.shape[:2]
        total_pixels = height * width
        
        # Skip very small images (like stickers)
        if total_pixels < 10000:  # Very small images are usually safe
            logger.info(f"Small image detected ({total_pixels} pixels), likely safe")
            return False, None

        # Enhanced face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_area = 0
        for (x, y, w, h) in faces:
            face_area += w * h
        
        face_ratio = face_area / total_pixels if total_pixels > 0 else 0
        
        # If significant face area detected, apply much stricter rules
        if face_ratio > 0.15:  # If more than 15% of image is faces
            logger.info(f"Significant face area detected ({face_ratio:.2%}), applying strict rules")
            NSFW_MIN_SCORE_TEMP = 8.0  # Very high threshold for face images
        elif face_ratio > 0.05:  # If some faces detected
            logger.info(f"Face detected ({face_ratio:.2%}), applying lenient rules")
            NSFW_MIN_SCORE_TEMP = 7.0  # Higher threshold
        else:
            NSFW_MIN_SCORE_TEMP = NSFW_MIN_SCORE

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Skin color detection in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_CLOSE, kernel)
        
        skin_pixels = cv2.countNonZero(mask_skin)
        skin_percentage = skin_pixels / mask_skin.size
        logger.info(f"Skin pixels: {skin_percentage:.2%}")

        # Flesh tone detection in RGB (more accurate)
        b, g, r = cv2.split(img)
        
        # More specific flesh tone ranges to avoid false positives
        flesh_mask = (
            (r >= 100) & (r <= 200) &
            (g >= 50) & (g <= 180) &
            (b >= 30) & (b <= 150) &
            (r > g) & (g > b)  # Skin typically has R > G > B
        ).astype(np.uint8) * 255
        
        flesh_pixels = cv2.countNonZero(flesh_mask)
        flesh_percentage = flesh_pixels / flesh_mask.size
        logger.info(f"Flesh tone pixels: {flesh_percentage:.2%}")

        # Edge detection for shape analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        edge_density = edge_pixels / edges.size
        logger.info(f"Edge density: {edge_density:.2%}")

        # Color analysis
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)
        logger.info(f"Average saturation: {avg_saturation:.1f}")

        # Brightness analysis
        value_channel = hsv[:, :, 2]
        avg_brightness = np.mean(value_channel)
        logger.info(f"Average brightness: {avg_brightness:.1f}")

        nsfw_score = 0
        reasons = []

        # STRICTER SCORING SYSTEM TO REDUCE FALSE POSITIVES
        
        # Only penalize if both skin AND flesh are high
        if skin_percentage >= 0.50 and flesh_percentage >= 0.50:
            nsfw_score += 3.0
            reasons.append(f"High skin ({skin_percentage:.1%}) AND flesh ({flesh_percentage:.1%})")
        elif skin_percentage >= 0.40 and flesh_percentage >= 0.40:
            nsfw_score += 2.0
            reasons.append(f"Moderate skin ({skin_percentage:.1%}) + flesh ({flesh_percentage:.1%})")

        # Extreme skin coverage
        if skin_percentage >= 0.80:
            nsfw_score += 2.5
            reasons.append(f"Extreme skin: {skin_percentage:.1%}")
        elif skin_percentage >= 0.65:
            nsfw_score += 1.5
            reasons.append(f"Very high skin: {skin_percentage:.1%}")

        # Flesh-heavy with moderate skin
        if flesh_percentage > 0.60 and skin_percentage > 0.30:
            nsfw_score += 1.5
            reasons.append(f"Flesh-heavy with skin: {flesh_percentage:.1%}")

        # Skin with defined edges (potential curves)
        if skin_percentage > 0.35 and edge_density > 0.20:
            nsfw_score += 1.5
            reasons.append(f"Skin with edges: {edge_density:.1%}")

        # Natural skin detection (low saturation + skin)
        if skin_percentage > 0.40 and avg_saturation < 80:
            nsfw_score += 1.0
            reasons.append(f"Natural skin (low sat): {avg_saturation:.1f}")

        # DEDUCTIONS FOR LIKELY SAFE CONTENT
        
        # High brightness often indicates safe content
        if avg_brightness > 180:
            nsfw_score -= 1.0
            reasons.append(f"High brightness safe: {avg_brightness:.1f}")
        
        # High saturation often indicates cartoons/stickers
        if avg_saturation > 150:
            nsfw_score -= 1.5
            reasons.append(f"High saturation safe: {avg_saturation:.1f}")
            
        # Very low skin percentage is safe
        if skin_percentage < 0.10:
            nsfw_score -= 2.0
            reasons.append(f"Low skin safe: {skin_percentage:.1%}")

        # Ensure score doesn't go negative
        nsfw_score = max(0, nsfw_score)

        logger.info(f"NSFW Score: {nsfw_score:.1f} (Threshold: {NSFW_MIN_SCORE_TEMP})")
        logger.info(f"Reasons: {reasons}")

        if nsfw_score >= NSFW_MIN_SCORE_TEMP:
            return True, {
                "method": "improved_detection",
                "score": nsfw_score,
                "skin_percentage": skin_percentage,
                "flesh_percentage": flesh_percentage,
                "edge_density": edge_density,
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
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "3",
        "-frames:v", str(max_frames),
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
            if not os.path.exists(path):
                logger.warning(f"Could not read image: {path}")
                continue

            is_nsfw, result = await analyze_image(path)
            if is_nsfw:
                logger.warning(f"🚨 NSFW DETECTED in frame {idx}!")
                return True, result  # Return immediately if NSFW found

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
            {"$set": {
                "username": username,
                "first_name": first_name,
                "reason": reason,
                "banned_by": banned_by,
                "banned_at": time.time(),
                "banned": True
            }},
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
        return{}

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

        # Video files (including .webm video stickers)
        if file_lower.endswith((".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".3gp")):
            with tempfile.TemporaryDirectory() as tmpdir:
                frames = await extract_frames(file_path, tmpdir, fps=VIDEO_FPS, max_frames=MAX_VIDEO_FRAMES)
                if frames:
                    is_nsfw, reason = await analyze_images(frames)
                else:
                    # If frame extraction fails, try direct analysis
                    logger.warning("Frame extraction failed, trying direct analysis")
                    is_nsfw, reason = await analyze_images([file_path])

        # WEBP images
        elif file_lower.endswith(".webp"):
            jpg_path = convert_webp_to_jpg_pil(file_path)
            if jpg_path != file_path and os.path.exists(jpg_path):
                temp_files_to_cleanup.append(jpg_path)
                is_nsfw, reason = await analyze_images([jpg_path])
            else:
                is_nsfw, reason = await analyze_images([file_path])

        # GIF files
        elif file_lower.endswith((".gif", ".gifv")):
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
                    else:
                        is_nsfw, reason = await analyze_images([file_path])
                except:
                    is_nsfw, reason = await analyze_images([file_path])

        # Other image formats
        else:
            is_nsfw, reason = await analyze_images([file_path])

        if is_nsfw:
            logger.warning(f"🚨 NSFW DETECTED! msg_id={message.message_id}, user_id={user_id}, score={reason.get('score')}")
            await take_moderation_action(message, reason, user_id, chat_id)
        else:
            logger.info(f"✅ Content is safe")

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
        logger.info(f"🗑️ ATTEMPTING DELETE: msg_id={message.message_id}")
        await message.delete()
        logger.info(f"✅ SUCCESS: Deleted message {message.message_id}")
    except Exception as e:
        logger.error(f"❌ Delete failed: {e}")

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
        logger.info(f"⚠️ Warning #{current_warns} given to {user_id}")

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
            # Create warning message with control buttons
            control_buttons = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="➖ Remove Warn", callback_data=f"remove_warn_{user_id}_{chat_id}"),
                    InlineKeyboardButton(text="➕ Add Warn", callback_data=f"add_warn_{user_id}_{chat_id}")
                ],
                [
                    InlineKeyboardButton(text="🔊 Unmute User", callback_data=f"unmute_{user_id}_{chat_id}")
                ]
            ])
            
            warning_msg = await bot.send_message(
                chat_id,
                f"<b><i>⚠️ Warning:</i></b> <b><i>{user_name}</i></b> <b><i>warned ({current_warns}/{WARN_LIMIT}) for NSFW!</i></b>",
                reply_markup=control_buttons,
                parse_mode="HTML"
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
        logger.error(f"⚠️ Warning failed: {e}")

async def mute_user(chat_id: int, user_id: int, user_name: str):
    """Mute user with control buttons"""
    try:
        logger.info(f"🔇 MUTING {user_id} in {chat_id}")
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

        # Create control buttons for mute message
        control_buttons = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="➖ Remove Warn", callback_data=f"remove_warn_{user_id}_{chat_id}"),
                InlineKeyboardButton(text="➕ Add Warn", callback_data=f"add_warn_{user_id}_{chat_id}")
            ],
            [
                InlineKeyboardButton(text="🔊 Unmute User", callback_data=f"unmute_{user_id}_{chat_id}")
            ]
        ])

        mute_msg = await bot.send_message(
            chat_id,
            f"<b><i>🔇 {user_name} muted for {MUTE_DURATION // 60} min (NSFW violations)</i></b>\n\n<b><i>⏰ Will auto-unmute after timeout.</i></b>",
            reply_markup=control_buttons,
            parse_mode="HTML"
        )
        logger.info(f"✅ User {user_id} muted with control buttons")

    except Exception as e:
        logger.error(f"🔇 Mute failed: {e}")

# ==================== CALLBACK HANDLERS ====================

@dp.callback_query(lambda query: query.data.startswith("unmute_"))
async def unmute_callback(query: types.CallbackQuery):
    """Handle unmute button - also resets warnings"""
    try:
        data_parts = query.data.split("_")
        user_id = int(data_parts[1])
        chat_id = int(data_parts[2])

        # Check if callback user is admin
        if not await is_user_admin(chat_id, query.from_user.id):
            await query.answer("❌ Only admins can unmute!", show_alert=True)
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

        # Reset warnings when unmuted
        await warns_col.delete_one({"chat_id": chat_id, "user_id": user_id})

        await query.answer("✅ User unmuted and warnings reset!", show_alert=False)

        # Edit message
        user = await bot.get_chat_member(chat_id, user_id)
        await query.message.edit_text(
            f"<b><i>✅ {user.user.full_name} unmuted by {query.from_user.full_name}</i></b>\n<b><i>⚠️ Warnings have been reset.</i></b>",
            parse_mode="HTML"
        )
        logger.info(f"🔊 User {user_id} unmuted and warnings reset by {query.from_user.id}")

    except Exception as e:
        logger.error(f"Unmute error: {e}")
        await query.answer(f"❌ Error: {str(e)}", show_alert=True)

@dp.callback_query(lambda query: query.data.startswith("remove_warn_"))
async def remove_warn_callback(query: types.CallbackQuery):
    """Handle remove warning button"""
    try:
        data_parts = query.data.split("_")
        user_id = int(data_parts[2])
        chat_id = int(data_parts[3])

        # Check if callback user is admin
        if not await is_user_admin(chat_id, query.from_user.id):
            await query.answer("❌ Only admins can remove warnings!", show_alert=True)
            return

        # Get current warnings
        doc = await warns_col.find_one({"chat_id": chat_id, "user_id": user_id})
        current_warns = doc.get("warns", 0) if doc else 0

        if current_warns > 0:
            new_warns = max(0, current_warns - 1)
            if new_warns == 0:
                await warns_col.delete_one({"chat_id": chat_id, "user_id": user_id})
            else:
                await warns_col.update_one(
                    {"chat_id": chat_id, "user_id": user_id},
                    {"$set": {"warns": new_warns}}
                )

            user = await bot.get_chat_member(chat_id, user_id)
            await query.answer(f"✅ Warning removed! Current: {new_warns}/{WARN_LIMIT}", show_alert=False)
            
            # Update message with new warn count
            control_buttons = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="➖ Remove Warn", callback_data=f"remove_warn_{user_id}_{chat_id}"),
                    InlineKeyboardButton(text="➕ Add Warn", callback_data=f"add_warn_{user_id}_{chat_id}")
                ],
                [
                    InlineKeyboardButton(text="🔊 Unmute User", callback_data=f"unmute_{user_id}_{chat_id}")
                ]
            ])
            
            await query.message.edit_text(
                f"<b><i>⚠️ {user.user.full_name}: {new_warns}/{WARN_LIMIT} warnings</i></b>",
                reply_markup=control_buttons,
                parse_mode="HTML"
            )
        else:
            await query.answer("❌ No warnings to remove!", show_alert=True)

    except Exception as e:
        logger.error(f"Remove warn error: {e}")
        await query.answer(f"❌ Error: {str(e)}", show_alert=True)

@dp.callback_query(lambda query: query.data.startswith("add_warn_"))
async def add_warn_callback(query: types.CallbackQuery):
    """Handle add warning button"""
    try:
        data_parts = query.data.split("_")
        user_id = int(data_parts[2])
        chat_id = int(data_parts[3])

        # Check if callback user is admin
        if not await is_user_admin(chat_id, query.from_user.id):
            await query.answer("❌ Only admins can add warnings!", show_alert=True)
            return

        # Get current warnings
        doc = await warns_col.find_one({"chat_id": chat_id, "user_id": user_id})
        current_warns = doc.get("warns", 0) if doc else 0

        new_warns = current_warns + 1
        
        await warns_col.update_one(
            {"chat_id": chat_id, "user_id": user_id},
            {"$set": {"warns": new_warns}},
            upsert=True
        )

        user = await bot.get_chat_member(chat_id, user_id)
        await query.answer(f"✅ Warning added! Current: {new_warns}/{WARN_LIMIT}", show_alert=False)

        # Check if should mute
        if new_warns >= WARN_LIMIT:
            await mute_user(chat_id, user_id, user.user.full_name)

        # Update message with new warn count
        control_buttons = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="➖ Remove Warn", callback_data=f"remove_warn_{user_id}_{chat_id}"),
                InlineKeyboardButton(text="➕ Add Warn", callback_data=f"add_warn_{user_id}_{chat_id}")
            ],
            [
                InlineKeyboardButton(text="🔊 Unmute User", callback_data=f"unmute_{user_id}_{chat_id}")
            ]
        ])
        
        await query.message.edit_text(
            f"<b><i>⚠️ {user.user.full_name}: {new_warns}/{WARN_LIMIT} warnings</i></b>",
            reply_markup=control_buttons,
            parse_mode="HTML"
        )

    except Exception as e:
        logger.error(f"Add warn error: {e}")
        await query.answer(f"❌ Error: {str(e)}", show_alert=True)

# ==================== COMMANDS ====================

@dp.message(Command("start"))
async def start_cmd(msg: types.Message):
    """Start command with image and add button"""
    # Create add button
    add_button = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="➕ Add to Group", url="https://t.me/NFSW_Protectionbot?startgroup=true")]
    ])

    # Send with image
    try:
        await bot.send_photo(
            msg.chat.id,
            photo=WELCOME_IMAGE_URL,
            caption="<b><i>🛡️ NSFW Content Moderation Bot</i></b>\n\n"
                   "<b><i>✅ Auto-detects & deletes NSFW</i></b>\n"
                   "<b><i>✅ Smart warning system (5 = mute)</i></b>\n"
                   "<b><i>✅ Global ban system</i></b>\n"
                   "<b><i>✅ Admin controls</i></b>\n\n"
                   "<b><i>🚀 Click below to add me to your group!</i></b>",
            parse_mode="HTML",
            reply_markup=add_button
        )
    except Exception as e:
        logger.error(f"Photo send failed: {e}")
        # Fallback to text
        await msg.answer(
            "<b><i>🛡️ NSFW Content Moderation Bot</i></b>\n\n"
            "<b><i>✅ Auto-detects & deletes NSFW</i></b>\n"
            "<b><i>✅ Smart warning system (5 = mute)</i></b>\n"
            "<b><i>✅ Global ban system</i></b>\n"
            "<b><i>✅ Admin controls</i></b>\n\n"
            "<b><i>🚀 Click below to add me to your group!</i></b>",
            parse_mode="HTML",
            reply_markup=add_button
        )

    logger.info(f"🚀 /start from {msg.from_user.id}")

@dp.message(Command("help"))
async def help_cmd(msg: types.Message):
    await msg.answer(
        "<b><i>📋 Available Commands:</i></b>\n\n"
        "<b><i>/start - Show welcome message</i></b>\n"
        "<b><i>/stats - Bot statistics (owner)</i></b>\n"
        "<b><i>/warn_status - Check warnings</i></b>\n"
        "<b><i>/warn_reset - Reset warnings (admin)</i></b>\n"
        "<b><i>/whitelist_add - Whitelist user (admin)</i></b>\n"
        "<b><i>/gban - Global ban (owner)</i></b>\n"
        "<b><i>/ungban - Remove ban (owner)</i></b>\n\n"
        "<b><i>🛡️ Protect your group now!</i></b>",
        parse_mode="HTML"
    )

@dp.message(Command("stats"))
async def stats_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        await msg.reply("<b><i>❌ Owner only!</i></b>", parse_mode="HTML")
        return

    try:
        stats = await get_bot_stats()
        text = (
            "<b><i>📊 Bot Statistics</i></b>\n\n"
            f"<b><i>💬 Total Chats: {stats.get('total_chats', 0)}</i></b>\n"
            f"<b><i>👥 Groups: {stats.get('total_groups', 0)}</i></b>\n"
            f"<b><i>👤 Users: {stats.get('total_users', 0)}</i></b>\n"
            f"<b><i>🚫 GBanned: {stats.get('total_gbanned', 0)}</i></b>\n"
            f"<b><i>⚠️ Total Warns: {stats.get('total_warns', 0)}</i></b>\n"
            f"<b><i>📝 Total Logs: {stats.get('total_logs', 0)}</i></b>"
        )
        await msg.reply(text, parse_mode="HTML")
    except:
        await msg.reply("<b><i>❌ Error fetching stats</i></b>", parse_mode="HTML")

@dp.message(Command("gban"))
async def gban_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        return

    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("<b><i>❌ Reply to a user message to gban them</i></b>", parse_mode="HTML")
        return

    try:
        target = msg.reply_to_message.from_user
        await gban_user(target.id, "NSFW ban", msg.from_user.id)
        await msg.reply(f"<b><i>✅ {target.full_name} globally banned</i></b>", parse_mode="HTML")
    except:
        await msg.reply("<b><i>❌ Error</i></b>", parse_mode="HTML")

@dp.message(Command("ungban"))
async def ungban_cmd(msg: types.Message):
    if not await is_owner(msg.from_user.id):
        return

    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("<b><i>❌ Reply to a user message</i></b>", parse_mode="HTML")
        return

    try:
        target = msg.reply_to_message.from_user
        if await ungban_user(target.id):
            await msg.reply(f"<b><i>✅ {target.full_name} unbanned</i></b>", parse_mode="HTML")
        else:
            await msg.reply("<b><i>❌ User not gbanned</i></b>", parse_mode="HTML")
    except:
        await msg.reply("<b><i>❌ Error</i></b>", parse_mode="HTML")

@dp.message(Command("warn_status"))
async def warn_status_cmd(msg: types.Message):
    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("<b><i>❌ Reply to a user</i></b>", parse_mode="HTML")
        return

    try:
        target = msg.reply_to_message.from_user
        doc = await warns_col.find_one({"chat_id": msg.chat.id, "user_id": target.id})
        warns = doc.get("warns", 0) if doc else 0
        await msg.reply(f"<b><i>⚠️ {target.full_name}: {warns}/{WARN_LIMIT} warnings</i></b>", parse_mode="HTML")
    except:
        pass

@dp.message(Command("warn_reset"))
async def warn_reset_cmd(msg: types.Message):
    if not await is_user_admin(msg.chat.id, msg.from_user.id):
        await msg.reply("<b><i>❌ Admin only!</i></b>", parse_mode="HTML")
        return

    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("<b><i>❌ Reply to a user</i></b>", parse_mode="HTML")
        return

    try:
        target = msg.reply_to_message.from_user
        await warns_col.delete_one({"chat_id": msg.chat.id, "user_id": target.id})
        await msg.reply(f"<b><i>✅ Warnings reset for {target.full_name}</i></b>", parse_mode="HTML")
    except:
        pass

@dp.message(Command("whitelist_add"))
async def whitelist_add_cmd(msg: types.Message):
    if not await is_user_admin(msg.chat.id, msg.from_user.id):
        await msg.reply("<b><i>❌ Admin only!</i></b>", parse_mode="HTML")
        return

    if not msg.reply_to_message or not msg.reply_to_message.from_user:
        await msg.reply("<b><i>❌ Reply to a user</i></b>", parse_mode="HTML")
        return

    try:
        target = msg.reply_to_message.from_user
        await whitelist_col.update_one(
            {"chat_id": msg.chat.id, "user_id": target.id},
            {"$set": {"username": target.username, "full_name": target.full_name}},
            upsert=True
        )
        await msg.reply(f"<b><i>✅ {target.full_name} whitelisted</i></b>", parse_mode="HTML")
    except:
        pass

# ==================== MEDIA HANDLERS ====================

@dp.message(F.photo)
async def handle_photo(msg: types.Message):
    if msg.chat.type != "private":
        await update_chat_data(msg.chat.id, msg.chat.title)
    if msg.from_user:
        await update_user_data(msg.from_user.id, msg.from_user.username, msg.from_user.first_name)

    logger.info(f"🖼️ Photo from {msg.from_user.id}")
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

    logger.info(f"🎭 Sticker")
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

    logger.info(f"🎬 Video")
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

    logger.info(f"🎞️ GIF")
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
    if mime.startswith(('image/', 'video/')) or any(name.lower().endswith(x) for x in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.mp4', '.mkv', '.webm']):
        logger.info(f"📄 Document: {name}")
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

    logger.info(f"🎥 Video note")
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
        logger.info("✅ Database ready")
    except:
        pass

async def on_startup(bot, **kwargs):
    # your initialization code here
    logger.info("🚀 Bot starting...")
    await setup_database()
    await log_to_channel("✅ NSFW Bot Started!")
    logger.info("✅ Ready!")

async def on_shutdown(bot_instance):
    logger.info("🛑 Shutting down...")
    await log_to_channel("⭕ NSFW Bot Offline")
    mongo.close()

async def main():
    Path("logs").mkdir(exist_ok=True)
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
