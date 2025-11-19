import asyncio
import os
import subprocess
import tempfile
import logging
import time
import cv2
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Tuple, List, Dict, Any

# Aiogram imports
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters.command import Command
from aiogram.types import ChatPermissions, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.exceptions import TelegramAPIError

# Image processing
from PIL import Image

# Database
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

# ==================== CONFIGURATION ====================
# REPLACE WITH YOUR ACTUAL BOT TOKEN
BOT_TOKEN = os.getenv("BOT_TOKEN", "8395371421:AAHeXyUrhwFf-4WdgLe3eU5xCymdCH1snyA")

# Other settings
OWNER_IDS = [int(x) for x in os.getenv("OWNER_IDS", "7641743441,6361404699").split(",")]
LOGGER_GROUP_ID = int(os.getenv("LOGGER_GROUP_ID", "-1002529491709"))
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://deathhide08:UZYSj9T0VuAgIFAB@cluster0.elg19jx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
DATABASE_NAME = "nsfw_bot"

# Bot Settings
WARN_LIMIT = 5
MAX_VIDEO_FRAMES = 6
VIDEO_FPS = 1
MUTE_DURATION = 1800  # 30 minutes

# Detection thresholds
ML_PORN_THRESHOLD = 0.60
ML_SEXY_THRESHOLD = 0.75
ML_HENTAI_THRESHOLD = 0.70
ML_COMBINED_THRESHOLD = 0.50

FALLBACK_SKIN_THRESHOLD = 0.65
FALLBACK_FLESH_THRESHOLD = 0.60
FALLBACK_MIN_SCORE = 7.5

# Welcome image
WELCOME_IMAGE_URL = "https://i.ibb.co/KzK6R4zW/IMG-20251117-212221-098.jpg"

# ==================== SETUP LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== NSFW MODEL SETUP ====================
USE_ML_MODEL = False
NSFW_MODEL = None

try:
    # Try to import and load NSFW model
    from nsfw_detector import predict
    model_path = './nsfw_mobilenet2.224x224.h5'
    if os.path.exists(model_path):
        logger.info("🔄 Loading NSFW ML model...")
        NSFW_MODEL = predict.load_model(model_path)
        USE_ML_MODEL = True
        logger.info("✅ NSFW ML model loaded successfully")
    else:
        logger.warning("⚠️ ML model file not found, using fallback detection")
        USE_ML_MODEL = False
except ImportError as e:
    logger.warning(f"⚠️ nsfw-detector not available: {e}")
    USE_ML_MODEL = False
except Exception as e:
    logger.warning(f"⚠️ ML model loading failed: {e}")
    USE_ML_MODEL = False

# ==================== BOT INITIALIZATION ====================
# Validate bot token
if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
    logger.error("❌ PLEASE SET YOUR BOT_TOKEN!")
    logger.error("💡 Run: export BOT_TOKEN='your_bot_token_here'")
    logger.error("💡 Or edit the BOT_TOKEN variable in this file")
    exit(1)

try:
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    logger.info("✅ Bot initialized successfully")
except Exception as e:
    logger.error(f"❌ Bot initialization failed: {e}")
    exit(1)

# ==================== DATABASE SETUP ====================
try:
    mongo = AsyncIOMotorClient(MONGODB_URI)
    db = mongo[DATABASE_NAME]
    warns_col = db.warns
    log_col = db.logs
    whitelist_col = db.whitelist
    gban_col = db.gban
    chats_col = db.chats
    users_col = db.users
    logger.info("✅ Database connected successfully")
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    exit(1)

# GBan Cache
gban_cache = set()
gban_cache_time = 0
GBAN_CACHE_TTL = 300

# ==================== NSFW DETECTION FUNCTIONS ====================

def detect_face(image_path: str) -> Tuple[bool, float]:
    """Detect faces in image and return (has_face, face_ratio)"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            logger.error("❌ Face cascade classifier not loaded")
            return False, 0.0
            
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0

        height, width = img.shape[:2]
        total_pixels = height * width

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_area = sum(w * h for (x, y, w, h) in faces)
        face_ratio = face_area / total_pixels if total_pixels > 0 else 0

        return len(faces) > 0, face_ratio
    except Exception as e:
        logger.error(f"❌ Face detection error: {e}")
        return False, 0.0

def detect_nsfw_ml(image_path: str) -> Tuple[bool, Optional[dict]]:
    """ML-based NSFW detection using nsfw_detector model"""
    try:
        if not NSFW_MODEL:
            return False, None
            
        # Run prediction
        predictions = NSFW_MODEL.predict([image_path])
        preds = predictions[image_path]

        porn_score = preds.get('porn', 0)
        sexy_score = preds.get('sexy', 0)
        hentai_score = preds.get('hentai', 0)
        neutral_score = preds.get('neutral', 0)
        drawings_score = preds.get('drawings', 0)

        # Detect faces for smart threshold adjustment
        has_face, face_ratio = detect_face(image_path)

        # Calculate combined NSFW score
        combined_nsfw = porn_score + (sexy_score * 0.5) + hentai_score

        # Dynamic threshold based on face detection
        if has_face and face_ratio > 0.15:
            porn_threshold = 0.80
            combined_threshold = 0.70
            logger.info(f"Face detected ({face_ratio:.2%}), applying strict threshold")
        elif has_face:
            porn_threshold = 0.70
            combined_threshold = 0.60
            logger.info(f"Small face detected ({face_ratio:.2%}), moderate threshold")
        else:
            porn_threshold = ML_PORN_THRESHOLD
            combined_threshold = ML_COMBINED_THRESHOLD

        # Check if NSFW
        is_nsfw = (
            porn_score > porn_threshold or
            hentai_score > ML_HENTAI_THRESHOLD or
            combined_nsfw > combined_threshold
        )

        logger.info(f"ML Predictions - Porn: {porn_score:.2f}, Sexy: {sexy_score:.2f}, "
                   f"Hentai: {hentai_score:.2f}, Combined: {combined_nsfw:.2f}, "
                   f"Face: {face_ratio:.2%}, NSFW: {is_nsfw}")

        if is_nsfw:
            return True, {
                "method": "ml_detection",
                "porn_score": porn_score,
                "sexy_score": sexy_score,
                "hentai_score": hentai_score,
                "combined_score": combined_nsfw,
                "face_detected": has_face,
                "face_ratio": face_ratio,
                "predictions": preds
            }

        return False, None

    except Exception as e:
        logger.error(f"❌ ML NSFW detection failed: {e}")
        return False, None

def detect_nsfw_fallback(image_path: str) -> Tuple[bool, Optional[dict]]:
    """Fallback NSFW detection (improved version with stricter rules)"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return False, None

        height, width = img.shape[:2]
        total_pixels = height * width

        # Skip very small images (stickers, icons)
        if total_pixels < 10000:
            logger.info(f"Small image ({total_pixels}px), likely safe")
            return False, None

        # Detect faces
        has_face, face_ratio = detect_face(image_path)

        # Apply much stricter threshold if face detected
        if face_ratio > 0.20:
            MIN_SCORE = 9.0  # Very high threshold for faces
        elif face_ratio > 0.10:
            MIN_SCORE = 8.0
        else:
            MIN_SCORE = FALLBACK_MIN_SCORE

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Skin color detection
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_CLOSE, kernel)

        skin_pixels = cv2.countNonZero(mask_skin)
        skin_percentage = skin_pixels / mask_skin.size

        # Flesh tone detection
        b, g, r = cv2.split(img)
        flesh_mask = (
            (r >= 100) & (r <= 200) &
            (g >= 50) & (g <= 180) &
            (b >= 30) & (b <= 150) &
            (r > g) & (g > b)
        ).astype(np.uint8) * 255

        flesh_pixels = cv2.countNonZero(flesh_mask)
        flesh_percentage = flesh_pixels / flesh_mask.size

        # Edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        edge_density = edge_pixels / edges.size

        # Color analysis
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)
        value_channel = hsv[:, :, 2]
        avg_brightness = np.mean(value_channel)

        nsfw_score = 0
        reasons = []

        # STRICT SCORING - Only flag obvious pornography

        # Extreme skin + flesh combination
        if skin_percentage >= 0.60 and flesh_percentage >= 0.55:
            nsfw_score += 3.5
            reasons.append(f"High skin+flesh: {skin_percentage:.1%}/{flesh_percentage:.1%}")

        # Very high skin coverage
        if skin_percentage >= 0.75:
            nsfw_score += 2.5
            reasons.append(f"Extreme skin: {skin_percentage:.1%}")

        # Flesh-heavy with curves
        if flesh_percentage > 0.65 and edge_density > 0.18:
            nsfw_score += 2.0
            reasons.append(f"Flesh+curves: {flesh_percentage:.1%}")

        # Natural skin tones
        if skin_percentage > 0.50 and avg_saturation < 80 and avg_brightness > 100:
            nsfw_score += 1.5
            reasons.append(f"Natural skin tone")

        # DEDUCTIONS for safe content

        # High brightness = likely safe
        if avg_brightness > 180:
            nsfw_score -= 1.5
            reasons.append(f"High brightness safe")

        # High saturation = cartoons/stickers
        if avg_saturation > 150:
            nsfw_score -= 2.0
            reasons.append(f"High saturation (cartoon)")

        # Very low skin = safe
        if skin_percentage < 0.15:
            nsfw_score -= 2.0
            reasons.append(f"Low skin safe")

        nsfw_score = max(0, nsfw_score)

        logger.info(f"Fallback Score: {nsfw_score:.1f} (Threshold: {MIN_SCORE}), "
                   f"Skin: {skin_percentage:.1%}, Face: {face_ratio:.2%}")

        if nsfw_score >= MIN_SCORE:
            return True, {
                "method": "fallback_detection",
                "score": nsfw_score,
                "skin_percentage": skin_percentage,
                "flesh_percentage": flesh_percentage,
                "face_ratio": face_ratio,
                "reasons": reasons
            }

        return False, None

    except Exception as e:
        logger.error(f"❌ Fallback detection failed: {e}")
        return False, None

def detect_nsfw_improved(image_path: str) -> Tuple[bool, Optional[dict]]:
    """Main NSFW detection - uses ML if available, fallback otherwise"""
    if USE_ML_MODEL and NSFW_MODEL:
        return detect_nsfw_ml(image_path)
    else:
        return detect_nsfw_fallback(image_path)

# ==================== FILE OPERATIONS ====================

@asynccontextmanager
async def temporary_download(downloadable, file_extension: str = ""):
    """Download file from Telegram to temporary location"""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            temp_path = tmp.name
            await bot.download(downloadable, destination=temp_path)
            temp_file = temp_path
            logger.debug(f"Downloaded to {temp_path}")
        yield temp_file
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        yield None
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.debug(f"Cleaned up {temp_file}")
            except Exception as e:
                logger.warning(f"⚠️ Cleanup failed for {temp_file}: {e}")

def convert_webp_to_jpg_pil(webp_path: str) -> str:
    """Convert WEBP to JPG using PIL"""
    try:
        jpg_path = webp_path.rsplit('.webp', 1)[0] + '.jpg'
        with Image.open(webp_path) as im:
            if im.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[-1] if im.mode in ('RGBA', 'LA') else None)
                background.save(jpg_path, "JPEG", quality=95)
            else:
                im.convert("RGB").save(jpg_path, "JPEG", quality=95)
        logger.debug(f"Converted WEBP to JPG: {jpg_path}")
        return jpg_path
    except Exception as e:
        logger.error(f"❌ WEBP conversion failed: {e}")
        return webp_path

async def extract_frames(video_path: str, out_dir: str, fps: int = 1, max_frames: int = 6) -> List[str]:
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
        subprocess.run(cmd, check=True, timeout=60)
        frames = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg')])
        logger.debug(f"Extracted {len(frames)} frames from video")
        return frames[:max_frames]
    except subprocess.TimeoutExpired:
        logger.error("❌ Frame extraction timeout")
        return []
    except Exception as e:
        logger.error(f"❌ Frame extraction failed: {e}")
        return []

# ==================== ANALYSIS FUNCTIONS ====================

async def analyze_image(image_path: str) -> Tuple[bool, Optional[dict]]:
    """Analyze single image for NSFW content"""
    if not image_path or not os.path.exists(image_path):
        return False, None
    try:
        return detect_nsfw_improved(image_path)
    except Exception as e:
        logger.error(f"❌ Image analysis error: {e}")
        return False, None

async def analyze_images(paths: List[str]) -> Tuple[bool, Optional[dict]]:
    """Analyze multiple images - returns True if ANY is NSFW"""
    if not paths:
        return False, None

    try:
        logger.info(f"🔍 Analyzing {len(paths)} image(s)...")
        nsfw_count = 0
        nsfw_result = None

        for idx, path in enumerate(paths, 1):
            if not os.path.exists(path):
                logger.warning(f"Image not found: {path}")
                continue

            is_nsfw, result = await analyze_image(path)
            if is_nsfw:
                nsfw_count += 1
                nsfw_result = result
                logger.warning(f"🚨 NSFW detected in frame {idx}/{len(paths)}")

                # For videos, require at least 2 frames to be NSFW to reduce false positives
                if len(paths) > 1 and nsfw_count >= 2:
                    return True, nsfw_result
                elif len(paths) == 1:
                    return True, nsfw_result

        # For videos, if only 1 frame detected, might be false positive
        if len(paths) > 1 and nsfw_count == 1:
            logger.info(f"⚠️ Only 1/{len(paths)} frames NSFW - likely false positive")
            return False, None
        elif nsfw_count >= 2:
            return True, {"method": "multi_frame", "nsfw_frames": nsfw_count, "total_frames": len(paths)}

        logger.info(f"✅ All {len(paths)} frame(s) are safe")
        return False, None
    except Exception as e:
        logger.error(f"❌ Multi-image analysis error: {e}")
        return False, None

# ==================== DATABASE HELPERS ====================

async def is_user_whitelisted(chat_id: int, user_id: int) -> bool:
    """Check if user is whitelisted"""
    try:
        doc = await whitelist_col.find_one({"chat_id": chat_id, "user_id": user_id})
        return doc is not None
    except Exception as e:
        logger.error(f"❌ Whitelist check error: {e}")
        return False

async def is_user_admin(chat_id: int, user_id: int) -> bool:
    """Check if user is admin"""
    if user_id in OWNER_IDS:
        return True
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        return member.status in ["creator", "administrator"]
    except Exception as e:
        logger.error(f"❌ Admin check error: {e}")
        return False

async def log_to_channel(message: str):
    """Log message to channel"""
    try:
        await bot.send_message(LOGGER_GROUP_ID, message, parse_mode="HTML")
    except Exception as e:
        logger.error(f"❌ Logger channel error: {e}")

async def update_chat_data(chat_id: int, chat_title: str = ""):
    """Update chat information in database"""
    try:
        await chats_col.update_one(
            {"chat_id": chat_id},
            {"$set": {
                "chat_title": chat_title, 
                "last_seen": time.time(), 
                "is_group": chat_id < 0
            }},
            upsert=True
        )
    except Exception as e:
        logger.error(f"❌ Chat data update error: {e}")

async def update_user_data(user_id: int, username: str = "", first_name: str = ""):
    """Update user information in database"""
    try:
        await users_col.update_one(
            {"user_id": user_id},
            {"$set": {
                "username": username, 
                "first_name": first_name, 
                "last_seen": time.time()
            }},
            upsert=True
        )
    except Exception as e:
        logger.error(f"❌ User data update error: {e}")

async def is_user_gbanned(user_id: int) -> bool:
    """Check if user is globally banned (with caching)"""
    global gban_cache_time

    # Refresh cache every 5 minutes
    if time.time() - gban_cache_time > GBAN_CACHE_TTL:
        gban_cache.clear()
        try:
            async for doc in gban_col.find({"banned": True}):
                gban_cache.add(doc["user_id"])
            gban_cache_time = time.time()
            logger.debug(f"🔄 GBan cache refreshed: {len(gban_cache)} users")
        except Exception as e:
            logger.error(f"❌ GBan cache refresh error: {e}")

    return user_id in gban_cache

async def gban_user(user_id: int, reason: str = "No reason", banned_by: int = 0):
    """Globally ban user"""
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

        # Add to cache
        gban_cache.add(user_id)

        # Ban in all groups
        banned_count = 0
        async for chat in chats_col.find({"is_group": True}):
            try:
                await bot.ban_chat_member(chat["chat_id"], user_id)
                banned_count += 1
            except Exception as e:
                logger.debug(f"Ban failed in {chat['chat_id']}: {e}")

        logger.info(f"🚫 GBanned user {user_id} in {banned_count} groups")
        return banned_count
    except Exception as e:
        logger.error(f"❌ GBan error: {e}")
        return 0

async def ungban_user(user_id: int) -> bool:
    """Remove global ban from user"""
    try:
        result = await gban_col.update_one(
            {"user_id": user_id},
            {"$set": {"banned": False}}
        )

        # Remove from cache
        gban_cache.discard(user_id)

        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ Ungban error: {e}")
        return False

async def get_bot_stats() -> Dict[str, Any]:
    """Get bot statistics"""
    try:
        return {
            "total_chats": await chats_col.count_documents({}),
            "total_groups": await chats_col.count_documents({"is_group": True}),
            "total_users": await users_col.count_documents({}),
            "total_gbanned": await gban_col.count_documents({"banned": True}),
            "total_warns": await warns_col.count_documents({}),
            "total_logs": await log_col.count_documents({})
        }
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        return {}

# ==================== MODERATION FUNCTIONS ====================

async def handle_detect_and_action(message: types.Message, file_path: str, media_type: str = "media"):
    """Analyze media and take action if NSFW"""
    user_id = message.from_user.id if message.from_user else None
    chat_id = message.chat.id

    if not file_path:
        return

    logger.info(f"🔍 Analyzing {media_type}: msg_id={message.message_id}, user={user_id}")

    # GBan check
    if user_id and await is_user_gbanned(user_id):
        try:
            await message.delete()
            logger.info(f"🗑️ Deleted message from gbanned user {user_id}")
        except Exception as e:
            logger.error(f"❌ Delete failed for gbanned user: {e}")
        return

    # Skip admins and whitelisted users
    if user_id:
        if await is_user_admin(chat_id, user_id) or await is_user_whitelisted(chat_id, user_id):
            logger.info(f"⏭️ Skipping admin/whitelisted user {user_id}")
            return

    is_nsfw = False
    reason = None
    temp_files_to_cleanup = []

    try:
        file_lower = file_path.lower()

        # Handle different media types
        if file_lower.endswith((".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".3gp")):
            # Video files - extract and analyze frames
            with tempfile.TemporaryDirectory() as tmpdir:
                frames = await extract_frames(file_path, tmpdir, fps=VIDEO_FPS, max_frames=MAX_VIDEO_FRAMES)
                if frames:
                    is_nsfw, reason = await analyze_images(frames)
                else:
                    logger.warning("No frames extracted from video")

        elif file_lower.endswith(".webp"):
            # WEBP images/stickers - convert to JPG
            jpg_path = convert_webp_to_jpg_pil(file_path)
            if jpg_path != file_path and os.path.exists(jpg_path):
                temp_files_to_cleanup.append(jpg_path)
                is_nsfw, reason = await analyze_images([jpg_path])
            else:
                is_nsfw, reason = await analyze_images([file_path])

        elif file_lower.endswith((".gif", ".gifv")):
            # GIF files - extract frames
            with tempfile.TemporaryDirectory() as tmpdir:
                frames = await extract_frames(file_path, tmpdir, fps=VIDEO_FPS, max_frames=MAX_VIDEO_FRAMES)
                if frames:
                    is_nsfw, reason = await analyze_images(frames)
                else:
                    logger.warning("No frames extracted from GIF")

        else:
            # Regular images
            is_nsfw, reason = await analyze_images([file_path])

        # Take action if NSFW detected
        if is_nsfw:
            logger.warning(f"🚨 NSFW DETECTED! msg={message.message_id}, user={user_id}, "
                         f"method={reason.get('method')}, score={reason.get('score', reason.get('porn_score', 'N/A'))}")
            await take_moderation_action(message, reason, user_id, chat_id)
        else:
            logger.info(f"✅ Content is safe")

    except Exception as e:
        logger.error(f"❌ Detection error: {e}", exc_info=True)
    finally:
        # Cleanup temporary files
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"⚠️ Temp file cleanup failed: {e}")

async def take_moderation_action(message: types.Message, reason: dict, user_id: int, chat_id: int):
    """Delete NSFW content and warn user"""
    # Delete the message
    try:
        logger.info(f"🗑️ Deleting message {message.message_id}")
        await message.delete()
        logger.info(f"✅ Message deleted")
    except Exception as e:
        logger.error(f"❌ Delete failed: {e}")

    # Warn user
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
        logger.info(f"⚠️ Warning #{current_warns} issued to user {user_id}")

        # Log to database
        await log_col.insert_one({
            "chat_id": chat_id,
            "user_id": user_id,
            "message_id": message.message_id,
            "reason": str(reason),
            "timestamp": time.time(),
            "media_type": message.content_type,
            "warn_count": current_warns
        })

        # Send warning message with control buttons
        try:
            user_name = message.from_user.full_name if message.from_user else "User"
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
                f"<b>⚠️ Warning</b>\n"
                f"User: {user_name}\n"
                f"Warnings: {current_warns}/{WARN_LIMIT}\n"
                f"<i>NSFW content detected and removed</i>",
                reply_markup=control_buttons,
                parse_mode="HTML"
            )

            # Auto-delete warning after 10 seconds
            await asyncio.sleep(10)
            try:
                await warning_msg.delete()
            except:
                pass
                
        except Exception as e:
            logger.error(f"❌ Warning message error: {e}")

        # Mute if limit reached
        if current_warns >= WARN_LIMIT:
            await mute_user(chat_id, user_id, message.from_user.full_name if message.from_user else "User")

    except Exception as e:
        logger.error(f"❌ Warning system error: {e}")

async def mute_user(chat_id: int, user_id: int, user_name: str):
    """Mute user for NSFW violations"""
    try:
        logger.info(f"🔇 Muting user {user_id} in chat {chat_id}")
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

        # Control buttons
        control_buttons = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="➖ Remove Warn", callback_data=f"remove_warn_{user_id}_{chat_id}"),
                InlineKeyboardButton(text="➕ Add Warn", callback_data=f"add_warn_{user_id}_{chat_id}")
            ],
            [
                InlineKeyboardButton(text="🔊 Unmute User", callback_data=f"unmute_{user_id}_{chat_id}")
            ]
        ])

        await bot.send_message(
            chat_id,
            f"<b>🔇 User Muted</b>\n"
            f"User: {user_name}\n"
            f"Duration: {MUTE_DURATION // 60} minutes\n"
            f"<i>Reached warning limit for NSFW content</i>",
            reply_markup=control_buttons,
            parse_mode="HTML"
        )
        
        logger.info(f"✅ User {user_id} muted successfully")

    except Exception as e:
        logger.error(f"❌ Mute error: {e}")

# ==================== CALLBACK HANDLERS ====================

@dp.callback_query(lambda query: query.data.startswith("unmute_"))
async def unmute_callback(query: types.CallbackQuery):
    """Unmute user and reset warnings"""
    try:
        data_parts = query.data.split("_")
        user_id = int(data_parts[1])
        chat_id = int(data_parts[2])

        # Admin check
        if not await is_user_admin(chat_id, query.from_user.id):
            await query.answer("❌ Only admins can unmute!", show_alert=True)
            return

        # Unmute
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

        # Reset warnings
        await warns_col.delete_one({"chat_id": chat_id, "user_id": user_id})

        await query.answer("✅ User unmuted and warnings reset!", show_alert=False)

        # Update message
        user = await bot.get_chat_member(chat_id, user_id)
        await query.message.edit_text(
            f"<b>✅ User Unmuted</b>\n"
            f"User: {user.user.full_name}\n"
            f"By: {query.from_user.full_name}\n"
            f"<i>Warnings have been reset</i>",
            parse_mode="HTML"
        )
        
        logger.info(f"🔊 User {user_id} unmuted and warnings reset by {query.from_user.id}")

    except Exception as e:
        logger.error(f"❌ Unmute error: {e}")
        await query.answer(f"❌ Error: {str(e)}", show_alert=True)

@dp.callback_query(lambda query: query.data.startswith("remove_warn_"))
async def remove_warn_callback(query: types.CallbackQuery):
    """Remove one warning from user"""
    try:
        data_parts = query.data.split("_")
        user_id = int(data_parts[2])
        chat_id = int(data_parts[3])

        # Admin check
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

            # Update message
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
                f"<b>⚠️ Warning Status</b>\n"
                f"User: {user.user.full_name}\n"
                f"Warnings: {new_warns}/{WARN_LIMIT}",
                reply_markup=control_buttons,
                parse_mode="HTML"
            )
        else:
            await query.answer("❌ No warnings to remove!", show_alert=True)

    except Exception as e:
        logger.error(f"❌ Remove warn error: {e}")
        await query.answer(f"❌ Error: {str(e)}", show_alert=True)

@dp.callback_query(lambda query: query.data.startswith("add_warn_"))
async def add_warn_callback(query: types.CallbackQuery):
    """Add one warning to user"""
    try:
        data_parts = query.data.split("_")
        user_id = int(data_parts[2])
        chat_id = int(data_parts[3])

        # Admin check
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

        # Mute if limit reached
        if new_warns >= WARN_LIMIT:
            await mute_user(chat_id, user_id, user.user.full_name)

        # Update message
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
            f"<b>⚠️ Warning Status</b>\n"
            f"User: {user.user.full_name}\n"
            f"Warnings: {new_warns}/{WARN_LIMIT}",
            reply_markup=control_buttons,
            parse_mode="HTML"
        )

    except Exception as e:
        logger.error(f"❌ Add warn error: {e}")
        await query.answer(f"❌ Error: {str(e)}", show_alert=True)

# ==================== COMMAND HANDLERS ====================

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    """Start command handler"""
    add_button = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="➕ Add to Group", 
            url=f"https://t.me/{(await bot.get_me()).username}?startgroup=true"
        )]
    ])

    try:
        await bot.send_photo(
            message.chat.id,
            photo=WELCOME_IMAGE_URL,
            caption="<b>🛡️ NSFW Content Moderation Bot</b>\n\n"
                   "✅ AI-powered NSFW detection\n"
                   "✅ Smart warning system (5 = mute)\n"
                   "✅ Global ban system\n"
                   "✅ Admin controls\n"
                   "✅ Protects against pornography\n\n"
                   "<i>Click below to add me to your group!</i>",
            parse_mode="HTML",
            reply_markup=add_button
        )
    except Exception as e:
        logger.error(f"❌ Photo send failed: {e}")
        # Fallback to text message
        await message.answer(
            "<b>🛡️ NSFW Content Moderation Bot</b>\n\n"
            "✅ AI-powered NSFW detection\n"
            "✅ Smart warning system (5 = mute)\n"
            "✅ Global ban system\n"
            "✅ Admin controls\n"
            "✅ Protects against pornography\n\n"
            "<i>Add me to your group to get started!</i>",
            parse_mode="HTML",
            reply_markup=add_button
        )

    logger.info(f"🚀 /start from user {message.from_user.id}")

@dp.message(Command("help"))
async def help_cmd(message: types.Message):
    """Help command handler"""
    help_text = (
        "<b>📋 Available Commands:</b>\n\n"
        "<b>For Everyone:</b>\n"
        "/start - Welcome message\n"
        "/help - This help message\n\n"
        "<b>For Admins:</b>\n"
        "/warn_status - Check user warnings\n"
        "/warn_reset - Reset warnings\n"
        "/whitelist_add - Whitelist user\n"
        "/whitelist_remove - Remove from whitelist\n\n"
        "<b>For Owners:</b>\n"
        "/stats - Bot statistics\n"
        "/gban - Global ban user\n"
        "/ungban - Remove global ban\n\n"
        "<i>🛡️ Bot automatically detects and removes NSFW content!</i>\n"
        "<i>⚠️ 5 warnings = automatic mute</i>"
    )
    
    await message.answer(help_text, parse_mode="HTML")

@dp.message(Command("stats"))
async def stats_cmd(message: types.Message):
    """Statistics command (owner only)"""
    if message.from_user.id not in OWNER_IDS:
        await message.reply("<b>❌ Owner only command!</b>", parse_mode="HTML")
        return

    try:
        stats = await get_bot_stats()
        detection_method = "ML Model" if USE_ML_MODEL else "Fallback Algorithm"

        stats_text = (
            "<b>📊 Bot Statistics</b>\n\n"
            f"💬 Total Chats: <code>{stats.get('total_chats', 0)}</code>\n"
            f"👥 Groups: <code>{stats.get('total_groups', 0)}</code>\n"
            f"👤 Users: <code>{stats.get('total_users', 0)}</code>\n"
            f"🚫 GBanned: <code>{stats.get('total_gbanned', 0)}</code>\n"
            f"⚠️ Total Warnings: <code>{stats.get('total_warns', 0)}</code>\n"
            f"📝 Total Logs: <code>{stats.get('total_logs', 0)}</code>\n\n"
            f"🔬 Detection: <code>{detection_method}</code>"
        )
        
        await message.reply(stats_text, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        await message.reply("<b>❌ Error fetching statistics</b>", parse_mode="HTML")

@dp.message(Command("gban"))
async def gban_cmd(message: types.Message):
    """Global ban command (owner only)"""
    if message.from_user.id not in OWNER_IDS:
        return

    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply("<b>❌ Reply to a user message to gban them</b>", parse_mode="HTML")
        return

    try:
        target = message.reply_to_message.from_user
        banned_count = await gban_user(target.id, "Global ban", message.from_user.id)
        await message.reply(
            f"<b>✅ {target.full_name} globally banned in {banned_count} groups</b>",
            parse_mode="HTML"
        )
        await log_to_channel(
            f"<b>🚫 Global Ban</b>\n"
            f"User: {target.full_name} (<code>{target.id}</code>)\n"
            f"By: {message.from_user.full_name} (<code>{message.from_user.id}</code>)\n"
            f"Banned in: {banned_count} groups"
        )
    except Exception as e:
        logger.error(f"❌ GBan error: {e}")
        await message.reply("<b>❌ Error</b>", parse_mode="HTML")

@dp.message(Command("ungban"))
async def ungban_cmd(message: types.Message):
    """Remove global ban (owner only)"""
    if message.from_user.id not in OWNER_IDS:
        return

    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply("<b>❌ Reply to a user message</b>", parse_mode="HTML")
        return

    try:
        target = message.reply_to_message.from_user
        if await ungban_user(target.id):
            await message.reply(f"<b>✅ {target.full_name} unbanned</b>", parse_mode="HTML")
            await log_to_channel(
                f"<b>✅ Global Unban</b>\n"
                f"User: {target.full_name} (<code>{target.id}</code>)\n"
                f"By: {message.from_user.full_name} (<code>{message.from_user.id}</code>)"
            )
        else:
            await message.reply("<b>❌ User not gbanned</b>", parse_mode="HTML")
    except Exception as e:
        logger.error(f"❌ Ungban error: {e}")
        await message.reply("<b>❌ Error</b>", parse_mode="HTML")

@dp.message(Command("warn_status"))
async def warn_status_cmd(message: types.Message):
    """Check warning status"""
    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply("<b>❌ Reply to a user message</b>", parse_mode="HTML")
        return

    try:
        target = message.reply_to_message.from_user
        doc = await warns_col.find_one({"chat_id": message.chat.id, "user_id": target.id})
        warns = doc.get("warns", 0) if doc else 0
        await message.reply(
            f"<b>⚠️ {target.full_name}: {warns}/{WARN_LIMIT} warnings</b>",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"❌ Warn status error: {e}")
        await message.reply("<b>❌ Error</b>", parse_mode="HTML")

@dp.message(Command("warn_reset"))
async def warn_reset_cmd(message: types.Message):
    """Reset warnings (admin only)"""
    if not await is_user_admin(message.chat.id, message.from_user.id):
        await message.reply("<b>❌ Admin only!</b>", parse_mode="HTML")
        return

    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply("<b>❌ Reply to a user message</b>", parse_mode="HTML")
        return

    try:
        target = message.reply_to_message.from_user
        await warns_col.delete_one({"chat_id": message.chat.id, "user_id": target.id})
        await message.reply(
            f"<b>✅ Warnings reset for {target.full_name}</b>",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"❌ Warn reset error: {e}")
        await message.reply("<b>❌ Error</b>", parse_mode="HTML")

@dp.message(Command("whitelist_add"))
async def whitelist_add_cmd(message: types.Message):
    """Add user to whitelist (admin only)"""
    if not await is_user_admin(message.chat.id, message.from_user.id):
        await message.reply("<b>❌ Admin only!</b>", parse_mode="HTML")
        return

    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply("<b>❌ Reply to a user message</b>", parse_mode="HTML")
        return

    try:
        target = message.reply_to_message.from_user
        await whitelist_col.update_one(
            {"chat_id": message.chat.id, "user_id": target.id},
            {"$set": {"username": target.username, "full_name": target.full_name}},
            upsert=True
        )
        await message.reply(
            f"<b>✅ {target.full_name} whitelisted</b>",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"❌ Whitelist add error: {e}")
        await message.reply("<b>❌ Error</b>", parse_mode="HTML")

@dp.message(Command("whitelist_remove"))
async def whitelist_remove_cmd(message: types.Message):
    """Remove user from whitelist (admin only)"""
    if not await is_user_admin(message.chat.id, message.from_user.id):
        await message.reply("<b>❌ Admin only!</b>", parse_mode="HTML")
        return

    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply("<b>❌ Reply to a user message</b>", parse_mode="HTML")
        return

    try:
        target = message.reply_to_message.from_user
        result = await whitelist_col.delete_one({"chat_id": message.chat.id, "user_id": target.id})
        if result.deleted_count > 0:
            await message.reply(
                f"<b>✅ {target.full_name} removed from whitelist</b>",
                parse_mode="HTML"
            )
        else:
            await message.reply(
                f"<b>❌ {target.full_name} is not whitelisted</b>",
                parse_mode="HTML"
            )
    except Exception as e:
        logger.error(f"❌ Whitelist remove error: {e}")
        await message.reply("<b>❌ Error</b>", parse_mode="HTML")

# ==================== MEDIA HANDLERS ====================

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    """Handle photo messages"""
    if message.chat.type != "private":
        await update_chat_data(message.chat.id, message.chat.title)
    if message.from_user:
        await update_user_data(message.from_user.id, message.from_user.username, message.from_user.first_name)

    logger.debug(f"📸 Photo from user {message.from_user.id}")
    try:
        async with temporary_download(message.photo[-1], ".jpg") as file_path:
            if file_path:
                await handle_detect_and_action(message, file_path, "photo")
    except Exception as e:
        logger.error(f"❌ Photo handler error: {e}")

@dp.message(F.sticker)
async def handle_sticker(message: types.Message):
    """Handle sticker messages"""
    if message.chat.type != "private":
        await update_chat_data(message.chat.id, message.chat.title)
    if message.from_user:
        await update_user_data(message.from_user.id, message.from_user.username, message.from_user.first_name)

    logger.debug(f"🎭 Sticker from user {message.from_user.id}")
    try:
        ext = ".webm" if message.sticker.is_video else ".webp"
        async with temporary_download(message.sticker, ext) as file_path:
            if file_path:
                await handle_detect_and_action(message, file_path, "sticker")
    except Exception as e:
        logger.error(f"❌ Sticker handler error: {e}")

@dp.message(F.video)
async def handle_video(message: types.Message):
    """Handle video messages"""
    if message.chat.type != "private":
        await update_chat_data(message.chat.id, message.chat.title)
    if message.from_user:
        await update_user_data(message.from_user.id, message.from_user.username, message.from_user.first_name)

    logger.debug(f"🎬 Video from user {message.from_user.id}")
    try:
        async with temporary_download(message.video, ".mp4") as file_path:
            if file_path:
                await handle_detect_and_action(message, file_path, "video")
    except Exception as e:
        logger.error(f"❌ Video handler error: {e}")

@dp.message(F.animation)
async def handle_animation(message: types.Message):
    """Handle GIF/animation messages"""
    if message.chat.type != "private":
        await update_chat_data(message.chat.id, message.chat.title)
    if message.from_user:
        await update_user_data(message.from_user.id, message.from_user.username, message.from_user.first_name)

    logger.debug(f"🎞️ GIF from user {message.from_user.id}")
    try:
        async with temporary_download(message.animation, ".mp4") as file_path:
            if file_path:
                await handle_detect_and_action(message, file_path, "animation")
    except Exception as e:
        logger.error(f"❌ Animation handler error: {e}")

@dp.message(F.document)
async def handle_document(message: types.Message):
    """Handle document messages (images/videos only)"""
    if message.chat.type != "private":
        await update_chat_data(message.chat.id, message.chat.title)
    if message.from_user:
        await update_user_data(message.from_user.id, message.from_user.username, message.from_user.first_name)

    mime = message.document.mime_type or ""
    name = message.document.file_name or ""

    # Only process image/video documents
    if (mime.startswith(('image/', 'video/')) or 
        any(name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.mp4', '.mkv', '.webm'])):
        
        logger.debug(f"📄 Document from user {message.from_user.id}: {name}")
        try:
            ext = '.jpg' if mime.startswith('image/') else '.mp4'
            async with temporary_download(message.document, ext) as file_path:
                if file_path:
                    await handle_detect_and_action(message, file_path, "document")
        except Exception as e:
            logger.error(f"❌ Document handler error: {e}")

@dp.message(F.video_note)
async def handle_video_note(message: types.Message):
    """Handle video note messages"""
    if message.chat.type != "private":
        await update_chat_data(message.chat.id, message.chat.title)
    if message.from_user:
        await update_user_data(message.from_user.id, message.from_user.username, message.from_user.first_name)

    logger.debug(f"🎥 Video note from user {message.from_user.id}")
    try:
        async with temporary_download(message.video_note, ".mp4") as file_path:
            if file_path:
                await handle_detect_and_action(message, file_path, "video_note")
    except Exception as e:
        logger.error(f"❌ Video note handler error: {e}")

@dp.message(F.text)
async def handle_text(message: types.Message):
    """Handle text messages (for user tracking)"""
    if message.chat.type != "private":
        await update_chat_data(message.chat.id, message.chat.title)
    if message.from_user:
        await update_user_data(message.from_user.id, message.from_user.username, message.from_user.first_name)

# ==================== DATABASE SETUP ====================

async def setup_database():
    """Initialize database indexes"""
    try:
        await warns_col.create_index([("chat_id", 1), ("user_id", 1)], unique=True)
        await log_col.create_index([("chat_id", 1), ("timestamp", -1)])
        await whitelist_col.create_index([("chat_id", 1), ("user_id", 1)], unique=True)
        await gban_col.create_index([("user_id", 1)], unique=True)
        await chats_col.create_index([("chat_id", 1)], unique=True)
        await users_col.create_index([("user_id", 1)], unique=True)
        logger.info("✅ Database indexes created")
    except Exception as e:
        logger.error(f"❌ Database setup error: {e}")

# ==================== MAIN FUNCTION ====================

async def main():
    """Main bot entry point"""
    try:
        # Setup database
        await setup_database()

        # Get bot info
        bot_info = await bot.get_me()
        logger.info(f"🤖 Bot: @{bot_info.username}")
        logger.info(f"🔬 Detection: {'ML Model' if USE_ML_MODEL else 'Fallback Algorithm'}")
        logger.info("🚀 Bot starting...")

        # Start polling
        await dp.start_polling(bot, skip_updates=True)

    except KeyboardInterrupt:
        logger.info("⏹️ Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        # Cleanup
        try:
            await bot.session.close()
            mongo.close()
            logger.info("✅ Cleanup complete")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Run the bot
    asyncio.run(main())
