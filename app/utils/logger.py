import logging
import os
from logging.handlers import TimedRotatingFileHandler

# Định nghĩa đường dẫn log
LOG_DIR = "logs"
LOG_FILE = "app.log"

# Tạo thư mục log nếu chưa có
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_path = os.path.join(LOG_DIR, LOG_FILE)

# Tạo logger với tên cụ thể
logger = logging.getLogger("QuanLyTaiLieuChatBot")
logger.setLevel(logging.INFO)  # Cấp độ log

# Tạo handler xoay log theo ngày
handler = TimedRotatingFileHandler(
    filename=log_path,
    when="midnight",  # Reset log mỗi ngày lúc 00:00
    backupCount=30,  # Giữ lại 30 file log cũ
    encoding="utf-8",
)

# Định dạng log
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)

# Thêm handler vào logger
logger.addHandler(handler)

# Tắt propagation để tránh log trùng lặp lên logger gốc
logger.propagate = False

# Expose the logger's methods at the module level
error = logger.error
info = logger.info
warning = logger.warning
