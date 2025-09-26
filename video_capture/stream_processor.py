import os
import time
import json
import threading
import subprocess
import logging
from typing import Dict
import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn


# 创建日志目录
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# 配置日志
log_name = datetime.datetime.now().strftime("%Y-%m-%d")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"./logs/video_{log_name}.log", "a", encoding="utf-8")
    ]
)



# 全局变量
CONFIG_FILE = os.path.join(os.getcwd(), "./stream_config.json")
RUNNING_STREAMS = {}  # Dictionary to track running processes
SYNC_RECORDING = False  # 同步录制状态
RECORDING_START_TIME = None  # 录制开始时间
ACTUAL_RECORDING_START_TIME = None  # 实际录制开始时间（第一个摄像头开始录制时）
RECORDING_LOGS = []  # 录制日志

# 默认配置（当配置文件不存在时使用）
DEFAULT_CONFIG = {
    "streams": [
        {
            "name": "Camera 1",
            "rtsp_url": "rtsp://admin:zhangleiSuccess@@192.168.2.64:554/Streaming/Channels/101?transportmode=unicast",
            "enabled": True,
        }
    ],
    "output_directory": "recordings",
     "segment_time": 60,
}


class StreamProcessor:
    """使用ffmpeg处理RTSP流的类"""

    def __init__(self, stream_config: Dict, output_dir: str,segment_time: int):
        self.name = stream_config.get("name", "Unnamed Stream")
        self.rtsp_url = stream_config["rtsp_url"]
        # 单位是秒
        self.segment_time = segment_time
        self.output_dir = output_dir
        self.enabled = stream_config.get("enabled", True)
        self.process = None
        self.status = "stopped"
        self.start_time = None
        self.thread = None
        self.actual_start_time = None  # 实际开始录制的时间

    def get_status(self):
        """返回当前状态信息"""
        return {
            "name": self.name,
            "status": self.status,
            "duration": self.get_duration(),
            "output_dir": self.output_dir,
            "enabled": self.enabled,
            "rtsp_url": self.rtsp_url,
        }

    def get_duration(self):
        """计算运行过程的持续时间"""
        if self.start_time and self.status == "running":
            return int((time.time()*1000) - self.start_time)
        return 0

    def start(self):
        """开始处理视频流"""
        if self.status == "running":
            logging.warning(f"Stream {self.name} is already running")
            return

        if not self.enabled:
            logging.info(f"Stream {self.name} is disabled, not starting")
            return

        #  视频输出根目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 日期
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        cam_id = self.name.replace(" ", "").lower()  # 去掉空格并转换为小写
        
        #  创建目录: video/<⽇期>/<摄像头ID>
        date_dir = os.path.join(self.output_dir, date_str)
        cam_dir = os.path.join(date_dir, cam_id)
        os.makedirs(cam_dir, exist_ok=True)
        
        # video/<⽇期>/<摄像头ID>/<时间>.mp4  日期格式yyyy-mm-dd
        # output_file = os.path.join(cam_dir, "%Y-%m-%d_%H-%M.mp4")
        # %Y-%m-%d_%H-%M_%02d.mp4
        output_file = os.path.join(cam_dir, "%Y-%m-%d_%H-%M-%S.mp4")
        print(output_file)

        # 计算等待时间，直到下一个10秒整数倍开始录制(更快同步)
        now = datetime.datetime.now()
        current_second = now.second
        # 等待到下一个10秒的整数倍 (00, 10, 20, 30, 40, 50)
        seconds_to_wait = 10 - (current_second % 10)
        if seconds_to_wait == 10:
            seconds_to_wait = 0  # 如果当前就是整10秒，不需要等待
        
        if seconds_to_wait > 0:
            target_second = current_second + seconds_to_wait
            logging.info(f"Stream {self.name} will start in {seconds_to_wait} seconds (waiting for :{target_second:02d} seconds)")
            time.sleep(seconds_to_wait)
            logging.info(f"Stream {self.name} starting now at {datetime.datetime.now().strftime('%H:%M:%S')}")

        # 构建ffmpeg命令
        cmd = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",  # 使用TCP传输协议，减少丢包
            "-use_wallclock_as_timestamps",
            "1",  # 使用系统时钟作为时间戳，有助于同步
             "-err_detect", 
             "ignore_err",  # 忽略可恢复的错误
            "-i",
            self.rtsp_url,
            "-an",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict", "-2",
            "-fflags", "+genpts+discardcorrupt", 
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_time),
            "-segment_format",
            "mp4",
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            output_file,
        ]

        logging.info(f"Starting stream processing for {self.name}")
        logging.debug(f"Command: {' '.join(cmd)}")

        # 启动ffmpeg进程
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE
            )
            self.status = "running"
            self.start_time = int(time.time()*1000)  # Store start time
            self.actual_start_time = time.time()  # 实际开始录制的时间

            # 启动监控线程
            self.thread = threading.Thread(target=self._monitor_process)
            self.thread.daemon = True
            self.thread.start()

            logging.info(f"Stream {self.name} started successfully")
            
            # 通知全局录制开始时间更新
            update_global_recording_start_time(self.actual_start_time)
        except Exception as e:
            logging.error(f"Failed to start stream {self.name}: {str(e)}", exc_info=True)
            self.status = "error"

    def stop(self):
        """优雅停止视频流处理"""
        if self.process and self.status == "running":
            logging.info(f"Gracefully stopping stream {self.name}")
            
            # 优雅停止：向ffmpeg发送'q'命令
            try:
                self.process.stdin.write(b'q\n')
                self.process.stdin.flush()
                logging.info(f"Sent quit command to stream {self.name}")
                
                # 等待进程优雅退出
                try:
                    self.process.wait(timeout=10)  # 给更多时间让ffmpeg完成写入
                    logging.info(f"Stream {self.name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logging.warning(f"Stream {self.name} didn't stop gracefully, terminating")
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logging.warning(f"Force killing stream {self.name}")
                        self.process.kill()
                        
            except Exception as e:
                logging.warning(f"Error during graceful stop of {self.name}: {e}, using terminate")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()

            self.status = "stopped"
            self.process = None
            self.actual_start_time = None
            return True
        return False

    def _monitor_process(self):
        """监控ffmpeg进程并处理其完成状态"""
        if not self.process:
            return
            
        # 在单独的线程中读取输出
        def log_output(pipe, log_level):
            while True:
                line = pipe.readline()
                if not line:
                    break
                try:
                    line = line.decode('utf-8').strip()
                    if line:
                        logging.info(f"[{self.name}] {line}")
                        # if log_level == logging.INFO:
                        #     logging.info(f"[{self.name}] {line}")
                        # else:
                        #     logging.error(f"[{self.name}] {line}")
                except UnicodeDecodeError:
                    pass

        # 创建读取输出的线程
        stdout_thread = threading.Thread(target=log_output, args=(self.process.stdout, logging.INFO))
        stderr_thread = threading.Thread(target=log_output, args=(self.process.stderr, logging.ERROR))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        self.process.wait()
        # 等待输出线程结束
        stdout_thread.join()
        stderr_thread.join()

        # 检查进程是否正常终止
        if self.process and self.process.returncode != 0:
            logging.warning(f"Stream {self.name} process exited with code {self.process.returncode}")
            self.status = "error"
        else:
            logging.info(f"Stream {self.name} process completed successfully")
            self.status = "stopped"

        self.process = None


def load_config():
    """从JSON文件加载流配置"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
            logging.info(f"Configuration loaded from {CONFIG_FILE}")
            return config
        else:
            # 如果配置文件不存在，创建默认配置
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            logging.info(f"Default configuration created at {CONFIG_FILE}")
            return DEFAULT_CONFIG
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}", exc_info=True)
        return DEFAULT_CONFIG


def save_config(config):
    """将流配置保存到JSON文件"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logging.info(f"Configuration saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        logging.error(f"Error saving configuration: {str(e)}", exc_info=True)
        return False


def update_global_recording_start_time(actual_start_time):
    """更新全局实际录制开始时间（仅第一个摄像头开始时设置）"""
    global ACTUAL_RECORDING_START_TIME
    if SYNC_RECORDING and ACTUAL_RECORDING_START_TIME is None:
        ACTUAL_RECORDING_START_TIME = actual_start_time
        add_recording_log(f"⏰ 实际录制开始于 {datetime.datetime.fromtimestamp(actual_start_time).strftime('%H:%M:%S')}")
        logging.info(f"Global recording start time set to {actual_start_time}")


def initialize_streams():
    """从配置初始化流处理器"""
    config = load_config()
    for stream_config in config["streams"]:
        name = stream_config.get("name", "Unnamed Stream")
        if stream_config.get("enabled", True):
            processor = StreamProcessor(stream_config, config["output_directory"],config['segment_time'])
            RUNNING_STREAMS[name] = processor
            logging.info(f"Initialized stream: {name}")


def start_streams():
    """启动所有启用的流处理器"""
    for name, processor in RUNNING_STREAMS.items():
        if processor.enabled:
            threading.Thread(target=processor.start).start()


def add_recording_log(message):
    """添加录制日志"""
    global RECORDING_LOGS
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    RECORDING_LOGS.append(log_entry)
    # 保持日志数量在合理范围内
    if len(RECORDING_LOGS) > 100:
        RECORDING_LOGS = RECORDING_LOGS[-100:]
    logging.info(message)


def start_sync_recording():
    """开始同步录制所有摄像头"""
    global SYNC_RECORDING, RECORDING_START_TIME, ACTUAL_RECORDING_START_TIME, RECORDING_LOGS
    
    if SYNC_RECORDING:
        add_recording_log("❌ 录制已在进行中，请先停止当前录制")
        return False
    
    SYNC_RECORDING = True
    RECORDING_START_TIME = time.time()  # 记录按钮点击时间
    ACTUAL_RECORDING_START_TIME = None  # 重置实际录制开始时间
    RECORDING_LOGS = []  # 清空日志
    
    add_recording_log("🎬 开始同步录制所有摄像头")
    
    success_count = 0
    failed_cameras = []
    
    for name, processor in RUNNING_STREAMS.items():
        if processor.enabled:
            try:
                if processor.status != "running":
                    threading.Thread(target=processor.start).start()
                    add_recording_log(f"✅ 启动摄像头: {name}")
                    success_count += 1
                else:
                    add_recording_log(f"ℹ️ 摄像头 {name} 已在运行")
                    success_count += 1
            except Exception as e:
                failed_cameras.append(name)
                add_recording_log(f"❌ 启动摄像头 {name} 失败: {str(e)}")
        else:
            add_recording_log(f"⚠️ 摄像头 {name} 已禁用")
    
    if failed_cameras:
        add_recording_log(f"⚠️ 部分摄像头启动失败: {', '.join(failed_cameras)}")
    
    add_recording_log(f"📊 录制状态: {success_count} 个摄像头成功启动")
    
    return True


def stop_sync_recording():
    """停止同步录制所有摄像头"""
    global SYNC_RECORDING, RECORDING_START_TIME, ACTUAL_RECORDING_START_TIME
    
    if not SYNC_RECORDING:
        add_recording_log("❌ 当前没有进行录制")
        return False
    
    # 使用实际录制开始时间计算时长
    if ACTUAL_RECORDING_START_TIME:
        duration = time.time() - ACTUAL_RECORDING_START_TIME
    else:
        duration = 0  # 如果没有实际开始录制，时长为0
    
    duration_str = f"{int(duration//60)}:{int(duration%60):02d}"
    
    add_recording_log(f"⏹️ 停止同步录制 (实际录制时长: {duration_str})")
    
    success_count = 0
    failed_cameras = []
    
    for name, processor in RUNNING_STREAMS.items():
        try:
            if processor.status == "running":
                if processor.stop():
                    add_recording_log(f"✅ 停止摄像头: {name}")
                    success_count += 1
                else:
                    failed_cameras.append(name)
                    add_recording_log(f"❌ 停止摄像头 {name} 失败")
            else:
                add_recording_log(f"ℹ️ 摄像头 {name} 已停止")
                success_count += 1
        except Exception as e:
            failed_cameras.append(name)
            add_recording_log(f"❌ 停止摄像头 {name} 异常: {str(e)}")
    
    if failed_cameras:
        add_recording_log(f"⚠️ 部分摄像头停止失败: {', '.join(failed_cameras)}")
    
    add_recording_log(f"📊 停止结果: {success_count} 个摄像头成功停止")
    add_recording_log("🏁 同步录制已结束")
    
    SYNC_RECORDING = False
    RECORDING_START_TIME = None
    ACTUAL_RECORDING_START_TIME = None
    
    return True


def get_recording_status():
    """获取录制状态信息"""
    duration = 0
    if SYNC_RECORDING and ACTUAL_RECORDING_START_TIME:
        duration = time.time() - ACTUAL_RECORDING_START_TIME
    
    running_cameras = sum(1 for p in RUNNING_STREAMS.values() if p.status == "running")
    total_cameras = len([p for p in RUNNING_STREAMS.values() if p.enabled])
    
    return {
        "is_recording": SYNC_RECORDING,
        "duration": int(duration),
        "duration_formatted": f"{int(duration//60)}:{int(duration%60):02d}",
        "running_cameras": running_cameras,
        "total_cameras": total_cameras,
        "logs": RECORDING_LOGS[-20:]  # 返回最近20条日志
    }


# FastAPI应用
app = FastAPI(title="RTSP Stream Processor")

# 设置静态文件和模板
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

# 确保目录存在
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

# 挂载静态文件和模板
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """渲染主仪表盘页面"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/streams")
async def get_streams():
    """获取所有流的状态"""
    return {
        "streams": [processor.get_status() for processor in RUNNING_STREAMS.values()],
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.post("/api/streams/{stream_name}/start")
async def start_stream(stream_name: str):
    """启动特定视频流"""
    if stream_name in RUNNING_STREAMS:
        processor = RUNNING_STREAMS[stream_name]
        threading.Thread(target=processor.start).start()
        return {"status": "starting", "name": stream_name}
    return {"error": "Stream not found"}


@app.post("/api/streams/{stream_name}/stop")
async def stop_stream(stream_name: str):
    """停止特定视频流"""
    if stream_name in RUNNING_STREAMS:
        processor = RUNNING_STREAMS[stream_name]
        result = processor.stop()
        return {"status": "stopped" if result else "unchanged", "name": stream_name}
    return {"error": "Stream not found"}


@app.post("/api/streams/{stream_name}/toggle")
async def toggle_stream(stream_name: str):
    """切换视频流开关状态"""
    if stream_name in RUNNING_STREAMS:
        processor = RUNNING_STREAMS[stream_name]
        if processor.status == "running":
            processor.stop()
            return {"status": "stopped", "name": stream_name}
        else:
            threading.Thread(target=processor.start).start()
            return {"status": "starting", "name": stream_name}
    return {"error": "Stream not found"}


@app.post("/api/streams/start-all")
async def start_all_streams():
    """启动所有摄像头流"""
    success_count = 0
    failed_streams = []
    
    for name, processor in RUNNING_STREAMS.items():
        try:
            if processor.enabled and processor.status != "running":
                threading.Thread(target=processor.start).start()
                success_count += 1
            elif processor.status == "running":
                # 已经在运行，也算成功
                success_count += 1
        except Exception as e:
            failed_streams.append({"name": name, "error": str(e)})
            logging.error(f"启动摄像头 {name} 失败: {e}")
    
    result = {
        "status": "completed",
        "success_count": success_count,
        "failed_streams": failed_streams,
        "total_streams": len(RUNNING_STREAMS),
        "message": f"启动摄像头结果: 成功 {success_count}/{len(RUNNING_STREAMS)}"
    }
    
    logging.info(f"启动所有摄像头: 成功 {success_count}/{len(RUNNING_STREAMS)}")
    if failed_streams:
        logging.warning(f"失败的摄像头: {[s['name'] for s in failed_streams]}")
    
    return result


@app.post("/api/streams/stop-all")
async def stop_all_streams():
    """关闭所有摄像头流"""
    success_count = 0
    failed_streams = []
    
    for name, processor in RUNNING_STREAMS.items():
        try:
            if processor.status == "running":
                if processor.stop():
                    success_count += 1
                else:
                    failed_streams.append({"name": name, "error": "停止失败"})
            else:
                # 已经停止，也算成功
                success_count += 1
        except Exception as e:
            failed_streams.append({"name": name, "error": str(e)})
            logging.error(f"关闭摄像头 {name} 失败: {e}")
    
    result = {
        "status": "completed",
        "success_count": success_count,
        "failed_streams": failed_streams,
        "total_streams": len(RUNNING_STREAMS),
        "message": f"关闭摄像头结果: 成功 {success_count}/{len(RUNNING_STREAMS)}"
    }
    
    logging.info(f"关闭所有摄像头: 成功 {success_count}/{len(RUNNING_STREAMS)}")
    if failed_streams:
        logging.warning(f"失败的摄像头: {[s['name'] for s in failed_streams]}")
    
    return result


@app.get("/api/config")
async def get_config():
    """获取当前配置"""
    return load_config()


@app.post("/api/config")
async def update_config(config: dict):
    """更新配置"""
   
    for processor in RUNNING_STREAMS.values():
        processor.stop()

    
    save_config(config)

   
    RUNNING_STREAMS.clear()
    initialize_streams()

    # Remove automatic start - let user start manually
    # start_streams()

    return {"status": "Configuration updated"}


@app.post("/api/recording/start")
async def start_recording():
    """开始同步录制"""
    success = start_sync_recording()
    return {
        "status": "success" if success else "failed",
        "recording_status": get_recording_status()
    }


@app.post("/api/recording/stop")
async def stop_recording():
    """停止同步录制"""
    success = stop_sync_recording()
    return {
        "status": "success" if success else "failed",
        "recording_status": get_recording_status()
    }


@app.get("/api/recording/status")
async def recording_status():
    """获取录制状态"""
    return get_recording_status()


@app.get("/api/recording/logs")
async def get_logs():
    """获取录制日志"""
    return {"logs": RECORDING_LOGS}


# Initialize when script starts
if __name__ == "__main__":
   
    initialize_streams()

    # Remove automatic start - streams will only start when user clicks buttons
    # start_streams()

    
    uvicorn.run(app, host="0.0.0.0", port=8001)
