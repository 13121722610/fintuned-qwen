# web_launcher.py - 修改版
import os
import sys
import subprocess
import socket
import time
from datetime import datetime

def find_available_port(start_port=8501, max_attempts=10):
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result != 0:
                return port
        except:
            continue
    return start_port

def main():
    """启动Web应用"""
    print("=" * 70)
    print("🚀 医疗问答网页应用启动器")
    print("=" * 70)
    
    # 项目信息
    project_root = "/amax/home/yhji/LM-Course"
    web_app_file = os.path.join(project_root, "medical_web_app.py")
    
    if not os.path.exists(web_app_file):
        print(f"❌ 找不到Web应用文件: {web_app_file}")
        print("请确保 medical_web_app.py 在项目根目录")
        return
    
    print(f"📁 项目目录: {project_root}")
    print(f"🌐 Web应用: {web_app_file}")
    
    # 检查依赖
    print("\n🔍 检查Python依赖...")
    try:
        import streamlit
        import torch
        print(f"✅ Streamlit: {streamlit.__version__}")
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install streamlit transformers peft accelerate")
        return
    
    # 查找可用端口
    port = find_available_port(8501)
    print(f"🔌 使用端口: {port}")
    
    # 获取本地IP地址
    try:
        # 获取本地主机名对应的IP地址
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    
    # 构建URL
    local_url = f"http://localhost:{port}"
    network_url = f"http://{local_ip}:{port}"
    
    print("\n" + "=" * 70)
    print("🌐 应用已启动！请手动复制以下链接访问：")
    print("\n" + "─" * 50)
    print(f"🔗 本地访问链接：")
    print(f"   {local_url}")
    print(f"\n🔗 网络访问链接：")
    print(f"   {network_url}")
    print("─" * 50)
    print("\n💡 提示：")
    print("   1. 复制以上链接到浏览器中打开")
    print("   2. 首次加载模型可能需要几分钟")
    print("   3. 按 Ctrl+C 停止服务器")
    print("=" * 70)
    
    # 启动Streamlit
    print("\n⏳ 正在启动Streamlit服务器...")
    time.sleep(1)
    
    try:
        # 切换到项目目录
        os.chdir(project_root)
        
        # 启动Streamlit（不自动打开浏览器）
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "medical_web_app.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--theme.base", "light",
            "--browser.serverAddress", "localhost",
            "--server.headless", "true"  # 不在启动时自动打开浏览器
        ]
        
        print("✅ 服务器启动成功！")
        print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📋 访问链接已复制到剪贴板，请手动粘贴到浏览器访问。")
        print("\n" + "=" * 70)
        
        # 尝试将链接复制到剪贴板（可选功能）
        try:
            import pyperclip
            pyperclip.copy(local_url)
            print("📋 本地链接已复制到剪贴板")
        except:
            print("⚠️  剪贴板复制失败，请手动复制链接")
        
        print("=" * 70)
        print("\n🔄 服务器日志:")
        print("-" * 50)
        
        # 运行Streamlit
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出日志
        try:
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
        except KeyboardInterrupt:
            print("\n\n👋 正在停止服务器...")
            process.terminate()
            process.wait()
            print("✅ 服务器已停止")
        except Exception as e:
            print(f"\n❌ 服务器异常: {e}")
            process.terminate()
            
    except KeyboardInterrupt:
        print("\n\n👋 启动被用户中断")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()