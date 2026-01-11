"""
消息推送工具模块

提供 PushPlus 微信消息推送和数字格式化功能。
"""

import logging
from typing import Optional, Dict, Any


def send_pushplus_msg(
    token: str,
    title: str,
    content: str,
    template: str = "html",
    topic: Optional[str] = None,
    channel: Optional[str] = None,
    timeout: float = 30.0,
    max_retries: int = 3
) -> bool:
    """
    使用 PushPlus 发送微信消息
    
    通过 PushPlus (https://www.pushplus.plus/) 将消息推送到微信。
    支持 HTML、Markdown 等多种格式。
    
    Parameters
    ----------
    token : str
        PushPlus 的用户 token，在官网注册后获取
    title : str
        消息标题
    content : str
        消息内容（支持 HTML/Markdown 格式）
    template : str, optional
        模板类型，可选：
        - 'html': HTML 格式（默认）
        - 'txt': 纯文本
        - 'json': JSON 格式
        - 'markdown': Markdown 格式
    topic : Optional[str]
        群组编码（如使用 PushPlus 的 topic 群组推送）
    channel : Optional[str]
        推送通道（PushPlus 支持的 channel 参数）
    timeout : float
        请求超时（秒）
    max_retries : int
        最大重试次数（网络异常/5xx 时重试）
    
    Returns
    -------
    bool
        发送是否成功
    
    Examples
    --------
    >>> token = "your_pushplus_token"
    >>> send_pushplus_msg(token, "交易提醒", "<h1>今日需买入 5 只股票</h1>")
    True
    
    Notes
    -----
    - PushPlus 免费版每天限制 200 条消息
    - 如果 token 为空或无效，函数会记录警告并返回 False
    - 网络异常不会导致程序崩溃
    """
    import requests
    
    # 检查 token
    if not token or token.strip() == "":
        logging.warning("PushPlus token 未配置，跳过消息推送")
        return False
    
    # 优先 https；部分网络环境会屏蔽/重定向 http
    urls = [
        "https://www.pushplus.plus/send",
        "http://www.pushplus.plus/send",
    ]

    payload: Dict[str, Any] = {
        "token": token.strip(),
        "title": title,
        "content": content,
        "template": template
    }
    if topic:
        payload["topic"] = topic
    if channel:
        payload["channel"] = channel
    
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "QuantBot/1.0",
        }

        last_err: Optional[str] = None
        for attempt in range(1, max(1, int(max_retries)) + 1):
            for url in urls:
                try:
                    # PushPlus 支持 application/json；如遇网关兼容问题可改为 data=payload
                    response = session.post(url, json=payload, headers=headers, timeout=float(timeout))
                    if response.status_code >= 500:
                        last_err = f"HTTP {response.status_code}"
                        continue

                    try:
                        result = response.json()
                    except Exception:
                        text = (response.text or "").strip()
                        last_err = f"响应非JSON (HTTP {response.status_code}): {text[:200]}"
                        continue

                    if result.get("code") == 200:
                        logging.info(f"PushPlus 消息发送成功: {title}")
                        return True

                    error_msg = result.get("msg", "未知错误")
                    last_err = f"{error_msg} (code={result.get('code')})"
                except requests.exceptions.Timeout:
                    last_err = "请求超时"
                except requests.exceptions.RequestException as e:
                    last_err = f"网络异常: {e}"

            # 重试
            if attempt < max(1, int(max_retries)):
                continue

        logging.error(f"PushPlus 消息发送失败: {last_err or '未知错误'}")
        return False
            
    except requests.exceptions.Timeout:
        logging.error("PushPlus 请求超时")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"PushPlus 网络请求异常: {e}")
        return False
    except ValueError as e:
        logging.error(f"PushPlus 响应解析失败: {e}")
        return False
    except Exception as e:
        logging.error(f"PushPlus 发送异常: {e}")
        return False


def format_number(
    value: float,
    precision: int = 2,
    as_percentage: bool = False
) -> str:
    """
    格式化数字
    
    Parameters
    ----------
    value : float
        数值
    precision : int, optional
        小数位数，默认2
    as_percentage : bool, optional
        是否格式化为百分比，默认False
    
    Returns
    -------
    str
        格式化后的字符串
    """
    if as_percentage:
        return f"{value * 100:.{precision}f}%"
    
    if abs(value) >= 1e8:
        return f"{value / 1e8:.{precision}f}亿"
    elif abs(value) >= 1e4:
        return f"{value / 1e4:.{precision}f}万"
    else:
        return f"{value:.{precision}f}"

