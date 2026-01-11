"""
消息推送工具模块

提供 PushPlus 微信消息推送和数字格式化功能。
"""

import logging


def send_pushplus_msg(
    token: str,
    title: str,
    content: str,
    template: str = "html"
) -> bool:
    """
    使用 PushPlus 发送微信消息
    
    通过 PushPlus (http://www.pushplus.plus/) 将消息推送到微信。
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
    
    url = "http://www.pushplus.plus/send"
    
    payload = {
        "token": token.strip(),
        "title": title,
        "content": content,
        "template": template
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        result = response.json()
        
        if result.get("code") == 200:
            logging.info(f"PushPlus 消息发送成功: {title}")
            return True
        else:
            error_msg = result.get("msg", "未知错误")
            logging.error(f"PushPlus 消息发送失败: {error_msg}")
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

